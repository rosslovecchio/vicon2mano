#!/usr/bin/env python3
"""Fit MANO so the markers sit ON THE MESH SURFACE, not at joint centres.

Why the existing plots look wrong
---------------------------------
Every fit in this repo so far minimises ``||J_j(theta,beta) - marker||^2``:
the *joint centre* is pulled onto the marker. A joint centre is inside the
hand, so a perfect fit by that objective necessarily buries each marker
roughly one half-thickness (10-25mm) below the skin -- exactly what
``static_fit.html`` shows. It is not a rendering artifact and not a
labelling problem; the objective is asking for the wrong thing.

``refit_with_marker_offsets.py`` already corrects this *post hoc* with a
per-marker offset in a bone-segment frame. This script takes the direct
route instead, the standard one for optical markers on a parametric body
model (SMPL/MANO "virtual markers"): attach each marker to a point on the
*mesh*, and fit that mesh point to the marker.

Method
------
1. Joint-based fit, as today, to get a plausible starting pose + betas.
2. Attach each used marker to its nearest mesh vertex (778 MANO verts) at
   that pose -- a fixed vertex id per marker, valid because a marker is
   taped to one spot of skin.
3. Re-optimise betas/pose/orient/transl against ``||V[v_m] - marker||^2``.
4. Repeat 2-3 a few times (iterated closest point): the attachment is only
   as good as the pose it was chosen at, and it stabilises in 2-3 rounds.

The reported number to judge is **marker-to-surface distance** (signed:
positive = outside the mesh, negative = buried inside). Joint-to-marker
residual is deliberately *not* the success metric here -- this objective is
no longer trying to make those equal.

Usage
-----
python scripts/mano/fit_surface_markers.py \\
    --static "<static CSV>" \\
    --out results/mano/P10_Trial2_handsonly/eval/static_fit_surface.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inspect as _inspect
import numpy as _np
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
for _attr, _builtin in {"int": int, "float": float, "bool": bool, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _builtin)

from vicon2mano.core.loader import load_csv
from vicon2mano.core.fitter import FitConfig, MANOFitter, mano_output_to_joints21
from vicon2mano.core.correspondence import MANO_JOINT_NAMES, label_seed


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for a triangle mesh."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(n, 1e-12)


# repo 21-joint slot -> smplx/MANO bone index (the inverse of fitter.py's
# _REPO_JOINT_SLOTS/_SMPLX_JOINT_ORDER pairing)
_REPO_TO_BONE = {0: 0, 1: 13, 2: 14, 3: 15, 5: 1, 6: 2, 7: 3,
                 9: 4, 10: 5, 11: 6, 13: 10, 14: 11, 15: 12,
                 17: 7, 18: 8, 19: 9}
# MANO kinematic parents, smplx bone order
_BONE_PARENT = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 9: 8,
                10: 0, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14}


def allowed_faces(lbs, F, repo_joint, min_own=0.0):
    """Faces whose skin belongs to the bone a marker is taped to.

    Without this the nearest-point search is free to attach, say, an
    Index3 marker to a neighbouring finger or to the palm, which lets the
    pose stay wrong while the *attachment* absorbs the error -- the curled
    fingers under extended markers. A marker is physically taped to one
    segment, so restrict it to that segment's skin (and its parent's, since
    skin over a joint is blended between the two).
    """
    dom = np.asarray(lbs).argmax(1)                       # (778,) dominant bone
    if repo_joint < 0:                                    # surface-only: palm
        ok_bones = {0}
    else:
        # ONLY the marker's own segment -- the one DISTAL to its joint.
        # Allowing the parent segment too (as this did) is anatomically
        # reasonable, since skin over a joint is blended between both, but it
        # destroys observability: a joint does not move the segment proximal
        # to it, so a marker that drifts onto the parent's skin cannot
        # constrain that joint at all. Measured consequence: all four PIP
        # markers sat 82-97% on the proximal phalanx, so PIP angle was
        # unconstrained and the MCP absorbed its motion (index MCP fitted
        # 51 deg against the markers' 42.7, PIP 25 against 34.7).
        # REVERTED to {own, parent} after measuring the alternative: {own}
        # alone makes each joint observable in principle, but forces the
        # attachment onto skin that can be well distal of where the marker
        # physically sits (at the joint crease), so the fit carries a built-in
        # offset and simply stops articulating that joint -- middle PIP/DIP
        # went to -2.3/-2.2 deg and overall residual 17.7 -> 20.3mm.
        # Observability has to come from somewhere other than the attachment.
        b = _REPO_TO_BONE[repo_joint]
        ok_bones = {b, _BONE_PARENT[b]}
    ok_v = np.isin(dom, list(ok_bones))
    # A marker must attach to skin its OWN joint actually moves. Measured:
    # Index1's chosen face was 89.5% skinned to the palm and 7% to the index
    # proximal bone, so rotating the index MCP barely moved it -- the MCP
    # angle was left unconstrained by its own marker, and the error showed up
    # compounded at Index3. Requiring a minimum own-bone weight stops the
    # search taking the palm-side face at a joint boundary.
    if min_own > 0 and repo_joint >= 0:
        b = _REPO_TO_BONE[repo_joint]
        if b != 0:
            own = np.asarray(lbs)[:, b]
            ok_v = ok_v & (own >= min_own)
    cand = np.flatnonzero(ok_v[F].all(1))
    if len(cand) == 0:            # never leave a marker with nothing to attach to
        ok_v = np.isin(dom, list(ok_bones))
        cand = np.flatnonzero(ok_v[F].all(1))
    return cand


# --- biomechanical joint limits -------------------------------------------
# MANO's hand_pose is 15 joints x 3 axis-angle components, in the order
# index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3. Which component does what
# was measured, not assumed (perturb one component, see which way the
# fingertip moves relative to the palm frame):
#   axis 2 = flexion/extension  (|dot| ~0.99 with the palm normal; +z EXTENDS)
#   axis 1 = abduction/adduction (~0.9 with the ulnar direction)
#   axis 0 = twist/mixed -- anatomically ~0 for a finger joint
# Ranges below are ordinary anatomy: PIP/DIP are hinges (flexion only, no
# hyperextension), MCP adds abduction and a little hyperextension.
_FINGER_JOINTS = ["index1", "index2", "index3", "middle1", "middle2", "middle3",
                  "pinky1", "pinky2", "pinky3", "ring1", "ring2", "ring3",
                  "thumb1", "thumb2", "thumb3"]
# (twist_lo, twist_hi), (abduct_lo, abduct_hi), (flex_lo, flex_hi), radians.
# Flexion is NEGATIVE about axis 2, hence the asymmetric bounds.
_MCP_LIMITS = ((-0.15, 0.15), (-0.35, 0.35), (-1.60, 0.30))
_PIP_LIMITS = ((-0.10, 0.10), (-0.15, 0.15), (-1.90, 0.05))
_DIP_LIMITS = ((-0.10, 0.10), (-0.15, 0.15), (-1.50, 0.05))
# The thumb's CMC/MCP/IP are not finger joints and its axes are less cleanly
# separated than the fingers'. Measured loadings (perturb one component, watch
# the thumb tip): axis 0 -> flexion ~-0.91, axis 1 -> twist ~-0.83, axis 2 ->
# flexion ~-0.96. So axis 1 is the one with no anatomical business moving, and
# axes 0 and 2 share the flexion that the three joints each allow in differing
# amounts. Tightened per joint rather than sharing one loose triple:
#   CMC  (thumb1): the mobile one -- real flexion AND abduction
#   MCP  (thumb2): mostly flexion, little else
#   IP   (thumb3): a hinge, like a finger DIP
_THUMB_LIMITS = {
    "thumb1": ((-0.60, 0.60), (-0.25, 0.25), (-1.00, 0.50)),
    "thumb2": ((-0.40, 0.40), (-0.15, 0.15), (-1.10, 0.20)),
    "thumb3": ((-0.25, 0.25), (-0.10, 0.10), (-1.40, 0.10)),
}


def joint_limit_bounds():
    """(45,) lower and upper bounds on the raw axis-angle pose."""
    lo, hi = np.empty(45, np.float32), np.empty(45, np.float32)
    for ji, name in enumerate(_FINGER_JOINTS):
        if name.startswith("thumb"):
            lim = _THUMB_LIMITS[name]
        elif name.endswith("1"):
            lim = _MCP_LIMITS
        elif name.endswith("2"):
            lim = _PIP_LIMITS
        else:
            lim = _DIP_LIMITS
        for ax in range(3):
            lo[ji * 3 + ax], hi[ji * 3 + ax] = lim[ax]
    return lo, hi


def joint_limit_penalty(hp, lo, hi):
    """Squared hinge loss for any pose component outside its range."""
    import torch
    return (torch.relu(hp - hi) ** 2 + torch.relu(lo - hp) ** 2).sum(-1).mean()


_JOINT_TO_REPO = {"index1": 5, "index2": 6, "index3": 7,
                  "middle1": 9, "middle2": 10, "middle3": 11,
                  "pinky1": 17, "pinky2": 18, "pinky3": 19,
                  "ring1": 13, "ring2": 14, "ring3": 15,
                  "thumb1": 1, "thumb2": 2, "thumb3": 3}
# the joint at the far end of each segment, giving its long axis
_SEGMENT_CHILD = {"index1": 6, "index2": 7, "index3": 8,
                  "middle1": 10, "middle2": 11, "middle3": 12,
                  "pinky1": 18, "pinky2": 19, "pinky3": 20,
                  "ring1": 14, "ring2": 15, "ring3": 16,
                  "thumb1": 2, "thumb2": 3, "thumb3": 4}
_TIP_OF_JOINT = {"index": 8, "middle": 12, "pinky": 20, "ring": 16, "thumb": 4}


def joint_axis_basis(model, dev, pose_dim):
    """Per-joint orthonormal basis (flexion, abduction, twist) in POSE space.

    Built from ROTATION AXES, not tip displacements. The earlier version
    derived flexion from the fingertip Jacobian and then took twist as
    ``cross(flex, abd)`` -- whatever was left over. That is only meaningful
    when flexion and abduction separate cleanly, and for the middle finger
    they do not (its flexion direction moves the tip 0.78 palmward but 0.59
    sideways, against ~0.97/0.25 elsewhere). Its "twist" therefore contained
    real flexion, and pinning twist to zero dragged the finger straight:
    middle PIP went 9.5 -> -3.4 deg and DIP 5.7 -> -8.2 with flexion
    explicitly unbounded, so only the torsion penalty could have done it.

    A small pose change about axis ``a`` moves a point at offset ``r`` from
    the joint by ``a x r``, so the world rotation axis each pose component
    produces can be recovered by least squares from the displacements of
    several vertices on that segment. With the pose->axis map ``M`` in hand,
    the anatomical axes are unambiguous:

        twist     = the segment's own long axis            u
        abduction = the palm normal                        p
        flexion   = cross(u, p)   (rotation about it sends the tip palmward)

    and the pose-space basis is ``M^-1`` applied to each.
    """
    import torch
    with torch.no_grad():
        o0 = model(global_orient=torch.zeros(1, 3, device=dev),
                   hand_pose=torch.zeros(1, pose_dim, device=dev),
                   betas=torch.zeros(1, 10, device=dev), return_verts=True)
        J0 = mano_output_to_joints21(o0)[0].cpu().numpy()
        V0 = o0.vertices[0].cpu().numpy()
    nrm0 = np.cross(J0[5] - J0[0], J0[17] - J0[0])
    nrm0 /= np.linalg.norm(nrm0)
    # thumb sits palmar of the MCP plane in MANO's rest pose -- pins the sign
    palmward = nrm0 if (J0[4] - J0[0]) @ nrm0 > 0 else -nrm0

    lbs = model.lbs_weights.detach().cpu().numpy()
    dom = lbs.argmax(1)
    eps = 1e-3
    B = np.zeros((15, 3, 3), np.float32)
    for ji, name in enumerate(_FINGER_JOINTS):
        bone = _REPO_TO_BONE[_JOINT_TO_REPO[name]]
        vsel = np.flatnonzero(dom == bone)
        if len(vsel) < 4:
            vsel = np.flatnonzero(dom == bone) if len(vsel) else np.arange(len(V0))
        cen = J0[_JOINT_TO_REPO[name]]
        R = V0[vsel] - cen                                  # (n,3) lever arms
        M = np.zeros((3, 3), np.float32)
        for ax in range(3):
            hp = np.zeros((1, pose_dim), np.float32)
            hp[0, ji * 3 + ax] = eps
            with torch.no_grad():
                o = model(global_orient=torch.zeros(1, 3, device=dev),
                          hand_pose=torch.tensor(hp, device=dev),
                          betas=torch.zeros(1, 10, device=dev), return_verts=True)
                D = (o.vertices[0].cpu().numpy()[vsel] - V0[vsel]) / eps
            # solve D_i = a x R_i for a, least squares over all vertices
            A = np.zeros((3 * len(vsel), 3), np.float32)
            for i, r in enumerate(R):
                A[3 * i:3 * i + 3] = np.array([[0, r[2], -r[1]],
                                               [-r[2], 0, r[0]],
                                               [r[1], -r[0], 0]], np.float32)
            M[:, ax] = np.linalg.lstsq(A, D.reshape(-1), rcond=None)[0]
        # anatomical rotation axes, in world
        child = _SEGMENT_CHILD[name]
        u = J0[child] - cen
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-9 else np.array([1.0, 0, 0], np.float32)
        p = palmward - (palmward @ u) * u
        np_ = np.linalg.norm(p)
        p = p / np_ if np_ > 1e-9 else np.cross(u, [0, 0, 1.0])
        a_flex = np.cross(u, p)
        Minv = np.linalg.pinv(M)
        cols = []
        for a_w in (a_flex, p, u):
            v = Minv @ a_w
            n = np.linalg.norm(v)
            cols.append(v / n if n > 1e-9 else v)
        B[ji] = np.stack(cols, axis=1)
    return B


# Anatomical ranges, now expressed in the MEASURED basis above rather than
# per raw component: (flex_lo, flex_hi), (abduct_lo, abduct_hi), (twist...).
# Flexion is positive in this basis by construction.
# Finger torsion is pinned to exactly zero: the interphalangeal and
# metacarpophalangeal joints have no axial rotation worth modelling, and any
# the optimiser introduces is fitting noise, not anatomy. A (0, 0) range makes
# the hinge penalty quadratic in the twist itself.
_LIMITS_BASIS = {
    "mcp":   ((-0.30, 1.60), (-0.40, 0.40), (0.0, 0.0)),
    "pip":   ((-0.05, 1.90), (-0.20, 0.20), (0.0, 0.0)),
    "dip":   ((-0.05, 1.50), (-0.15, 0.15), (0.0, 0.0)),
    "thumb1": ((-0.50, 1.20), (-0.60, 0.60), (-0.30, 0.30)),
    "thumb2": ((-0.20, 1.20), (-0.30, 0.30), (-0.20, 0.20)),
    "thumb3": ((-0.10, 1.40), (-0.20, 0.20), (-0.15, 0.15)),
}


def basis_limit_bounds(torsion_only=False, free_flexion=False):
    """(15,3) lo/hi in the measured per-joint basis."""
    lo = np.zeros((15, 3), np.float32)
    hi = np.zeros((15, 3), np.float32)
    for ji, name in enumerate(_FINGER_JOINTS):
        key = name if name.startswith("thumb") else (
            "mcp" if name.endswith("1") else "pip" if name.endswith("2") else "dip")
        for c in range(3):
            lo[ji, c], hi[ji, c] = _LIMITS_BASIS[key][c]
    if free_flexion:
        # Bound abduction and torsion, leave FLEXION unbounded. Flexion
        # genuinely needs range beyond textbook (markers show MCP flexion up
        # to 128 deg here) and bounding it collapsed the fingers. Abduction
        # does not: measured on the whole trial its p99 is 22-24 deg, right at
        # the anatomical limit, but the maxima reach 66-100 deg -- the
        # optimiser splaying a finger sideways THROUGH its neighbour to reach
        # a marker. Fingertips come out in the wrong order in 8.8% of frames
        # and the index crosses the middle in 1.5%.
        lo[:, 0], hi[:, 0] = -10.0, 10.0      # flexion: unbounded
    elif torsion_only:
        # Keep ONLY the zero-torsion constraint; open the flexion and
        # abduction bounds wide. Measured: applying all three together
        # collapsed the middle finger's PIP/DIP back to the extension bound
        # (9.5 deg -> -2.2) and pushed the finger residual 13.2 -> 21.7mm, so
        # the two kinds of limit have to be separable to tell which is at
        # fault rather than re-entangling them.
        lo[:, 0], hi[:, 0] = -10.0, 10.0      # flexion: unbounded
        lo[:, 1], hi[:, 1] = -10.0, 10.0      # abduction: unbounded
    return lo, hi


def basis_limit_penalty(hp, Bt, lo, hi, twist_w=1.0):
    """Hinge penalty on the projection onto each joint's measured basis.

    ``twist_w`` scales the twist column only, so axial rotation can be
    suppressed hard without making the flexion and abduction bounds equally
    stiff -- those are real ranges a finger should be free to use.
    """
    import torch
    T = hp.shape[0]
    hpj = hp.view(T, 15, 3)
    c = torch.einsum("tjk,jkc->tjc", hpj, Bt)      # project into the basis
    pen = torch.relu(c - hi) ** 2 + torch.relu(lo - c) ** 2
    if twist_w != 1.0:
        w = torch.ones(3, device=pen.device)
        w[2] = twist_w
        pen = pen * w
    return pen.sum((-1, -2)).mean()


def _interp_gaps(th):
    """Fill NaN entries of a per-frame angle series by linear interpolation
    in time (nearest value past either end); all-NaN stays NaN."""
    ok = np.isfinite(th)
    if ok.all() or not ok.any():
        return th
    t = np.arange(len(th))
    return np.interp(t, t[ok], th[ok]).astype(th.dtype)


def angles_from_markers(mk, names, Bmat, gap_fill=True):
    """Initial hand_pose read straight off the markers, per frame.

    Where a finger's angle is unobservable in a frame (one of its markers,
    or Palm2, missing) it is interpolated in time from the frames either side
    (``gap_fill``). It used to become 0 -- a straight finger -- and a straight
    start got stuck: on SOMA-labelled P10/Trial2 (raw gaps kept) the fit left
    pinky/ring at ~0-13 deg PIP for hundreds of frames where the real markers
    show ~90-100 deg, while the same markers gap-filled fitted correctly, and
    the temporal smoothness spread the straight pose to neighbouring frames.

    The fit starts every frame from a flat hand, so a pinch has to travel a
    long way with nothing guiding it -- a reliable way to find a local
    minimum. But the markers already carry the flexion angles: with MCP, PIP
    and nailbed markers on each finger, the two segment directions give the
    MCP and PIP flexion directly.

        theta_mcp = angle between the palm's distal axis and MCP->PIP
        theta_pip = angle between MCP->PIP and PIP->nailbed

    The DIP is not separately observable -- the nailbed marker sits on the
    distal phalanx, so PIP->nailbed spans both joints -- so it is seeded with
    the usual DIP ~ 2/3 PIP coupling. This is an INITIALISATION, not a
    constraint: the optimiser is free to move away from it.

    Angles are converted to pose space through each joint's measured flexion
    direction (``Bmat[:, :, 0]``), so this does not re-make the axis-role
    mistake the limits originally had.

    mk: (T, M, 3) marker positions in mm. Returns (T, 45).
    """
    T = mk.shape[0]
    hp = np.zeros((T, 45), np.float32)
    idx = {n: i for i, n in enumerate(names)}
    need = ("Palm1", "Palm2", "Palm3")
    if not all(n in idx for n in need):
        return hp
    p1, p2, p3 = (mk[:, idx[n]] for n in need)
    distal = np.zeros((T, 3), np.float32)
    for fname, jbase in (("Index", 0), ("Middle", 3), ("Pinky", 6), ("Ring", 9)):
        m1, m2, m3 = (f"{fname}1", f"{fname}2", f"{fname}3")
        if not all(n in idx for n in (m1, m2, m3)):
            continue
        a, b, c = mk[:, idx[m1]], mk[:, idx[m2]], mk[:, idx[m3]]
        # palm distal axis: wrist(Palm2) -> this finger's MCP marker
        d0 = a - p2
        v1 = b - a                      # proximal phalanx
        v2 = c - b                      # middle+distal
        def _u(v):
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            return v / np.where(n < 1e-9, 1.0, n)
        d0, v1u, v2u = _u(d0), _u(v1), _u(v2)
        th_mcp = np.arccos(np.clip((d0 * v1u).sum(-1), -1, 1))
        th_pip = np.arccos(np.clip((v1u * v2u).sum(-1), -1, 1))
        if gap_fill:
            th_mcp, th_pip = _interp_gaps(th_mcp), _interp_gaps(th_pip)
        th_mcp = np.nan_to_num(th_mcp)
        th_pip = np.nan_to_num(th_pip)
        for off, th in ((0, th_mcp), (1, th_pip), (2, th_pip * (2.0 / 3.0))):
            ji = jbase + off
            hp[:, ji * 3:ji * 3 + 3] = th[:, None] * Bmat[ji, :, 0][None, :]
    return hp


# marker-name suffix -> (governing joint index in _FINGER_JOINTS,
#                        proximal repo joint, distal repo joint of its segment)
# (marker repo joint, proximal, distal) for the segment-frame offset model
_OFFSET_SEG = []     # filled per run from the fitted marker set

_STRETCH_SEG = {
    "Index2": (0, 5, 6), "Index3": (1, 6, 7),
    "Middle2": (3, 9, 10), "Middle3": (4, 10, 11),
    "Pinky2": (6, 17, 18), "Pinky3": (7, 18, 19),
    "Ring2": (9, 13, 14), "Ring3": (10, 14, 15),
}


def stretch_offsets(J21, hp, Bt, names, slopes, ref_deg=None, gate=45.0):
    """Distal slide of each attachment, as a function of fitted flexion.

    A marker is taped to one material point of skin, but the dorsal surface
    over a flexed knuckle is longer than over an extended one, so that point
    travels distally along the segment as the joint flexes. Measured on
    P10/Trial2: MCP->PIP marker spacing grows 7-10mm (~20%) from extension to
    full flexion, correlating +0.80..+0.85 with flexion, while the palm
    control grows 1.3mm. MANO's linear blend skinning does not reproduce that,
    so a FIXED attachment is structurally unable to follow the marker.

    Returns (T, P, 3) metres to add to each attachment point: the calibrated
    slope times the joint's current flexion, along the segment's own axis.
    Differentiable in the pose, so the slide tracks the fit rather than being
    baked in from a prior estimate.
    """
    import torch
    T = J21.shape[0]
    off = torch.zeros((T, len(names), 3), device=J21.device, dtype=J21.dtype)
    hpj = hp.view(T, 15, 3)
    for k, nm in enumerate(names):
        if nm not in _STRETCH_SEG or not slopes.get(nm):
            continue
        ji, a, b = _STRETCH_SEG[nm]
        # flexion of the governing joint, in degrees, from the measured basis
        th = torch.einsum("tk,k->t", hpj[:, ji], Bt[ji, :, 0]) * (180.0 / np.pi)
        # relative to the flexion already present in the static, where the
        # attachment was calibrated -- the static is NOT an extended hand
        # (19-28 deg of MCP flexion), so measuring the slide from zero
        # double-counts the stretch the attachment already encodes
        if ref_deg:
            th = th - float(ref_deg.get(nm, 0.0))
        if gate > 0:
            # Gated: no correction below `gate` degrees of flexion. Measured
            # both ways -- the slide helps where the stretch actually exists
            # (fist window, fingers 21.01 -> 20.11mm) and hurts where it does
            # not (pinch window, 6.65 -> 7.14), because there the correction is
            # ~1mm and competes with noise. The static's own flexion (19-28 deg)
            # sits below the gate, so relu(ref - gate) is zero and the
            # reference subtraction above is left undisturbed.
            th = torch.relu(th + float(ref_deg.get(nm, 0.0)) - gate)
        u = J21[:, b] - J21[:, a]
        u = u / torch.sqrt((u ** 2).sum(-1, keepdim=True) + 1e-18)
        off[:, k] = (slopes[nm] * 1e-3 * th).unsqueeze(-1) * u
    return off


def segment_frames_t(J21):
    """(T, P, 3, 3) per-joint frames aligned to each joint's own bone, in torch.

    A torch port of strategies.mano.relabel._segment_frames, so the
    offset-target mode can be differentiated through the pose exactly as the
    surface mode is. x runs along the bone (joint -> child); the parent
    direction fixes the roll.
    """
    import torch
    T = J21.shape[0]
    out = []
    for j, a, b in _OFFSET_SEG:
        x = J21[:, b] - J21[:, a]
        x = x / torch.sqrt((x ** 2).sum(-1, keepdim=True) + 1e-18)
        ref = J21[:, a] - J21[:, 0]
        ref = ref / torch.sqrt((ref ** 2).sum(-1, keepdim=True) + 1e-18)
        z = torch.cross(x, ref, dim=-1)
        z = z / torch.sqrt((z ** 2).sum(-1, keepdim=True) + 1e-18)
        y = torch.cross(z, x, dim=-1)
        out.append(torch.stack([x, y, z], dim=-1))      # columns = axes
    return torch.stack(out, dim=1)


def calibrate_segment_offsets(J21, M, names, repo_of):
    """Per-marker offset in its joint's segment frame -- the ALTERNATIVE to a
    mesh attachment, ported from scripts/mano/refit_with_marker_offsets.py.

    The marker is modelled as a fixed 3-vector from its joint centre, in a
    frame that rotates with the bone. That is pose-invariant for a rigid
    offset but, unlike a barycentric point on the mesh, it cannot deform with
    shape or with the neighbouring joints, and it knows nothing about the
    surface -- so it cannot tell that a marker has ended up inside the hand or
    on the palmar side.

    Returns (offsets (P,3) metres, spread (P,) mm). A large spread means the
    rigid-offset model does not hold for that marker.
    """
    import torch
    R = segment_frames_t(J21)                       # (T,P,3,3)
    d = M - torch.stack([J21[:, r] for _, r in zip(names, repo_of)], dim=1)
    loc = torch.einsum("tpij,tpj->tpi", R.transpose(-1, -2), d)
    ok = torch.isfinite(loc).all(-1)
    offs, spread = [], []
    for k in range(len(names)):
        v = loc[:, k][ok[:, k]]
        if len(v) < 5:
            offs.append(torch.zeros(3, device=J21.device)); spread.append(0.0); continue
        mu = v.mean(0)
        offs.append(mu)
        spread.append(float((v - mu).norm(dim=-1).mean() * 1000))
    return torch.stack(offs), spread


def onehot_attachment(F, vid):
    """Express "attach to vertex ``vid``" as a (face, barycentric) pair.

    The fit machinery works in face+barycentric terms, so a hand-picked
    vertex is just a degenerate barycentric: any face containing it, with
    all the weight on that corner.
    """
    f = int(np.flatnonzero((F == vid).any(1))[0])
    b = np.zeros(3, dtype=np.float32)
    b[int(np.flatnonzero(F[f] == vid)[0])] = 1.0
    return f, b


def write_vertex_picker(V, F, mk, names, path):
    """Mesh + every vertex id, hoverable, so attachments can be read off."""
    fig = go.Figure(data=[
        go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                  i=F[:, 0], j=F[:, 1], k=F[:, 2],
                  color="#e8a0a0", opacity=0.55, flatshading=True,
                  lighting=dict(ambient=0.8, diffuse=0.4), name="MANO"),
        go.Scatter3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], mode="markers",
                     marker=dict(size=2.5, color="#2563eb"),
                     text=[str(i) for i in range(len(V))],
                     hovertemplate="vertex %{text}<extra></extra>",
                     name="mesh vertices (hover for id)"),
        go.Scatter3d(x=mk[:, 0], y=mk[:, 1], z=mk[:, 2], mode="markers+text",
                     marker=dict(size=6, color="#1b3a5c"), text=names,
                     textposition="top center", name="Vicon markers"),
    ])
    fig.update_layout(title="Hover a blue point to read its vertex id",
                      scene=dict(aspectmode="data",
                                 camera=dict(projection=dict(type="orthographic"))),
                      width=1100, height=900)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"saved vertex picker: {path}")


def dorsal_direction(J, palm_marker, wrist_marker=None):
    """Unit vector pointing out the BACK of the hand (palm-level).

    Every marker in this protocol is taped to the dorsum. Without this, the
    nearest-point search happily attaches a PIP/DIP marker to the *palmar*
    surface: the outward-side test only asks that the marker be outside the
    face, and a palmar face satisfies that whenever the mesh finger has
    drifted above the marker.

    The sign is resolved against a marker known to be dorsal (Palm2), rather
    than assumed from handedness -- a left/right mix-up would otherwise flip
    the whole constraint silently.
    """
    n = np.cross(J[5] - J[0], J[17] - J[0])      # wrist->index MCP x wrist->pinky MCP
    n = n / np.linalg.norm(n)
    ref = (wrist_marker if wrist_marker is not None else J[0])
    return n if n @ (palm_marker - ref) > 0 else -n


# repo joint slot -> (proximal, distal) joint slots of the segment its skin
# belongs to. Used to give each marker its OWN dorsal direction.
_SEGMENT_OF = {1: (0, 1), 2: (1, 2), 3: (2, 3),
               5: (0, 5), 6: (5, 6), 7: (6, 7),
               9: (0, 9), 10: (9, 10), 11: (10, 11),
               13: (0, 13), 14: (13, 14), 15: (14, 15),
               17: (0, 17), 18: (17, 18), 19: (18, 19)}


def segment_dorsals(J, palm_dorsal, repo_joints):
    """Per-marker dorsal direction that rotates with the finger.

    A single palm-level dorsal is fine for an extended static but wrong the
    moment a finger flexes: a flexed distal phalanx has its dorsal side
    rotated up to 90 degrees away from the palm's, so a palm-level test
    rejects the correct faces and the fallback silently attaches palmar.

    Finger flexion is rotation about an axis parallel to the ulnar direction
    (verified by perturbation: axis 2 is flexion, loading 0.99 on the palm
    normal). So ``cross(ulnar, segment_axis)`` tracks the segment's own dorsal
    as it flexes, where a fixed palm normal does not. The sign is pinned once
    against the palm dorsal using the palm itself, which never flexes.
    """
    ulnar = J[17] - J[5]
    ulnar = ulnar / np.linalg.norm(ulnar)
    out = []
    for rj in repo_joints:
        if rj < 0 or rj not in _SEGMENT_OF:
            out.append(palm_dorsal)                   # palm markers
            continue
        a, b = _SEGMENT_OF[rj]
        u = J[b] - J[a]
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            out.append(palm_dorsal)
            continue
        d = np.cross(ulnar, u / nu)
        nd = np.linalg.norm(d)
        if nd < 1e-6:                                 # segment parallel to ulnar
            out.append(palm_dorsal)
            continue
        d = d / nd
        out.append(d if d @ palm_dorsal > 0 else -d)
    return np.asarray(out)


def closest_point_on_faces(V, F, pts, allow=None, dorsal=None, min_dot=0.0):
    """Nearest point on the mesh *surface* for each of ``pts``.

    Vertex-only attachment is limited by MANO's ~8mm vertex spacing, which
    floors the achievable marker-to-surface distance. Returns, per point, the
    face id and its barycentric weights -- a continuous, differentiable
    attachment ("virtual marker"), which is what lets the fit drive the
    distance to the *surface* rather than to the nearest vertex.
    """
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (A + B + C) / 3.0
    fid, bary = [], []
    for pi, p in enumerate(pts):
        pool = np.arange(len(F)) if allow is None else allow[pi]
        # restrict the exact test to the 64 nearest faces by centroid -- the
        # closest point is always on one of them for a mesh this smooth
        order = np.argsort(np.linalg.norm(cen[pool] - p, axis=1))[:64]
        cand = pool[order]
        best, bd, bb = cand[0], np.inf, np.array([1 / 3, 1 / 3, 1 / 3])
        # Fallback used only if no face passes the outward test below.
        fb, fbd, fbb = best, np.inf, bb
        for f in cand:
            a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
            e1, e2 = b - a, c - a
            n = np.cross(e1, e2)
            nn = n @ n
            if nn < 1e-18:
                continue
            w = p - a
            # project onto the triangle's plane, then clamp into the triangle
            u = (np.cross(w, e2) @ n) / nn
            v = (np.cross(e1, w) @ n) / nn
            u, v = max(u, 0.0), max(v, 0.0)
            if u + v > 1:
                u, v = u / (u + v), v / (u + v)
            q = a + u * e1 + v * e2
            d = np.linalg.norm(p - q)
            if d < fbd:
                fb, fbd, fbb = f, d, np.array([1 - u - v, u, v])
            # Side test: the marker must lie on the face's OUTWARD side.
            # Without it, plain nearest-point snaps a dorsal marker through a
            # thin segment onto the palmar surface -- which is what put the
            # PIP markers under the hand. A palmar face sees a dorsal marker
            # on its inward side, so this rejects it.
            if (p - q) @ n <= 0:
                continue
            # Dorsal test: the face must look out the back of the hand. This
            # is what stops a PIP/DIP marker attaching to the palmar surface.
            dvec = None if dorsal is None else (
                dorsal[pi] if np.ndim(dorsal) == 2 else dorsal)
            if dvec is not None and (n / np.sqrt(nn)) @ dvec <= min_dot:
                continue
            if d < bd:
                best, bd, bb = f, d, np.array([1 - u - v, u, v])
        if not np.isfinite(bd):
            best, bb = fb, fbb
        fid.append(best)
        bary.append(bb)
    return np.asarray(fid), np.asarray(bary, dtype=np.float32)


def surface_report(V, F, mk, names, tag):
    """Signed distance marker -> nearest vertex (+ outside / - inside)."""
    Nv = vertex_normals(V, F)
    d = np.linalg.norm(mk[:, None, :] - V[None, :, :], axis=-1)
    vid = d.argmin(1)
    delta = mk - V[vid]
    signed = np.einsum("ij,ij->i", delta, Nv[vid])
    dist = np.linalg.norm(delta, axis=1) * np.sign(np.where(signed == 0, 1, signed))
    print(f"\n--- {tag}: marker-to-surface distance "
          f"(+ outside the mesh, - buried inside) ---")
    for nm, v in zip(names, dist):
        print(f"  {nm:10s} {v:+7.2f} mm")
    print(f"  {'MEAN |d|':10s} {np.abs(dist).mean():7.2f} mm   "
          f"({int((dist < 0).sum())}/{len(dist)} markers inside the mesh)")
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", required=True)
    ap.add_argument("--out", default=None,
                     help="required unless --picker-out is given")
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--exclude", default="",
                     help="marker names to drop (comma separated)")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--iters-joint", type=int, default=600)
    ap.add_argument("--iters-surface", type=int, default=600)
    ap.add_argument("--icp-rounds", type=int, default=3)
    ap.add_argument("--protocol", choices=["data_june25", "label_hints"],
                     default="data_june25",
                     help="data_june25: <finger>1=MCP, 2=PIP, 3=nailbed base. "
                          "label_hints: whatever core.correspondence currently says.")
    ap.add_argument("--surface-only", default="Palm1,Palm3",
                     help="markers fitted to the surface but with no joint to "
                          "seed from (they skip stage 1)")
    ap.add_argument("--min-dorsal-dot", type=float, default=0.0,
                     help="attachment faces must have normal.dorsal above this. "
                          "0 = any face on the back half; raise it to insist on "
                          "faces squarely facing the back.")
    ap.add_argument("--offset-mode", action="store_true",
                     help="ABLATION: model each marker as a rigid offset from "
                          "its joint in that joint's segment frame (the "
                          "refit_with_marker_offsets.py approach) instead of as "
                          "a point on the mesh surface. Everything else -- "
                          "staging, marker init, joint limits, smoothing -- is "
                          "unchanged, so the comparison isolates the attachment "
                          "representation.")
    ap.add_argument("--stretch", default=None,
                     help="JSON of flexion-dependent marker slide (see "
                          "configs/surface_fit/P10_stretch.json). Lets each "
                          "attachment travel distally along its segment as the "
                          "joint flexes, which is what the skin actually does.")
    ap.add_argument("--stretch-gate", type=float, default=45.0,
                     help="degrees of flexion below which no slide is applied. "
                          "The correction is worth ~1mm at low flexion and "
                          "loses to noise there; it is worth 7-10mm near a fist.")
    ap.add_argument("--attach", default=None,
                     help='JSON mapping marker name -> mesh vertex id, e.g. '
                          '{"Index2": 237, "Middle2": 396}. Either inline JSON '
                          "or a path to a .json file. Listed markers skip ICP and "
                          "are pinned to that vertex; the rest are unchanged.")
    ap.add_argument("--picker-out", default=None,
                     help="write an HTML view of the fitted mesh with every vertex "
                          "id hoverable, to choose --attach values from, then exit")
    ap.add_argument("--pca-comps", type=int, default=45,
                     help="MANO pose PCA components (default 45 = full). The "
                          "repo default of 6 cannot represent finger abduction.")
    ap.add_argument("--w-pose", type=float, default=1e-3,
                     help="L2 shrinkage of the pose toward MANO's mean. Needed "
                          "once --pca-comps is raised: the narrow PCA basis was "
                          "itself the anatomical prior.")
    ap.add_argument("--biomech", type=float, default=0.0,
                     help="weight for anatomical joint limits (hinge PIP/DIP, "
                          "bounded MCP abduction, no twist). Implies --no-pca, "
                          "since limits are per-joint facts the PCA basis mixes. "
                          "Try 1.0; 0 disables.")
    ap.add_argument("--no-pca", action="store_true",
                     help="use raw 45-DOF axis-angle pose instead of the PCA basis")
    ap.add_argument("--movement", default=None,
                     help="movement trial CSV to apply the static-calibrated "
                          "shape and attachments to (pose-only fit)")
    ap.add_argument("--mv-start", type=int, default=None)
    ap.add_argument("--mv-end", type=int, default=None)
    ap.add_argument("--mv-frames", type=int, default=0,
                     help="cap on frames (0 = all in the window)")
    ap.add_argument("--staged", action="store_true",
                     help="root-first: solve global orient/transl from the palm "
                          "and MCP markers alone, then fit the fingers")
    ap.add_argument("--free-flexion", action="store_true",
                     help="bound abduction and torsion but NOT flexion. Stops "
                          "fingers splaying through each other without "
                          "re-imposing flexion limits that collapse them.")
    ap.add_argument("--torsion-only", action="store_true",
                     help="apply ONLY the zero-torsion constraint, leaving "
                          "flexion and abduction unbounded")
    ap.add_argument("--twist-weight", type=float, default=10.0,
                     help="extra weight on the axial-torsion penalty. Finger "
                          "twist bounds are (0,0), so this sets how hard that "
                          "is enforced relative to the flexion/abduction range.")
    ap.add_argument("--min-own-weight", type=float, default=0.25,
                     help="a marker may only attach to skin whose LBS weight on "
                          "its own bone is at least this. Stops MCP markers "
                          "landing on palm-skinned faces.")
    ap.add_argument("--marker-init", action="store_true", default=True,
                     help="seed hand_pose from flexion angles measured off the "
                          "markers rather than starting from a flat hand")
    ap.add_argument("--no-marker-init", dest="marker_init", action="store_false")
    ap.add_argument("--freeze-root", action="store_true", default=True,
                     help="with --staged, hold global orient/transl fixed during "
                          "the finger stage (default on; the point of staging)")
    ap.add_argument("--no-freeze-root", dest="freeze_root", action="store_false",
                     help="let the finger stage re-optimise the root too")
    ap.add_argument("--root-only", action="store_true",
                     help="fit ONLY the palm+MCP markers (root stage alone); "
                          "finger markers are shown but given no vote")
    ap.add_argument("--iters-thumb", type=int, default=0,
                     help="iterations for a thumb-only stage between the root "
                          "and finger stages (0 = skip). Fits the thumb joints "
                          "against the thumb markers alone, root frozen.")
    ap.add_argument("--iters-finger", type=int, default=0,
                     help="iterations for a per-finger stage (index, middle, "
                          "ring, pinky each fitted alone against their own "
                          "markers, root frozen). 0 = skip.")
    ap.add_argument("--iters-root", type=int, default=0,
                     help="absolute iterations for the root stage (0 = use "
                          "--root-frac of the finger budget). The root stage is "
                          "cheap and converging it fully is what gets palm+MCP "
                          "markers to a few mm.")
    ap.add_argument("--root-frac", type=float, default=0.4,
                     help="fraction of the iterations spent on the root stage")
    ap.add_argument("--mv-chunk", type=int, default=1500,
                     help="frames per optimisation chunk; a whole trial does "
                          "not fit in one batch. Chunks warm-start from the "
                          "previous chunk's last frame.")
    ap.add_argument("--npz-out", default=None,
                     help="write a FitResult-style npz (readable by "
                          "scripts/mano/animate_mesh.py)")
    ap.add_argument("--iters-mv", type=int, default=0,
                     help="iterations for the movement fit (0 = use "
                          "--iters-surface). A pinch needs more than a static.")
    ap.add_argument("--mv-stride", type=int, default=1,
                     help="frame stride; 1 = every frame, which is what the "
                          "temporal term assumes")
    ap.add_argument("--w-smooth", type=float, default=1.0,
                     help="temporal smoothness on pose/orient/transl "
                          "(2nd difference). 0 = independent per-frame fits.")
    ap.add_argument("--no-seed-gap-fill", dest="seed_gap_fill", action="store_false",
                     help="old behaviour: seed an unobservable finger angle as 0 (straight) "
                          "instead of interpolating it across the gap")
    ap.add_argument("--from-npz", default=None,
                     help="skip the movement optimisation and load pose from "
                          "this npz (an earlier --npz-out with the same config)")
    ap.add_argument("--pred-out", default=None,
                     help="npz of predicted marker positions per movement "
                          "frame (pred_mm (T,K,3), frames, names)")
    ap.add_argument("--movement-out", default=None,
                     help="HTML for a representative movement frame")
    ap.add_argument("--png", default=None, help="also bake a static PNG view")
    ap.add_argument("--w-shape", type=float, default=1e-5)
    ap.add_argument("--standoff", type=float, default=5.0,
                     help="mm the marker centre should sit ABOVE the skin "
                          "(half a reflective marker's diameter; 0 = on the surface)")
    ap.add_argument("--w-inside", type=float, default=10.0,
                     help="extra weight on markers ending up inside the mesh")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    st, lab_s = load_csv(args.static)
    base = [l.split(":")[-1] for l in lab_s]
    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}

    # MANO's 6-component pose PCA is a grasp basis: finger abduction lives in
    # the later components, so a 6-comp fit cannot adduct the fingers at all.
    # Measured on P10's static: marker lateral spans are 0.85-0.89x the mesh's,
    # i.e. the subject's fingers are closer together than the basis can express.
    if args.biomech > 0:
        args.no_pca = True      # limits are defined on raw axis-angle components
    cfg = FitConfig(mano_model_path=args.mano_dir, hand_side=args.side,
                    use_pca=not args.no_pca, n_pca_comps=args.pca_comps)
    jl_lo = jl_hi = None
    fitter = MANOFitter(cfg)
    model, dev = fitter._model, fitter.device
    F = np.asarray(model.faces, dtype=np.int64)

    jidx = {n: i for i, n in enumerate(MANO_JOINT_NAMES)}
    if args.protocol == "data_june25":
        # Physical placement, per the person who placed the markers:
        #   <finger>1 = MCP, <finger>2 = PIP, <finger>3 = base of the nailbed.
        #   Thumb1 ~ CMC, Thumb2 = MCP, Thumb3 = IP.
        # The thumb needs no special case: MANO's thumb chain starts at the
        # CMC, so its `thumb_mcp`/`thumb_pip`/`thumb_dip` slots already line up
        # with CMC/MCP/IP positionally.
        # The nailbed base is on the distal phalanx, just past the DIP, so DIP
        # is the joint to SEED from; stage 2's ICP then slides the attachment
        # onto the actual dorsal/nailbed surface point, which is the whole
        # reason this script does not need the mapping to be exact.
        # NB: this deliberately differs from core.correspondence._LABEL_HINTS,
        # which maps <finger>3 to the FINGERTIP (commit 49fb041). Only this
        # script is changed; the shared table is left alone.
        proto = {"wrist": "Palm2"}
        for f, pre in (("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"),
                       ("ring", "Ring"), ("pinky", "Pinky")):
            proto[f"{f}_mcp"] = f"{pre}1"
            proto[f"{f}_pip"] = f"{pre}2"
            proto[f"{f}_dip"] = f"{pre}3"
        seed = {j: base.index(nm) for j, nm in proto.items() if nm in base}
    else:
        seed = label_seed(lab_s, side=args.side)
    if not seed:
        raise SystemExit("could not seed correspondences from the static's labels")
    pairs = [(jidx[j], mi) for j, mi in seed.items() if base[mi] not in excluded]
    # Surface-only markers: Palm1/Palm3 have no MANO joint to seed from, so
    # they sit out of stage 1 (joint index -1) but are fitted in stage 2 --
    # which only ever needed a surface attachment, never a joint.
    extra = [s.strip() for s in args.surface_only.split(",") if s.strip()]
    extra = [nm for nm in extra if nm in base and nm not in excluded]
    pairs += [(-1, base.index(nm)) for nm in extra]
    pairs = [(j, mi) for j, mi in pairs if np.isfinite(st[:, mi]).all(-1).any()]
    names = [base[mi] for _, mi in pairs]
    print(f"{len(pairs)} markers: {names}")
    print(f"  surface-only (no stage-1 joint target): {extra}")

    sf = np.linspace(0, st.shape[0] - 1, min(args.n_frames, st.shape[0])).astype(int)
    S = len(sf)
    M = torch.tensor(st[sf][:, [mi for _, mi in pairs]] * 1e-3,
                     dtype=torch.float32, device=dev)          # (S, P, 3) metres
    mmask = torch.isfinite(M).all(-1)

    # ---- stage 1: the existing joint-centre objective -------------------
    tgt = torch.full((S, 21, 3), float("nan"), device=dev)
    for k, (j, _) in enumerate(pairs):
        if j >= 0:
            tgt[:, j] = M[:, k]
    jmask = torch.isfinite(tgt).all(-1)

    # Kabsch init, as MANOFitter does. Without it global_orient starts at
    # identity and gradient descent cannot recover a large wrist rotation --
    # it curls the fingers to reach the markers instead, which is exactly the
    # fist-shaped fit this script produced before this was added.
    tgt_np = tgt.detach().cpu().numpy()
    init_go, init_tr = fitter._init_global_pose(tgt_np, np.zeros(10, dtype=np.float32))
    # Always build the basis: it is needed for the limits and the marker
    # init, but also for the fitted-flexion REPORT, which has to work
    # regardless of which constraints are switched on -- otherwise the one
    # configuration you most want to inspect (all constraints off) is the one
    # that prints nothing.
    Bmat = jl_lo = jl_hi = Bt = None
    if fitter._pose_dim() == 45:
        Bmat = joint_axis_basis(model, dev, fitter._pose_dim())
        Bt = torch.tensor(Bmat, device=dev)
        _lo, _hi = basis_limit_bounds(args.torsion_only, args.free_flexion)
        jl_lo = torch.tensor(_lo, device=dev)
        jl_hi = torch.tensor(_hi, device=dev)
    if args.biomech <= 0:
        jl_lo = jl_hi = None
    betas = nn.Parameter(torch.zeros(10, device=dev))
    orient = nn.Parameter(torch.tensor(init_go, device=dev))
    hp = nn.Parameter(torch.zeros(S, fitter._pose_dim(), device=dev))
    tr = nn.Parameter(torch.tensor(init_tr, device=dev))

    def fwd(verts=True):
        # return_verts=False suppresses `joints` too in this smplx version
        return model(global_orient=orient, hand_pose=hp, transl=tr,
                     betas=betas.unsqueeze(0).expand(S, -1), return_verts=True)

    opt = torch.optim.Adam([{"params": [betas], "lr": 1e-2},
                            {"params": [hp], "lr": 5e-2},
                            {"params": [orient], "lr": 1e-2},
                            {"params": [tr], "lr": 3e-3}])
    print(f"\nstage 1 -- joint-centre fit ({args.iters_joint} iters)")
    for it in range(args.iters_joint):
        opt.zero_grad()
        pred = mano_output_to_joints21(fwd())
        loss = (((pred - tgt)[jmask] ** 2).sum(-1).mean()
                + args.w_shape * (betas ** 2).sum()
                + args.w_pose * (hp ** 2).sum(-1).mean())
        if args.biomech > 0 and jl_lo is not None:
            loss = loss + args.biomech * basis_limit_penalty(hp, Bt, jl_lo, jl_hi, args.twist_weight)
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            print(f"  iter {it+1:4d}  joint resid "
                  f"{(pred - tgt)[jmask].norm(dim=-1).mean().item()*1000:6.2f} mm")

    k0 = S // 2
    with torch.no_grad():
        out = fwd(verts=True)
    V0 = out.vertices[k0].cpu().numpy() * 1000
    mk0 = st[sf[k0]][[mi for _, mi in pairs]]
    surface_report(V0, F, mk0, names, "stage 1 (joint-centre fit, what static_fit.html shows)")

    # ---- flexion-dependent slide -----------------------------------------
    slopes: dict[str, float] = {}
    ref_deg: dict[str, float] = {}
    if args.stretch:
        import json as _json
        _sd = _json.loads(Path(args.stretch).read_text())
        slopes = {k: float(v) for k, v in _sd.get("slope_mm_per_deg", {}).items()}
        ref_deg = {k: float(v) for k, v in _sd.get("ref_deg", {}).items()}
        used = {k: v for k, v in slopes.items() if v}
        print(f"flexion-dependent slide for {len(used)} markers: "
              + ", ".join(f"{k} {v:.3f}mm/deg" for k, v in sorted(used.items())))

    # ---- parse hand-picked attachments ---------------------------------
    pinned: dict[str, int] = {}
    if args.attach:
        import json
        raw = (Path(args.attach).read_text() if Path(args.attach).exists()
               else args.attach)
        pinned = {k: int(v) for k, v in json.loads(raw).items()}
        unknown = sorted(set(pinned) - set(names))
        if unknown:
            raise SystemExit(f"--attach names not among the fitted markers: {unknown}")
        print(f"hand-picked attachments: {pinned}")

    if args.out is None and not args.picker_out:
        raise SystemExit("--out is required (unless you only want --picker-out)")

    if args.picker_out:
        with torch.no_grad():
            Vp0 = fwd().vertices[k0].cpu().numpy() * 1000
        write_vertex_picker(Vp0, F, mk0, names, args.picker_out)
        return

    # ---- stage 2: attach markers to the mesh surface, refit -------------
    # The attachment is a barycentric point on a face plus a standoff along
    # the outward normal: a real marker's centre sits ~half its diameter
    # above the skin, so the correct target is "just outside", not "on".
    mk_all = st[sf][:, [mi for _, mi in pairs]] * 1e-3
    lbs = model.lbs_weights.detach().cpu().numpy()
    allow = [allowed_faces(lbs, F, j, min_own=args.min_own_weight)
             for j, _ in pairs]
    for (j, _), nm, al in zip(pairs, names, allow):
        print(f"  {nm:8s} may attach to {len(al):4d} faces")
    for rnd in range(args.icp_rounds):
        with torch.no_grad():
            Vm = np.nanmean(fwd().vertices.cpu().numpy(), axis=0)
        mkm = np.nanmean(mk_all, axis=0)
        with torch.no_grad():
            Jm = mano_output_to_joints21(fwd()).cpu().numpy().mean(0)
        if "Palm2" in names:
            pd_ = dorsal_direction(Jm, mkm[names.index("Palm2")])
            dors = segment_dorsals(Jm, pd_, [j for j, _ in pairs])
        else:
            dors = None
        fid, bary = closest_point_on_faces(Vm, F, mkm, allow=allow,
                                           dorsal=dors, min_dot=args.min_dorsal_dot)
        # Hand-picked attachments override the search entirely. The automatic
        # one has to infer which bit of skin a marker is taped to; you may
        # simply know, and a pinned vertex is not re-litigated each round.
        for nm, vtx in pinned.items():
            k = names.index(nm)
            fid[k], bary[k] = onehot_attachment(F, vtx)
            # Pinned vertices used to bypass the own-bone weight check, which
            # runs inside allowed_faces -- the pin is written in afterwards.
            # That is how four hand-picked PIP vertices survived at 2-17%
            # weight on their own bone. Screen them too, and say so rather
            # than silently honouring an unobservable attachment.
            j_k = pairs[k][0]
            if rnd == 0 and j_k >= 0 and args.min_own_weight > 0:
                b_k = _REPO_TO_BONE[j_k]
                if b_k != 0:
                    w = float(lbs[vtx, b_k])   # the vertex itself, not its face
                    if w < args.min_own_weight:
                        print(f"    WARNING {nm}: pinned vertex {vtx} is only "
                              f"{w:.1%} skinned to its own bone -- its joint "
                              f"barely moves it, so that angle is near-"
                              f"unobservable. Consider a vertex further distal.")
        fidx = torch.tensor(F[fid], dtype=torch.long, device=dev)       # (P,3)
        bw = torch.tensor(bary, device=dev).unsqueeze(0).unsqueeze(-1)  # (1,P,3,1)
        print(f"\nstage 2 round {rnd+1} -- attached faces {fid.tolist()}")

        opt = torch.optim.Adam([{"params": [betas], "lr": 5e-3},
                                {"params": [hp], "lr": 2e-2},
                                {"params": [orient], "lr": 5e-3},
                                {"params": [tr], "lr": 3e-3}])
        for it in range(args.iters_surface):
            opt.zero_grad()
            # Occluded markers are NaN. Masking removes them from the loss
            # VALUE, but not from the backward pass: the inward term multiplies
            # by (M - P), so a masked entry contributes 0 * NaN = NaN to nrm's
            # gradient and kills every parameter on the next step. Substituting
            # 0 is exactly equivalent under the mask and keeps the graph finite.
            M_s = torch.nan_to_num(M, nan=0.0)
            Vt = fwd().vertices[:, fidx]                                # (S,P,3,3)
            P = (Vt * bw).sum(2)                                        # surface point
            nrm = torch.cross(Vt[:, :, 1] - Vt[:, :, 0],
                              Vt[:, :, 2] - Vt[:, :, 0], dim=-1)
            nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            tgt_pt = P + args.standoff * 1e-3 * nrm                     # just outside
            d = (tgt_pt - M_s)[mmask]
            # one-sided term: a marker can never be *inside* the skin, so
            # penalise the inward component extra rather than symmetrically
            inward = torch.relu(-((M_s - P) * nrm).sum(-1))[mmask]
            loss = ((d ** 2).sum(-1).mean()
                    + args.w_inside * (inward ** 2).mean()
                    + args.w_shape * (betas ** 2).sum()
                    # Shrink toward MANO's mean pose. Without this a full
                    # 45-DOF fit reproduces the markers with an anatomically
                    # impossible claw: the PCA basis was acting as the pose
                    # prior, so widening it requires putting one back.
                    + args.w_pose * (hp ** 2).sum(-1).mean())
            if args.biomech > 0 and jl_lo is not None:
                loss = loss + args.biomech * basis_limit_penalty(hp, Bt, jl_lo, jl_hi, args.twist_weight)
            loss.backward()
            opt.step()
            if (it + 1) % 300 == 0:
                print(f"  iter {it+1:4d}  surface-point-to-marker "
                      f"{d.norm(dim=-1).mean().item()*1000:6.2f} mm  "
                      f"(inward {inward.mean().item()*1000:5.2f} mm)")

    with torch.no_grad():
        out = fwd(verts=True)
    V1 = out.vertices[k0].cpu().numpy() * 1000
    J1 = mano_output_to_joints21(out)[k0].cpu().numpy() * 1000
    d1 = surface_report(V1, F, mk0, names, "stage 2 (surface fit)")

    # Per-marker check against the ATTACHED point specifically. The loop's
    # mean can hide a few markers that the fit never satisfied.
    Pk = (V1[F[fid]] * bary[:, :, None]).sum(1)
    print("\n--- per marker: distance to its own attached surface point ---")
    _bone_name = {0: "palm", 1: "idx-prox", 2: "idx-mid", 3: "idx-dist",
                  4: "mid-prox", 5: "mid-mid", 6: "mid-dist",
                  7: "pnk-prox", 8: "pnk-mid", 9: "pnk-dist",
                  10: "rng-prox", 11: "rng-mid", 12: "rng-dist",
                  13: "thb-prox", 14: "thb-mid", 15: "thb-dist"}
    dom = lbs.argmax(1)
    for nm, a, b, f in zip(names, Pk, mk0, fid):
        seg = _bone_name[int(np.bincount(dom[F[f]]).argmax())]
        print(f"  {nm:10s} {np.linalg.norm(a - b):7.2f} mm   on {seg}")
    print(f"\ncalibrated betas = {betas.detach().cpu().numpy().round(3)}")

    fig = go.Figure(data=[
        go.Mesh3d(x=V1[:, 0], y=V1[:, 1], z=V1[:, 2],
                  i=F[:, 0], j=F[:, 1], k=F[:, 2],
                  color="#e8a0a0", opacity=1.0, flatshading=True,
                  lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05),
                  name="MANO (surface-fitted)"),
        go.Scatter3d(x=mk0[:, 0], y=mk0[:, 1], z=mk0[:, 2], mode="markers+text",
                     marker=dict(size=6, color="#1b3a5c"),
                     text=names, textposition="top center",
                     hovertext=[f"{n}: {v:+.1f} mm" for n, v in zip(names, d1)],
                     name="Vicon markers (static)"),
        # Link each marker to the surface point it is actually fitted to.
        # A long link means a real residual; a marker that LOOKS adrift but
        # has a short link is a viewing artifact, not a fit error.
        go.Scatter3d(
            x=np.ravel([[a[0], b[0], None] for a, b in zip(mk0, Pk)]),
            y=np.ravel([[a[1], b[1], None] for a, b in zip(mk0, Pk)]),
            z=np.ravel([[a[2], b[2], None] for a, b in zip(mk0, Pk)]),
            mode="lines", line=dict(color="#16a34a", width=5),
            name="marker -> its attached surface point"),
        go.Scatter3d(x=Pk[:, 0], y=Pk[:, 1], z=Pk[:, 2], mode="markers",
                     marker=dict(size=4, color="#16a34a", symbol="diamond"),
                     name="attached surface point"),
        go.Scatter3d(x=J1[:, 0], y=J1[:, 1], z=J1[:, 2], mode="markers",
                     marker=dict(size=3, color="#555555"),
                     name="MANO joint centres", visible="legendonly"),
    ])
    # Default camera looking straight down the back of the hand. Plotly's
    # default eye is a fixed diagonal in *scene* coordinates, which for this
    # data lands edge-on down the fingers and reads as a fist -- the fit was
    # fine, the view was not. The dorsal direction is taken from the fitted
    # mesh itself: the outward normal at Palm2's attachment.
    Nv1 = vertex_normals(V1, F)
    dorsal = Nv1[F[fid[names.index("Palm2")] if "Palm2" in names else 0]].mean(0)
    dorsal = dorsal / np.linalg.norm(dorsal)
    up = J1[9] - J1[0]                      # wrist -> middle MCP, along the hand
    up = up - up.dot(dorsal) * dorsal
    up = up / np.linalg.norm(up)
    fig.update_layout(
        title=f"MANO surface-fitted to the static — markers on the skin, "
              f"mean |marker-to-surface| {np.abs(d1).mean():.1f} mm",
        scene=dict(aspectmode="data",
                   camera=dict(eye=dict(x=dorsal[0] * 1.9, y=dorsal[1] * 1.9,
                                        z=dorsal[2] * 1.9),
                               up=dict(x=up[0], y=up[1], z=up[2]),
                               center=dict(x=0, y=0, z=0),
                               projection=dict(type="orthographic"))),
        width=1000, height=850)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p), include_plotlyjs=True)
    print(f"saved {p}")
    if args.png:
        fig.write_image(args.png, width=1000, height=850, scale=1)
        print(f"saved {args.png}")

    # ---- ABLATION: rigid segment-frame offsets instead of mesh points ----
    # Calibrated on the SAME static fit the surface attachments come from, so
    # the only thing that differs downstream is how a marker's predicted
    # position is computed: J[joint] + R_segment @ offset, versus a barycentric
    # point on the mesh plus a standoff along its normal.
    off_vec = None
    if args.offset_mode:
        global _OFFSET_SEG
        _OFFSET_SEG = []
        repo_of = []
        for (j, _), nm in zip(pairs, names):
            rj = j if j >= 0 else 0
            a, b = _SEGMENT_OF.get(rj, (0, 9))
            _OFFSET_SEG.append((rj, a, b))
            repo_of.append(rj)
        with torch.no_grad():
            Js = mano_output_to_joints21(fwd())
        off_vec, off_spread = calibrate_segment_offsets(Js, M, names, repo_of)
        print("\noffset mode: per-marker rigid offset in its segment frame")
        print(f"{'marker':10s} {'|offset|':>9s} {'spread':>8s}")
        for k, nm in enumerate(names):
            mag = float(off_vec[k].norm()) * 1000
            flag = "  <-- not pose-invariant" if off_spread[k] > 0.6 * mag else ""
            print(f"  {nm:10s} {mag:7.2f}mm {off_spread[k]:7.2f}mm{flag}")

    # ---- apply to a movement trial -------------------------------------
    # Pose only. Both the shape (betas) and the attachments (which patch of
    # skin each marker is taped to) are physical facts about this subject and
    # this marker placement, so they are CALIBRATED ON THE STATIC and frozen
    # here -- not re-searched per frame. Re-running ICP on a movement trial
    # would let a marker slide to a different bit of skin whenever the fit
    # drifted, which is how a plausible-looking but wrong pose gets rewarded.
    if args.movement:
        mv, lab_m = load_csv(args.movement)
        bm = [l.split(":")[-1] for l in lab_m]
        missing = [n for n in names if n not in bm]
        if missing:
            raise SystemExit(f"movement trial lacks markers: {missing}")
        lo = args.mv_start or 0
        hi = args.mv_end if args.mv_end is not None else mv.shape[0] - 1
        mf = np.arange(lo, hi + 1, max(1, args.mv_stride))
        if args.mv_frames and len(mf) > args.mv_frames:
            mf = mf[:args.mv_frames]
        cols = [bm.index(n) for n in names]
        betas_f = betas.detach()
        chunk = args.mv_chunk if args.mv_chunk > 0 else len(mf)
        n_chunks = int(np.ceil(len(mf) / chunk))
        print(f"\n=== movement: {args.movement} ===")
        print(f"frames {mf[0]}..{mf[-1]} ({len(mf)} frames, stride "
              f"{args.mv_stride}) in {n_chunks} chunk(s) of up to {chunk}; "
              f"betas and attachments frozen from the static")

        GO, HP, TR = [], [], []
        prev_hp = prev_go = prev_tr = None
        # --from-npz: reuse a finished movement fit (same static/attach/stretch
        # config) and only rebuild its predictions -- ~80s instead of ~70min.
        pre = np.load(args.from_npz) if args.from_npz else None
        for ci in range(n_chunks):
            sl = slice(ci * chunk, min((ci + 1) * chunk, len(mf)))
            idx = mf[sl]
            Tm = len(idx)
            if pre is not None:
                GO.append(pre["global_orient"][idx]); HP.append(pre["hand_pose"][idx])
                TR.append(pre["transl"][idx])
                continue
            Mm = torch.tensor(mv[idx][:, cols] * 1e-3, dtype=torch.float32, device=dev)
            mmk = torch.isfinite(Mm).all(-1)
            Mm_s = torch.nan_to_num(Mm, nan=0.0)   # see the note in stage 2

            tgt_m = torch.full((Tm, 21, 3), float("nan"), device=dev)
            for kk, (jj, _) in enumerate(pairs):
                if jj >= 0:
                    tgt_m[:, jj] = Mm[:, kk]
            go_m, tr_m = fitter._init_global_pose(
                tgt_m.detach().cpu().numpy(), betas_f.cpu().numpy())
            hp0 = np.zeros((Tm, fitter._pose_dim()), np.float32)
            if args.marker_init and fitter._pose_dim() == 45:
                # Use EVERY marker in the file for the angle seed, not just the
                # fitted subset: a marker can be useless as a fit target (a PIP
                # marker sits on proximal skin, so it cannot constrain the PIP)
                # while still measuring the angle perfectly well from geometry.
                # Excluding it from the objective should not blind the seed.
                hp0 = angles_from_markers(mv[idx], bm, Bmat, gap_fill=args.seed_gap_fill)
                print(f"    seeded pose from markers: mean |angle| "
                      f"{np.abs(hp0).mean():.3f} rad")
            # Warm start from the previous chunk's last solved frame -- but
            # ONLY when there is no per-frame marker seed to destroy. This
            # used to overwrite the whole chunk's marker-derived init with a
            # single stale pose, so only chunk 1 was ever seeded from the
            # markers. It is why a one-chunk window scored 6.65mm on the
            # fingers while the full trial scored 19.28: at frame 3972 the
            # markers show a ~90 deg fist and the fit sat at ~0 deg PIP,
            # having started from a pose copied out of chunk 2.
            if prev_hp is not None and not args.marker_init:
                hp0[:] = prev_hp
                go_m[0] = prev_go
                tr_m[0] = prev_tr
            orient_m = nn.Parameter(torch.tensor(go_m, device=dev))
            hp_m = nn.Parameter(torch.tensor(hp0, device=dev))
            tr_m2 = nn.Parameter(torch.tensor(tr_m, device=dev))

            # Staged (root-first) fit. Solving the root transform at the same
            # time as finger pose lets a badly-fitting finger tilt the whole
            # hand: the index sits at ~22mm and drags Palm3 and the MCPs with
            # it. The palm and MCP markers are the ones rigid w.r.t. the palm,
            # so they alone determine orientation and translation first; the
            # fingers then only have to explain themselves.
            root_k = [kk for kk, nm in enumerate(names)
                      if nm.startswith("Palm") or nm.endswith("1")]
            n_it = args.iters_mv or args.iters_surface
            # --root-only: spend every iteration on the root stage and never
            # run the full one -- a diagnostic that answers "is the palm+MCP
            # subset even self-consistent?" without finger markers getting a
            # vote. If the hand still sits wrong here, no amount of finger
            # fitting downstream will rescue it.
            # --iters-root lets the root stage run to convergence rather than
            # taking a fraction of the finger budget. Measured: the root-only
            # fit (all 1500 iterations on palm+MCP) placed those markers at
            # 2.6-6.4mm, while --staged at root_frac=0.4 left them at 10.3mm --
            # the root simply had not converged before the fingers took over.
            if args.root_only:
                n_root = n_it
            elif not args.staged:
                n_root = 0
            elif args.iters_root > 0:
                n_root = args.iters_root
            else:
                n_root = int(n_it * args.root_frac)
            opt_root = torch.optim.Adam([{"params": [orient_m], "lr": 1e-2},
                                         {"params": [tr_m2], "lr": 5e-3}])
            # In staged mode the finger stage must NOT keep re-optimising the
            # root. Measured: palm+MCP alone place the hand to 2.6-6.4mm, but
            # letting the fingers vote on orient/transl afterwards pushes those
            # same markers to 11-28mm -- the finger error drags the root. So
            # stage 2 solves pose only, against a root that is now fixed.
            if args.staged and args.freeze_root:
                opt = torch.optim.Adam([{"params": [hp_m], "lr": 2e-2}])
            else:
                opt = torch.optim.Adam([{"params": [hp_m], "lr": 2e-2},
                                        {"params": [orient_m], "lr": 5e-3},
                                        {"params": [tr_m2], "lr": 3e-3}])
            for it in range(n_root):
                opt_root.zero_grad()
                om = model(global_orient=orient_m, hand_pose=hp_m, transl=tr_m2,
                           betas=betas_f.unsqueeze(0).expand(Tm, -1),
                           return_verts=True)
                if off_vec is not None:
                    Jc = mano_output_to_joints21(om)
                    R = segment_frames_t(Jc)[:, root_k]
                    P = torch.stack([Jc[:, _OFFSET_SEG[q][0]] for q in root_k], 1)                         + torch.einsum("tpij,pj->tpi", R, off_vec[root_k])
                else:
                    Vt = om.vertices[:, fidx[root_k]]
                    P = (Vt * bw[:, root_k]).sum(2)
                dr = (P - Mm_s[:, root_k])[mmk[:, root_k]]
                lr_ = (dr ** 2).sum(-1).mean()
                if args.w_smooth > 0 and Tm > 2:
                    lr_ = lr_ + args.w_smooth * (
                        (orient_m[2:] - 2 * orient_m[1:-1] + orient_m[:-2]) ** 2
                    ).sum(-1).mean()
                if not torch.isfinite(lr_):
                    raise RuntimeError(f"root stage non-finite at chunk {ci} it {it}")
                lr_.backward()
                opt_root.step()
            # ---- thumb stage: root fixed, ONLY thumb joints, ONLY thumb
            # markers. The thumb's CMC is a saddle joint whose axes separate
            # far less cleanly than a finger's (0.93 palmward / 0.30 sideways
            # against ~0.99/0.02), so it is the chain most likely to be
            # out-competed when all 18 markers pull at once. Solving it first,
            # against its own markers alone, gives it the same protection the
            # root stage gives the palm.
            # ---- per-chain stage ------------------------------------
            # With the root frozen, each finger is an INDEPENDENT kinematic
            # chain: its joints move only its own markers, so fitting the
            # chains separately is mathematically equivalent to fitting them
            # together. What differs is the optimisation: the data term is a
            # mean over all markers while the smoothness and joint-limit
            # penalties are summed over all joints, so per chain those sit at
            # a different relative weight, and each chain gets its own
            # convergence rather than a share of one budget. That this
            # matters is not theoretical -- fitting the thumb alone first
            # moved Index3 from ~27mm to 10.7mm.
            # pose component blocks, in MANO's joint order
            _CHAIN_C = {"Index": (0, 9), "Middle": (9, 18), "Pinky": (18, 27),
                        "Ring": (27, 36), "Thumb": (36, 45)}
            chains = []
            if args.iters_thumb > 0:
                chains.append(("Thumb", args.iters_thumb))
            if args.iters_finger > 0:
                for cn in ("Index", "Middle", "Ring", "Pinky"):
                    chains.append((cn, args.iters_finger))
            for cname, n_ch in chains:
                ch_k = [kk for kk, nm in enumerate(names) if nm.startswith(cname)]
                if not ch_k or fitter._pose_dim() != 45:
                    continue
                lo_c, hi_c = _CHAIN_C[cname]
                gmask = torch.zeros(fitter._pose_dim(), device=dev)
                gmask[lo_c:hi_c] = 1.0
                opt_c = torch.optim.Adam([{"params": [hp_m], "lr": 2e-2}])
                for it in range(n_ch):
                    opt_c.zero_grad()
                    om = model(global_orient=orient_m, hand_pose=hp_m,
                               transl=tr_m2,
                               betas=betas_f.unsqueeze(0).expand(Tm, -1),
                               return_verts=True)
                    if off_vec is not None:
                        Jc = mano_output_to_joints21(om)
                        R = segment_frames_t(Jc)[:, ch_k]
                        P = torch.stack([Jc[:, _OFFSET_SEG[q][0]] for q in ch_k], 1)                             + torch.einsum("tpij,pj->tpi", R, off_vec[ch_k])
                        nr = torch.zeros_like(P)
                    else:
                        Vt = om.vertices[:, fidx[ch_k]]
                        P = (Vt * bw[:, ch_k]).sum(2)
                        nr = torch.cross(Vt[:, :, 1] - Vt[:, :, 0],
                                         Vt[:, :, 2] - Vt[:, :, 0], dim=-1)
                        nr = nr / torch.sqrt((nr ** 2).sum(-1, keepdim=True) + 1e-18)
                    if slopes and Bt is not None:
                        Jc = mano_output_to_joints21(om)
                        P = P + stretch_offsets(Jc, hp_m, Bt,
                                                [names[q] for q in ch_k], slopes,
                                                ref_deg, args.stretch_gate)
                    dt = (P + args.standoff * 1e-3 * nr
                          - Mm_s[:, ch_k])[mmk[:, ch_k]]
                    lt = (dt ** 2).sum(-1).mean()
                    if args.w_smooth > 0 and Tm > 2:
                        lt = lt + args.w_smooth * (
                            (hp_m[2:] - 2 * hp_m[1:-1] + hp_m[:-2]) ** 2
                        ).sum(-1).mean()
                    if args.biomech > 0 and jl_lo is not None:
                        lt = lt + args.biomech * basis_limit_penalty(
                            hp_m, Bt, jl_lo, jl_hi, args.twist_weight)
                    if not torch.isfinite(lt):
                        raise RuntimeError(f"{cname} stage non-finite, chunk {ci}")
                    lt.backward()
                    hp_m.grad.mul_(gmask)      # only this chain may move
                    opt_c.step()
                with torch.no_grad():
                    print(f"    {cname:7s} stage: "
                          f"{dt.norm(dim=-1).mean().item()*1000:6.2f} mm over "
                          f"{len(ch_k)} markers")

            # An explicit --iters-root is additive: it buys the root more
            # iterations rather than taking them from the fingers. (Setting it
            # equal to --iters-mv previously left the finger stage with zero
            # iterations, so nothing was fitted and `dd` was never assigned.)
            n_fing = 0 if args.root_only else (
                n_it if args.iters_root > 0 else n_it - n_root)
            dd = None
            for it in range(n_fing):
                opt.zero_grad()
                om = model(global_orient=orient_m, hand_pose=hp_m, transl=tr_m2,
                           betas=betas_f.unsqueeze(0).expand(Tm, -1),
                           return_verts=True)
                if off_vec is not None:
                    Jc = mano_output_to_joints21(om)
                    R = segment_frames_t(Jc)
                    P = torch.stack([Jc[:, sg[0]] for sg in _OFFSET_SEG], 1)                         + torch.einsum("tpij,pj->tpi", R, off_vec)
                    nrm = torch.zeros_like(P)
                else:
                    Vt = om.vertices[:, fidx]
                    P = (Vt * bw).sum(2)
                    nrm = torch.cross(Vt[:, :, 1] - Vt[:, :, 0],
                                      Vt[:, :, 2] - Vt[:, :, 0], dim=-1)
                    nrm = nrm / torch.sqrt((nrm ** 2).sum(-1, keepdim=True) + 1e-18)
                if slopes and Bt is not None:
                    Jc = mano_output_to_joints21(om)
                    P = P + stretch_offsets(Jc, hp_m, Bt, names, slopes, ref_deg,
                                            args.stretch_gate)
                dd = (P + args.standoff * 1e-3 * nrm - Mm_s)[mmk]
                inw = torch.relu(-((Mm_s - P) * nrm).sum(-1))[mmk]
                loss = (dd ** 2).sum(-1).mean() + args.w_inside * (inw ** 2).mean()
                if args.w_smooth > 0 and Tm > 2:
                    loss = loss + args.w_smooth * (
                        (hp_m[2:] - 2 * hp_m[1:-1] + hp_m[:-2]) ** 2).sum(-1).mean()
                    loss = loss + args.w_smooth * (
                        (orient_m[2:] - 2 * orient_m[1:-1] + orient_m[:-2]) ** 2
                    ).sum(-1).mean()
                    loss = loss + args.w_smooth * 1e2 * (
                        (tr_m2[2:] - 2 * tr_m2[1:-1] + tr_m2[:-2]) ** 2).sum(-1).mean()
                if args.biomech > 0 and jl_lo is not None:
                    loss = loss + args.biomech * basis_limit_penalty(hp_m, Bt, jl_lo, jl_hi, args.twist_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"movement loss became non-finite at chunk {ci} iter {it}")
                loss.backward()
                opt.step()
            with torch.no_grad():
                _r = dr if (args.root_only or dd is None) else dd
                resid = _r.norm(dim=-1).mean().item() * 1000
            print(f"  chunk {ci+1}/{n_chunks}  frames {idx[0]}..{idx[-1]}  "
                  f"resid {resid:6.2f} mm")
            GO.append(orient_m.detach().cpu().numpy())
            HP.append(hp_m.detach().cpu().numpy())
            TR.append(tr_m2.detach().cpu().numpy())
            prev_hp = HP[-1][-1].copy()
            prev_go = GO[-1][-1].copy()
            prev_tr = TR[-1][-1].copy()

        GO = np.concatenate(GO); HP = np.concatenate(HP); TR = np.concatenate(TR)

        # residual report over everything
        with torch.no_grad():
            Vall = []
            for ci in range(n_chunks):
                sl = slice(ci * chunk, min((ci + 1) * chunk, len(mf)))
                om = model(global_orient=torch.tensor(GO[sl], device=dev),
                           hand_pose=torch.tensor(HP[sl], device=dev),
                           transl=torch.tensor(TR[sl], device=dev),
                           betas=betas_f.unsqueeze(0).expand(sl.stop - sl.start, -1),
                           return_verts=True)
                Vall.append(om.vertices.cpu().numpy() * 1000)
        Vm_all = np.concatenate(Vall)
        if off_vec is not None:
            with torch.no_grad():
                Jall = []
                for ci in range(n_chunks):
                    sl0 = slice(ci * chunk, min((ci + 1) * chunk, len(mf)))
                    om = model(global_orient=torch.tensor(GO[sl0], device=dev),
                               hand_pose=torch.tensor(HP[sl0], device=dev),
                               transl=torch.tensor(TR[sl0], device=dev),
                               betas=betas_f.unsqueeze(0).expand(sl0.stop - sl0.start, -1),
                               return_verts=True)
                    Jall.append(mano_output_to_joints21(om))
                Jc = torch.cat(Jall)
                R = segment_frames_t(Jc)
                Pm = (torch.stack([Jc[:, sg[0]] for sg in _OFFSET_SEG], 1)
                      + torch.einsum("tpij,pj->tpi", R, off_vec)).cpu().numpy() * 1000
        else:
            Pm = (Vm_all[:, F[fid]] * bary[None, :, :, None]).sum(2)
        if slopes and Bmat is not None:
            with torch.no_grad():
                _J = torch.tensor(np.concatenate([
                    mano_output_to_joints21(model(
                        global_orient=torch.tensor(GO[sl0], device=dev),
                        hand_pose=torch.tensor(HP[sl0], device=dev),
                        transl=torch.tensor(TR[sl0], device=dev),
                        betas=betas_f.unsqueeze(0).expand(sl0.stop - sl0.start, -1),
                        return_verts=True)).cpu().numpy()
                    for sl0 in [slice(ci * chunk, min((ci + 1) * chunk, len(mf)))
                                for ci in range(n_chunks)]]), device=dev)
                Pm = Pm + stretch_offsets(
                    _J, torch.tensor(HP, device=dev), Bt, names, slopes, ref_deg,
                    args.stretch_gate).cpu().numpy() * 1000
        if args.pred_out:
            Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.pred_out, pred_mm=Pm.astype(np.float32), frames=mf,
                     names=np.array(names))
            print(f"saved {args.pred_out}")
        per = np.linalg.norm(Pm - mv[mf][:, cols], axis=-1)
        print("\n--- movement: per-marker mean distance to its attached point ---")
        for kk, nm in enumerate(names):
            v = per[:, kk]
            v = v[np.isfinite(v)]
            if not len(v):          # marker absent for the whole window
                print(f"  {nm:10s}   (never observed)")
                continue
            print(f"  {nm:10s} {v.mean():7.2f} mm   (p95 {np.percentile(v, 95):6.2f})")
        allv = per[np.isfinite(per)]
        print(f"  {'OVERALL':10s} {allv.mean():7.2f} mm   "
              f"(p95 {np.percentile(allv, 95):6.2f})")
        # Split the mean. Palm and MCP markers are satisfied by the ROOT
        # transform alone, so they stay low whatever the fingers do -- 7 of 18
        # markers rewarding a hand that never articulates. Ranking configs by
        # the overall mean therefore favours a flat hand, which is exactly the
        # mistake that made an unflexed fit look like the best result.
        root_k = [k for k, nm in enumerate(names)
                  if nm.startswith("Palm") or nm.endswith("1")]
        fing_k = [k for k in range(len(names)) if k not in root_k]
        rv = per[:, root_k][np.isfinite(per[:, root_k])]
        fv = per[:, fing_k][np.isfinite(per[:, fing_k])]
        print(f"  {'  palm+MCP':10s} {rv.mean():7.2f} mm  (root-determined)")
        print(f"  {'  FINGERS':10s} {fv.mean():7.2f} mm  <- the number that "
              f"actually says whether the fingers articulate")

        # Did the joints actually bend? Residual alone cannot tell: a hand
        # that never flexes can still post a mediocre number. Compare the
        # FITTED flexion against what the markers plainly show.
        if Bmat is not None:
            hpj = HP.reshape(len(HP), 15, 3)
            fit_flex = np.einsum("tjk,jk->tj", hpj, Bmat[:, :, 0])
            print("\n--- flexion: fitted vs measured off the markers (deg) ---")
            print(f"{'joint':9s} {'fitted mean':>12s} {'fitted max':>11s}")
            for ji, nm in enumerate(_FINGER_JOINTS):
                d = np.degrees(fit_flex[:, ji])
                print(f"  {nm:9s} {d.mean():12.1f} {d.max():11.1f}")

        if args.npz_out:
            # Same layout as the stock fitter's FitResult npz, so
            # scripts/mano/animate_mesh.py reads it unchanged -- except
            # hand_pose is 45-DOF axis-angle here, which that script now
            # detects from the array width rather than assuming PCA.
            full_go = np.zeros((mv.shape[0], 3), np.float32)
            full_hp = np.zeros((mv.shape[0], HP.shape[1]), np.float32)
            full_tr = np.zeros((mv.shape[0], 3), np.float32)
            full_go[mf] = GO; full_hp[mf] = HP; full_tr[mf] = TR
            Path(args.npz_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.npz_out, global_orient=full_go, hand_pose=full_hp,
                     transl=full_tr, betas=betas_f.cpu().numpy().astype(np.float32),
                     frames_fitted=mf.astype(np.int64))
            print(f"saved {args.npz_out}")

        mf_local = mf
        Tm = len(mf)
        if args.movement_out:
            kk0 = Tm // 2
            Vv, mkv = Vm_all[kk0], mv[mf[kk0]][cols]
            figm = go.Figure(data=[
                go.Mesh3d(x=Vv[:, 0], y=Vv[:, 1], z=Vv[:, 2],
                          i=F[:, 0], j=F[:, 1], k=F[:, 2], color="#e8a0a0",
                          opacity=1.0, flatshading=True,
                          lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05),
                          name="MANO (static-calibrated)"),
                go.Scatter3d(x=mkv[:, 0], y=mkv[:, 1], z=mkv[:, 2],
                             mode="markers+text", text=names,
                             marker=dict(size=6, color="#1b3a5c"),
                             textposition="top center", name="Vicon markers"),
                go.Scatter3d(
                    x=np.ravel([[a[0], b[0], None] for a, b in zip(mkv, Pm[kk0])]),
                    y=np.ravel([[a[1], b[1], None] for a, b in zip(mkv, Pm[kk0])]),
                    z=np.ravel([[a[2], b[2], None] for a, b in zip(mkv, Pm[kk0])]),
                    mode="lines", line=dict(color="#16a34a", width=5),
                    name="marker -> attached point"),
            ])
            figm.update_layout(
                title=f"Movement frame {mf[kk0]} � mean "
                      f"{per[kk0][np.isfinite(per[kk0])].mean():.1f} mm",
                scene=dict(aspectmode="data",
                           camera=dict(projection=dict(type="orthographic"))),
                width=1000, height=850)
            Path(args.movement_out).parent.mkdir(parents=True, exist_ok=True)
            figm.write_html(args.movement_out, include_plotlyjs=True)
            print(f"saved {args.movement_out}")



if __name__ == "__main__":
    main()
