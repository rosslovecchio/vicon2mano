"""Calibrate the marker library to a new participant from their static.

The synthetic data's subject-specific parts are the hand **shape** and each
marker's **mean offset** from its MANO attachment point (skin standoff and
where on the finger it was stuck). Everything that varies with movement --
poses, skin sliding with flexion, placement noise -- is taken from the base
library (P10/Trial2), where it was measured on real motion. A static holds
one still pose, so it can supply exactly the first two and nothing else.

Steps:

1. ``static_cloud``: the stillest, most complete window of the static ->
   one median position per marker.
2. ``fit_static``: fit MANO shape + pose to that cloud, predicting each hand
   marker as its base-library attachment plus the base library's median
   offset. Markers that still sit far off after the fit are flagged as
   suspect (likely mislabelled in the static -- P9's palm is the documented
   case) and refit without. Attachment faces are kept from the base
   library: re-choosing them from one pose would trade pose against
   placement.
3. ``calibrated_library``: the base library with the participant's shape,
   and each calibrated marker's residuals re-centred on its offset measured
   in the static (base dynamics kept, base mean replaced).

Forearm markers are not on MANO and their offset from the wrist depends on
the wrist angle, which differs between any static and the base library's
motion; they are left out of the fit and keep the base library's residuals.

No imports outside numpy, torch and this strategy; the MANO layer is passed in.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from vicon2mano.strategies.soma.library import MarkerLibrary, face_frames

ROOT_PREFIX = ("Palm",)
ROOT_SUFFIX = "1"            # MCP markers: <Finger>1


def is_forearm(name: str) -> bool:
    return name.startswith("Forearm")


def is_root(name: str) -> bool:
    return name.startswith(ROOT_PREFIX) or (name.endswith(ROOT_SUFFIX) and not is_forearm(name))


def static_cloud(markers: np.ndarray, n: int = 200) -> tuple[np.ndarray, dict]:
    """(M, 3) median marker positions over the stillest, most complete
    ``n``-frame window of a static recording (NaN = never present there)."""
    T = len(markers)
    n = min(n, T)
    pres = np.isfinite(markers).all(-1)                          # (T, M)
    speed = np.nanmedian(np.linalg.norm(np.diff(markers, axis=0), axis=-1), axis=1)
    speed = np.r_[speed, speed[-1:]]
    speed = np.where(np.isfinite(speed), speed, np.inf)
    best, best_key = 0, None
    for s in range(0, T - n + 1, max(1, n // 10)):
        key = (-pres[s:s + n].sum(), np.median(speed[s:s + n]))
        if best_key is None or key < best_key:
            best, best_key = s, key
    w = markers[best:best + n]
    with np.errstate(all="ignore"):
        cloud = np.nanmedian(w, axis=0)
    info = dict(start=best, n=n, present=pres[best:best + n].mean(0),
                motion_mm=float(best_key[1]))
    return cloud, info


def _kabsch(X: np.ndarray, Y: np.ndarray):
    """R, t with R @ x + t ~= y."""
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cy - R @ cx


def _axis_angle(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def fit_static(model, lib: MarkerLibrary, cloud_mm: np.ndarray, *, device: str = "cpu",
               iters_root: int = 300, iters: int = 1500, w_beta: float = 2.0,
               w_pose: float = 20.0, suspect_mm: float = 15.0, rounds: int = 2,
               offset_only: tuple[str, ...] = ()) -> dict:
    """Fit MANO shape and pose to one static cloud (mm). See module docstring.

    ``w_pose`` (mm^2 per rad^2) holds the static near a flat hand. At 1.0 the
    fit kinked the fingertips to land attachment points on markers: on P12's
    flat static the PIP/DIP joints came out rotated 28-63 deg (index and pinky
    tips visibly broken). 20 leaves the 4-17 deg of a relaxed hand at a fit
    error of median 7.0 mm (vs 5.0); 100+ forces it so flat that Palm1 no
    longer fits (flagged suspect). The extra error moves into each marker's
    measured offset, where placement differences belong.

    Returns betas (10,), hand_pose (45,), global_orient (3,), transl (3,) m,
    the posed vertices (778, 3) mm, per-marker fit error ``err_mm`` (M,),
    the static offset ``r_s`` (M, 3) in each attachment face's frame (NaN
    where absent, suspect, or forearm not listed in ``offset_only``), and the
    ``suspects`` (label list).

    ``offset_only``: markers left out of the fit whose static offsets are
    still recorded. For a participant whose marker LAYOUT differs from the
    base subject's, those markers are not mislabelled -- they sit elsewhere,
    and fitting them would drag the hand; recording their offset is exactly
    what calibration is for. P1/P2: Forearm1-Forearm2 52-56 mm apart vs P10's
    25, Palm2 ~20 mm more distal, Thumb1-Thumb2 49-58 vs 40 mm. A forearm
    offset also bakes in that recording's wrist angle, a smaller error than
    the wrong layout.
    """
    model = model.to(device)
    F = np.asarray(model.faces, dtype=np.int64)
    M = len(lib.labels)
    rbar = torch.tensor(np.nanmedian(lib.residual, axis=0), dtype=torch.float32, device=device)
    X = torch.tensor(np.nan_to_num(cloud_mm), dtype=torch.float32, device=device)
    present = np.isfinite(cloud_mm).all(-1)
    hand = np.array([not is_forearm(n) for n in lib.labels])
    root = np.array([is_root(n) for n in lib.labels])
    oo = np.array([n in offset_only for n in lib.labels])
    unknown = set(offset_only) - set(lib.labels)
    if unknown:
        raise ValueError(f"offset_only names not in the library: {sorted(unknown)}")

    def forward(betas, pose, orient, transl):
        o = model(betas=betas[None], hand_pose=pose[None], global_orient=orient[None],
                  transl=transl[None])
        V = o.vertices[0] * 1000.0
        P, R = face_frames(V, F, lib.face_idx, lib.bary)
        return V, P + torch.einsum("mij,mj->mi", R, rbar), o.joints[0, 0] * 1000.0

    # initial orientation: Kabsch of rest-pose predictions onto the palm/MCP markers
    z = lambda k: torch.zeros(k, device=device)
    with torch.no_grad():
        _, P0, J0 = forward(z(10), z(45), z(3), z(3))
    use = root & present & ~oo
    R0, t0 = _kabsch(P0[use].cpu().numpy(), cloud_mm[use])
    orient = torch.tensor(_axis_angle(R0), dtype=torch.float32, device=device, requires_grad=True)
    transl = torch.tensor((t0 + R0 @ J0.cpu().numpy() - J0.cpu().numpy()) / 1000.0,
                          dtype=torch.float32, device=device, requires_grad=True)
    betas = z(10).requires_grad_(True)
    pose = z(45).requires_grad_(True)

    active = present & hand & ~oo
    suspects: list[str] = []
    for rnd in range(rounds + 1):
        act = torch.tensor(active, device=device)
        stages = [([orient, transl], iters_root, torch.tensor(active & root, device=device)),
                  ([orient, transl, betas, pose], iters, act)]
        for params, n_it, msk in stages:
            opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in
                                    zip(params, [1e-2, 5e-4, 2e-2, 2e-2][:len(params)])])
            for _ in range(n_it):
                opt.zero_grad()
                _, Pm, _ = forward(betas, pose, orient, transl)
                loss = ((Pm - X) ** 2).sum(-1)[msk].mean()
                if len(params) > 2:
                    loss = loss + w_beta * (betas ** 2).sum() + w_pose * (pose ** 2).sum()
                loss.backward()
                opt.step()
        with torch.no_grad():
            V, Pm, _ = forward(betas, pose, orient, transl)
            err = (Pm - X).norm(dim=-1).cpu().numpy()
        new = [m for m in np.flatnonzero(active) if err[m] > suspect_mm]
        if not new or rnd == rounds:
            break
        worst = max(new, key=lambda m: err[m])        # drop the worst, refit
        suspects.append(lib.labels[worst])
        active[worst] = False

    with torch.no_grad():
        Vn = V.cpu().numpy()
        P, R = face_frames(Vn, F, lib.face_idx, lib.bary)
        r_s = np.einsum("mji,mj->mi", R, cloud_mm - P)
    keep = present & ((hand & np.array([n not in suspects for n in lib.labels])) | oo)
    r_s[~keep] = np.nan
    err[~present] = np.nan
    return dict(betas=betas.detach().cpu().numpy(), hand_pose=pose.detach().cpu().numpy(),
                global_orient=orient.detach().cpu().numpy(), transl=transl.detach().cpu().numpy(),
                vertices_mm=Vn, err_mm=err, r_s=r_s, suspects=suspects)


def calibrated_library(lib: MarkerLibrary, betas: np.ndarray, r_s: np.ndarray,
                       miss_rate: np.ndarray | None = None, source: str = "") -> MarkerLibrary:
    """Base library with the participant's shape and, for every marker with a
    finite static offset ``r_s``, residuals re-centred on it:
    ``residual[n, m] - median_n(residual[:, m]) + r_s[m]``."""
    res = lib.residual.copy()
    med = np.nanmedian(res, axis=0)
    cal = np.isfinite(r_s).all(-1)
    res[:, cal] = res[:, cal] - med[cal][None] + r_s[cal][None]
    return replace(lib, betas=np.asarray(betas, np.float32), residual=res.astype(np.float32),
                   miss_rate=lib.miss_rate if miss_rate is None else np.asarray(miss_rate),
                   source=f"{source} | base: {lib.source}")
