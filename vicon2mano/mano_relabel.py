"""Relabel *incorrectly labelled* markers using a subject-specific MANO model.

The idea, in three steps:

1. **Subject shape.** Fit MANO ``betas`` once per subject on frames the
   quality cascade verified as fully correct (plus the static trial when
   available). Shape is well determined there, and freezing it removes 10
   free parameters from every later fit.
2. **Skin offsets.** A marker sits on the skin, not at the joint centre —
   the repo's own measured residual is ~15-22mm. Predicting a marker
   position straight from a fitted joint inherits all of that. So estimate
   each marker's constant offset from its joint, expressed in a local frame
   that rotates with the hand, again from verified frames only. This turns
   a generic ~20mm error into a per-marker constant and is what makes the
   prediction sharp enough to be worth anything.
3. **Relabel.** In a frame with flagged markers, fit the frozen-shape model
   to the *trusted* markers only (the fitter already masks non-finite
   targets out of its loss, so a partial set works natively), predict where
   every marker should be, and solve a small assignment problem between the
   flagged marker *positions* and the predicted positions of the labels
   they could plausibly belong to.

Scope: this module **only reassigns labels between markers that are
present**. It never invents a position for a missing marker — gap filling is
a separate problem with a different error budget (see the module docstring
discussion in ``scripts/relabel_with_mano.py``).

Circularity warning: only ever fit on markers the cascade verified, and
treat the model's prediction as evidence about the *flagged* markers only.
Fitting on a mislabelled marker bends the model toward the wrong data, which
then "confirms" the wrong label.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from .correspondence import MANO_JOINT_NAMES, label_seed

# Status codes produced by scripts/label_marker_quality.label_quality_cascade
MISSING, INCORRECT, CORRECT = 0, 1, 2


def marker_joint_map(labels: list[str], side: str | None = None) -> dict[int, int]:
    """Map marker index -> MANO joint index, via the existing label seed.

    Returns only the markers that correspond to a MANO joint; Forearm and
    the non-wrist Palm markers have no MANO counterpart and are excluded
    (MANO models the hand, not the arm).
    """
    seed = label_seed(labels, side=side)
    if not seed:
        return {}
    joint_of_name = {n: i for i, n in enumerate(MANO_JOINT_NAMES)}
    return {m_idx: joint_of_name[jname] for jname, m_idx in seed.items()}


def verified_frames(status: np.ndarray, marker_idxs: list[int], *,
                     max_frames: int = 400) -> np.ndarray:
    """Frames where *every* marker in ``marker_idxs`` is CORRECT.

    Strict, and in practice often collapses onto a single motionless window
    (during real motion something is nearly always flagged) — which is fine
    for a sanity check but useless for calibration, because it pins every
    offset to one pose. Prefer ``select_fit_frames`` for that.
    """
    if not marker_idxs:
        return np.array([], dtype=int)
    idx = np.flatnonzero((status[:, marker_idxs] == CORRECT).all(axis=1))
    if idx.size > max_frames:
        idx = idx[np.linspace(0, idx.size - 1, max_frames).astype(int)]
    return idx


def select_fit_frames(
    status: np.ndarray,
    marker_idxs: list[int],
    *,
    min_correct: int = 12,
    max_frames: int = 300,
    n_bins: int = 30,
) -> np.ndarray:
    """Frames good enough to fit a pose on, spread across the recording.

    Requires only ``min_correct`` of the mapped markers to be CORRECT (the
    fitter masks the rest out of its loss anyway), and draws them evenly
    from ``n_bins`` time bins spanning the whole trial. Spreading matters
    more than the count: offsets calibrated inside one quiet window are
    only valid for that one pose, and the whole point is to predict during
    motion.
    """
    if not marker_idxs:
        return np.array([], dtype=int)
    n_ok = (status[:, marker_idxs] == CORRECT).sum(axis=1)
    idx = np.flatnonzero(n_ok >= min_correct)
    if idx.size == 0:
        return idx

    T = status.shape[0]
    per_bin = max(1, max_frames // n_bins)
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = idx[(idx >= lo) & (idx < hi)]
        if in_bin.size == 0:
            continue
        take = min(per_bin, in_bin.size)
        picked.append(in_bin[np.linspace(0, in_bin.size - 1, take).astype(int)])
    if not picked:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(picked))


def fit_frames(
    markers_mm: np.ndarray,     # (T, N, 3)
    labels: list[str],
    frames: np.ndarray,
    *,
    mano_dir: str,
    side: str = "right",
    betas: np.ndarray | None = None,
    n_iters_shape: int = 100,
    n_iters_pose: int = 250,
    verbose: bool = False,
):
    """Fit MANO to an arbitrary set of frames. Returns the ``FitResult``.

    **Temporal terms are disabled**, and that is not optional here.
    ``MANOFitter`` penalises pose difference (``w_temp``) and joint-space
    acceleration (``w_accel``) between *consecutive rows of the array it is
    given*. Those terms are correct for a contiguous recording, but ``frames``
    here is deliberately a scattered, pose-spanning sample — consecutive rows
    are seconds or minutes apart, so the penalties end up fighting the data
    term across unrelated poses. Measured cost of getting this wrong, on
    P8/Trial1_hoi: fit residual 17.3mm -> 36.4mm, and the calibrated offset
    spread blows up from 8.7mm to 40.0mm, which silently destroys the whole
    point of the offset model.

    Pass ``betas`` to freeze a previously estimated subject shape (skips the
    shape stage entirely).
    """
    from .fitter import FitConfig, MANOFitter   # local: torch import is slow

    cfg = FitConfig(
        mano_model_path=mano_dir,
        hand_side=side,
        n_iters_shape=0 if betas is not None else n_iters_shape,
        n_iters_pose=n_iters_pose,
        w_temp=0.0,
        w_accel=0.0,
    )
    return MANOFitter(cfg).fit(markers_mm[frames], labels, verbose=verbose)


def calibrate_marker_offsets(
    markers_mm: np.ndarray,      # (T, N, 3) full recording, mm
    frames: np.ndarray,          # frames the fit was run on
    fit_joints_m: np.ndarray,    # (len(frames), 21, 3) fitted joints, metres
    m2j: dict[int, int],
    *,
    status: np.ndarray | None = None,   # (T, N) cascade status, optional
    marker_scale: float = 1e-3,
    min_samples: int = 20,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Per-marker offset from its MANO joint, in that joint's *segment* frame.

    Each marker is calibrated only on the subset of ``frames`` where the
    cascade marked *that marker* CORRECT (when ``status`` is given), rather
    than demanding every marker be correct at once — which in practice
    collapses the usable set onto a single motionless window and calibrates
    the offsets for one pose only.

    Returns ({marker: (3,) offset in metres}, {marker: RMS spread in mm}).
    The spread is the diagnostic that matters: a genuinely pose-invariant
    offset has a small spread, and a large one means the model doesn't hold
    for that marker and its prediction should not be trusted.
    """
    joints = fit_joints_m                      # (F, 21, 3), metres
    mk = markers_mm[frames] * marker_scale     # (F, N, 3), metres
    R = _segment_frames(joints)                # (F, 21, 3, 3)

    offsets: dict[int, np.ndarray] = {}
    spreads: dict[int, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m_idx, j_idx in m2j.items():
            delta = mk[:, m_idx] - joints[:, j_idx]                  # world
            local = np.einsum("fab,fb->fa", R[:, j_idx], delta)      # segment frame
            good = np.isfinite(local).all(axis=1)
            if status is not None:
                good &= status[frames, m_idx] == CORRECT
            if good.sum() < min_samples:
                continue
            offsets[m_idx] = np.nanmedian(local[good], axis=0)
            spreads[m_idx] = float(np.linalg.norm(np.nanstd(local[good], axis=0)) * 1000)
    return offsets, spreads


# MANO 21-joint tree in this repo's order: wrist, then thumb/index/middle/
# ring/pinky as (mcp, pip, dip, tip). Parent of each finger base is the wrist.
_PARENT = {
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}
_CHILD = {p: c for c, p in _PARENT.items() if p != 0}   # within-finger links only


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _palm_frames(joints: np.ndarray) -> np.ndarray:
    """(F, 3, 3) palm-rigid frame; rows are the local axes."""
    o = joints[:, 0]
    x = _normalise(joints[:, 9] - o)
    y = joints[:, 5] - o
    z = _normalise(np.cross(x, y))
    return np.stack([x, np.cross(z, x), z], axis=1)


def _segment_frames(joints: np.ndarray) -> np.ndarray:
    """(F, 21, 3, 3) — a local frame per joint, aligned to *its own bone*.

    A marker is taped to a bone segment, so its offset is only genuinely
    pose-invariant in a frame that rotates with that segment. Expressing it
    in the palm frame instead (the obvious first guess) silently bakes in
    the finger's flexion at calibration time, which then re-appears as
    prediction error at any other pose.

    x runs along the bone (joint -> its child, or parent -> joint at a
    fingertip); the parent direction fixes the roll. The wrist and any joint
    without usable neighbours fall back to the palm frame.
    """
    F = joints.shape[0]
    palm = _palm_frames(joints)
    out = np.repeat(palm[:, None, :, :], 21, axis=1)

    for j in range(21):
        child, parent = _CHILD.get(j), _PARENT.get(j)
        if child is not None:
            v1 = joints[:, child] - joints[:, j]
        elif parent is not None:
            v1 = joints[:, j] - joints[:, parent]
        else:
            continue                                   # wrist -> palm frame
        ref = (joints[:, j] - joints[:, parent]) if parent is not None else palm[:, 1]
        x = _normalise(v1)
        z = np.cross(x, ref)
        n = np.linalg.norm(z, axis=-1, keepdims=True)
        # degenerate (bone collinear with its parent, i.e. finger straight):
        # fall back to the palm frame's normal to fix the roll
        z = np.where(n > 1e-8, z / (n + 1e-12), palm[:, 2])
        out[:, j] = np.stack([x, np.cross(z, x), z], axis=1)
    return out


def predict_marker_positions(
    joints_m: np.ndarray,             # (F, 21, 3) metres
    offsets: dict[int, np.ndarray],
    m2j: dict[int, int],
    n_markers: int,
) -> np.ndarray:
    """Expected position of every mapped marker, (F, n_markers, 3) metres.

    Markers with no MANO counterpart or no calibrated offset are NaN.
    """
    F = joints_m.shape[0]
    out = np.full((F, n_markers, 3), np.nan)
    R = _segment_frames(joints_m)                    # (F, 21, 3, 3)
    for m_idx, j_idx in m2j.items():
        off = offsets.get(m_idx)
        if off is None:
            continue
        # local -> world is R^T applied to the offset
        world = np.einsum("fab,b->fa", np.swapaxes(R[:, j_idx], 1, 2), off)
        out[:, m_idx] = joints_m[:, j_idx] + world
    return out


def model_reliability(
    markers_mm: np.ndarray,       # (T, N, 3)
    status: np.ndarray,           # (T, N) cascade status
    predicted_m: np.ndarray,      # (F, N, 3) predictions for `frames`
    frames: np.ndarray,           # frames `predicted_m` corresponds to
    m2j: dict[int, int],
    *,
    stride: int = 7,
    min_samples: int = 30,
) -> dict:
    """Can this trial's fitted model tell neighbouring markers apart?

    Two measured quantities decide it:

    * ``p95_err_mm`` -- ``|marker - prediction|`` at the 95th percentile over
      markers the cascade calls CORRECT. The label is known right there, so
      the distance is model error, not mislabelling.
    * ``spacing_mm`` -- median nearest-neighbour distance between markers.

    Their ratio is what predicts whether relabelling can work, measured over
    a gate sweep on three trials:

        P7   p95 10.2mm / spacing 25.7mm = 0.40  ->  +12.3% correctness
        P8   p95 17.5mm / spacing 22.8mm = 0.77  ->   +1.0%
        P10  p95 50.2mm / spacing 28.6mm = 1.75  ->   +1.3%

    P10 is the cautionary case: calibrated from a static recording alone, its
    model is less accurate than the distance between adjacent markers, so the
    13796 reassignments it proposes are largely arbitrary — only 18% of them
    verify, against 86% on P7. A ratio near or above 1 means the predictions
    carry no usable identity information and the output should not be trusted
    regardless of how confident the assignment looks.

    Note this is deliberately *not* used to set ``max_dist``. A sweep of
    8-40mm showed tighter gates are uniformly worse (P7: 78.3% at 8mm vs
    86.2% at 30mm) because the assignment is solved jointly across all
    flagged markers in a frame rather than as independent pairwise choices,
    so the naive "must be confident to better than spacing/2" bound does not
    apply.
    """
    err = []
    for k in range(0, len(frames), stride):
        t = int(frames[k])
        P, M = predicted_m[k], markers_mm[t] * 1e-3
        for m in m2j:
            if status[t, m] == CORRECT and np.isfinite(P[m]).all() and np.isfinite(M[m]).all():
                err.append(np.linalg.norm(M[m] - P[m]) * 1000)

    nn = []
    step = max(1, len(frames) // 50)
    for k in range(0, len(frames), step):
        M = markers_mm[int(frames[k])]
        idx = [m for m in m2j if np.isfinite(M[m]).all()]
        for a in idx:
            d = [np.linalg.norm(M[a] - M[b]) for b in idx if b != a]
            if d:
                nn.append(min(d))

    p95 = float(np.percentile(err, 95)) if err else float("inf")
    spacing = float(np.median(nn)) if nn else float("nan")
    ratio = p95 / spacing if spacing and np.isfinite(spacing) else float("inf")

    # "Cannot be measured" is a different failure from "measured and bad", and
    # conflating them hides a real data problem behind an apparent model
    # problem. P9/Trial2_hoi reads 16.4% correct overall, but only 0.24% of
    # the MANO-mapped (hand) markers are correct -- that 16.4% is almost
    # entirely forearm and palm markers, which MANO does not model. With no
    # verified hand markers there is nothing to fit a pose from and nothing to
    # score a prediction against, so no calibration source could rescue it.
    if len(err) < min_samples:
        verdict = "unmeasurable"
    elif ratio < 0.55:
        verdict = "good"
    elif ratio < 0.9:
        verdict = "marginal"
    else:
        verdict = "unusable"
    return dict(p95_err_mm=p95, median_err_mm=float(np.median(err)) if err else float("nan"),
                spacing_mm=spacing, ratio=ratio, n_err=len(err), verdict=verdict)


def relabel_frame(
    frame_markers_m: np.ndarray,   # (N, 3) metres
    predicted_m: np.ndarray,       # (N, 3) metres, NaN where unpredictable
    candidate_mask: np.ndarray,    # (N,) bool — observed positions eligible to move
    slot_mask: np.ndarray | None = None,   # (N,) bool — labels eligible to receive a
                                            # new position; None => same as candidate_mask
    *,
    max_dist_m: float = 0.030,
) -> tuple[dict[int, int], dict[int, float]]:
    """Reassign marker *positions* to the labels they best fit.

    Solves a small assignment problem between the observed positions allowed
    by ``candidate_mask`` and the predicted positions of the labels allowed
    by ``slot_mask``. Only pairs within ``max_dist_m`` are considered, and the
    identity mapping is omitted.

    The two original callers (cascade-flagged relabelling) pass the same
    ``flagged`` mask for both — a flagged position may only be reassigned to
    another flagged label. Label-agnostic fitting instead passes "every
    present marker" for both, so a marker can move even if its own label was
    never flagged, as long as the model prefers a different position for it.

    Returns ({marker_slot: source_marker_slot}, {marker_slot: distance_m}) —
    "the data now under label A should come from the position currently
    under label B", plus how far that position sat from where the model
    expected A. The distance is the confidence of the move and separates
    cleanly in practice: on P7/Trial1_handsonly frame 42789 the two accepted
    moves landed at 5.5mm and 5.9mm while every rejected option was 33mm or
    worse, against a model accurate to ~12mm at the 95th percentile.
    """
    if slot_mask is None:
        slot_mask = candidate_mask
    cand = np.flatnonzero(candidate_mask & np.isfinite(frame_markers_m).all(axis=1))
    slots = np.flatnonzero(slot_mask & np.isfinite(predicted_m).all(axis=1))
    if cand.size == 0 or slots.size == 0:
        return {}, {}

    cost = np.linalg.norm(
        predicted_m[slots][:, None, :] - frame_markers_m[cand][None, :, :], axis=-1
    )                                                    # (slots, cand)
    big = max_dist_m * 10
    cost_gated = np.where(cost > max_dist_m, big, cost)

    rows, cols = linear_sum_assignment(cost_gated)
    out: dict[int, int] = {}
    dists: dict[int, float] = {}
    for r, c in zip(rows, cols):
        if cost_gated[r, c] >= big:
            continue
        slot, src = int(slots[r]), int(cand[c])
        if slot != src:
            out[slot] = src
            dists[slot] = float(cost[r, c])
    return out, dists


def relabel_frame_iterative(
    markers_mm: np.ndarray,        # (T, N, 3) full recording
    t: int,
    labels: list[str],
    m2j: dict[int, int],
    offsets: dict[int, np.ndarray],
    betas: np.ndarray,
    *,
    seed_mask: np.ndarray,         # (N,) bool — cascade-CORRECT this frame; iter-1 fit source
    mano_dir: str,
    side: str = "right",
    max_dist_m: float = 0.030,
    max_iters: int = 3,
    min_trusted: int = 6,
) -> tuple[dict[int, int], dict[int, float], int]:
    """Label-agnostic relabelling for one frame, reference/test implementation.

    Iteration 1 fits pose on ``seed_mask`` only (identical to the existing
    cascade-CORRECT seed fit), predicts every mapped marker, then runs
    ``relabel_frame`` with *every present marker* eligible on both sides of
    the assignment — not just cascade-flagged ones. This is the actual
    label-agnostic step: a marker never flagged by the cascade can still be
    moved if the model prefers a different position for it.

    Each further iteration applies the previous mapping to a working copy of
    the frame, refits pose (betas frozen) on that full reassigned frame, and
    reassigns again. Stops early once the mapping stops changing.

    This is a single-frame reference implementation, useful for tests and
    for reasoning about one frame in isolation (see
    ``label_marker_quality.debug_frame_cascade`` for the analogous per-frame
    tool on the cascade side). The trial-level driver in
    ``scripts/relabel_with_mano.py`` does the batched multi-frame equivalent
    for performance — ``fit_frames``/``predict_marker_positions`` are already
    vectorized over frames, so refitting one frame at a time here would be
    far slower across a whole trial.

    Returns (final mapping, final distances, iterations actually run). An
    empty mapping on return means either nothing was reassignable or the
    frame had fewer than ``min_trusted`` seed markers (mirrors the existing
    under-constrained-fit guard in ``_assign_and_verify``).
    """
    if int(seed_mask.sum()) < min_trusted:
        return {}, {}, 0

    present = np.isfinite(markers_mm[t]).all(axis=1)
    working = markers_mm.copy()
    working[t][~seed_mask] = np.nan

    prev_mapping: dict[int, int] | None = None
    mapping: dict[int, int] = {}
    dists: dict[int, float] = {}
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        res = fit_frames(working, labels, np.array([t]), mano_dir=mano_dir,
                         side=side, betas=betas)
        pred_t = predict_marker_positions(res.joints, offsets, m2j, len(labels))[0]
        mapping, dists = relabel_frame(markers_mm[t] * 1e-3, pred_t, present,
                                       max_dist_m=max_dist_m)
        if mapping == prev_mapping:
            break
        prev_mapping = mapping
        working = markers_mm.copy()
        apply_relabel(working, t, mapping)
    return mapping, dists, n_iters


def apply_relabel(markers: np.ndarray, frame: int, mapping: dict[int, int]) -> None:
    """Apply {slot: source} to one frame, in place. Positions are permuted
    between marker columns; no coordinate is ever modified."""
    if not mapping:
        return
    snapshot = markers[frame].copy()
    for slot, src in mapping.items():
        markers[frame, slot] = snapshot[src]
