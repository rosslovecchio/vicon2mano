"""Shared loading/fitting logic for validate_gmm_labeler.py and
animate_gmm_relabel.py, factored out once both needed it to avoid the two
scripts silently drifting apart on how they compute anchor validity,
reference frames, etc.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.loader import load_csv                # noqa: E402
from vicon2mano import gmm_labeler as gl               # noqa: E402
import relabel_with_mano as rwm                        # noqa: E402
import label_marker_quality as lmq                     # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "gmm_labeler"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_CORRECT = 2
ANCHOR_LABEL_NAMES = ("Palm2", "Palm3", "Thumb1")
TARGET_LABEL_NAMES = ("Ring1", "Pinky1")
DOCUMENTED_SWAP_FRAME = 42789


def _contiguous_run(bad_mask: np.ndarray, t: int) -> tuple[int, int]:
    lo = hi = t
    while lo > 0 and bad_mask[lo - 1]:
        lo -= 1
    while hi < len(bad_mask) - 1 and bad_mask[hi + 1]:
        hi += 1
    return lo, hi


@dataclass
class TrialContext:
    markers: np.ndarray
    labels: list[str]
    anchor_labels: tuple[str, str, str]
    marker_idxs: list[int]        # [Ring1, Pinky1]
    status: np.ndarray            # (T, N) cascade verdict
    anchor_valid: np.ndarray      # (T,) bool -- all 3 anchors CORRECT
    run_lo: int
    run_hi: int
    ref_frames: np.ndarray


def load_p7_ring1_pinky1(*, buf: int = 500, max_ref: int = 3000) -> TrialContext:
    trial_path, *_ = rwm.find_trial_csv("P7", "Trial 1 Hands only")
    print(f"Loading {trial_path}")
    markers, labels = load_csv(str(trial_path))
    T = markers.shape[0]

    def find(name: str) -> int:
        return [i for i, l in enumerate(labels) if l.endswith(":" + name)][0]

    anchor_labels = tuple(labels[find(n)] for n in ANCHOR_LABEL_NAMES)
    anchor_idxs = [find(n) for n in ANCHOR_LABEL_NAMES]
    marker_idxs = [find(n) for n in TARGET_LABEL_NAMES]
    idx_ring, idx_pinky = marker_idxs

    ref = lmq.load_ref_ranges_csv(str(rwm.REF_CSV), "P7", "Trial 1 Hands only")
    ref = ref[ref < T]
    print("Running cascade...")
    status, _bones = lmq.label_quality_cascade(markers, labels, ref, static_markers=None)

    bad_ring = status[:, idx_ring] != CASCADE_CORRECT
    run_lo, run_hi = _contiguous_run(bad_ring, DOCUMENTED_SWAP_FRAME)
    print(f"Cascade's 'Ring1 INCORRECT' run containing frame "
          f"{DOCUMENTED_SWAP_FRAME}: [{run_lo}, {run_hi}] ({run_hi - run_lo + 1} frames)")

    anchor_valid = (status[:, anchor_idxs] == CASCADE_CORRECT).all(axis=1)
    bad_anchor_frames = int((~anchor_valid[run_lo:run_hi + 1]).sum())
    print(f"  anchor ({', '.join(ANCHOR_LABEL_NAMES)}) invalid on "
          f"{bad_anchor_frames}/{run_hi - run_lo + 1} frames of the flagged run "
          f"({(~anchor_valid).sum()} invalid overall)")

    both_ok = np.flatnonzero(
        (status[:, idx_ring] == CASCADE_CORRECT) & (status[:, idx_pinky] == CASCADE_CORRECT)
        & anchor_valid
    )
    ref_frames = both_ok[(both_ok < run_lo - buf) | (both_ok > run_hi + buf)]
    if len(ref_frames) > max_ref:
        ref_frames = ref_frames[np.linspace(0, len(ref_frames) - 1, max_ref).astype(int)]
    print(f"  {len(ref_frames)} cascade-verified (anchors + both targets) reference frames")

    return TrialContext(markers, labels, anchor_labels, marker_idxs, status,
                        anchor_valid, run_lo, run_hi, ref_frames)


def fit_and_relabel_window(
    ctx: TrialContext, lo: int, hi: int, *, n_hypotheses: int = 5,
    process_var: float = 25.0, obs_var: float = 100.0,
) -> tuple[np.ndarray, int]:
    """Fit on ``ctx.ref_frames`` (anchor-gated), relabel window [lo, hi).

    Returns ``(is_swapped, init_frame_global)``. Anchor-invalid frames
    within the window are treated as fully occluded (NaN local position
    for every marker that frame), not skipped -- the Kalman extrapolation
    already built into ``viterbi_select`` carries the trajectory across
    them the same way it does a genuine marker occlusion. Uses only the
    public ``GMMLabeler``/``anchor_valid`` API (vicon2mano/gmm_labeler.py),
    not any internal helper.
    """
    markers, labels = ctx.markers, ctx.labels

    labeler = gl.GMMLabeler.fit(
        markers, labels, ctx.anchor_labels, ctx.ref_frames, ctx.marker_idxs,
        n_components=3, anchor_valid=ctx.anchor_valid)
    assert labeler.marker_idxs == ctx.marker_idxs, (
        f"one of {ctx.marker_idxs} had too few reference samples")

    window_valid = ctx.anchor_valid[lo:hi]
    check_idxs = [labels.index(a) for a in ctx.anchor_labels] + ctx.marker_idxs
    present = window_valid & np.isfinite(markers[lo:hi][:, check_idxs]).all(axis=(1, 2))
    if not present.any():
        raise SystemExit(f"No frame in [{lo}, {hi}) has both trustworthy anchors and "
                          f"both targets present -- cannot pick an init_frame")
    init_frame = int(np.flatnonzero(present)[0])

    result = labeler.relabel(
        markers[lo:hi], labels, n_hypotheses=n_hypotheses,
        process_var=process_var, obs_var=obs_var,
        anchor_valid=window_valid, init_frame=init_frame)
    is_swapped = result[:, 0] == 1
    return is_swapped, lo + init_frame
