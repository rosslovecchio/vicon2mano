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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv                # noqa: E402
from vicon2mano.strategies.gmm import labeler as gl               # noqa: E402
from vicon2mano.core import dataset as rwm
from vicon2mano.strategies.cascade import quality_cascade as lmq

OUT_DIR = REPO_ROOT / "results" / "gmm" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_CORRECT = 2
# Palm1/Palm2/Palm3 are the true rigid palm plate (CLAUDE.md's cascade
# "palm gate"). The original choice (Palm2/Palm3/Thumb1) was swapped in
# for this one because it matches the cascade's own de-facto anchor set,
# but on P7/Trial1_handsonly specifically the cascade distrusts Thumb1
# often enough (~78% of the whole recording) that joint anchor validity
# with Palm1/Palm2/Palm3 is nearly 2x higher (21.4% vs ~12% of frames) --
# still physically correct (all 3 on the same rigid segment, unlike mixing
# in a Forearm marker, which is a different, non-rigidly-attached segment).
ANCHOR_LABEL_NAMES = ("Palm1", "Palm2", "Palm3")
TARGET_LABEL_NAMES = ("Ring1", "Pinky1")
DOCUMENTED_SWAP_FRAME = 42789

# Empirically measured on cascade-verified frames of this trial: Ring1/
# Pinky1 move ~0.26mm/frame at the median, ~0.9mm/frame at the 90th
# percentile, in the Palm1/2/3 local frame. The paper's Section 3.2 calls
# for tuning these against the training data rather than guessing -- the
# first draft's process_var=25/obs_var=100 (implying 5-10mm/frame, 20-40x
# too loose) barely penalised an incorrect transition at all, which is
# why that run's verdict flip-flopped so much.
DEFAULT_PROCESS_VAR = 0.5    # mm^2, per-step motion
DEFAULT_OBS_VAR = 1.0        # mm^2, measurement/reconstruction noise


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
    process_var: float = DEFAULT_PROCESS_VAR, obs_var: float = DEFAULT_OBS_VAR,
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
