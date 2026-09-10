#!/usr/bin/env python3
"""Ground-truth validation of strategyGMM: inject a known swap into clean
real marker data and measure whether it is recovered.

Why this and not the P7 frame-42789 episode: that episode turned out to be
a rolling mislabelling cascade across 5 markers with the rigid-frame
anchors themselves corrupted for most of it (see validate_gmm_labeler.py),
so there is no clean ground truth to score against there. Here the data is
real (real hand motion, real Vicon noise, real occlusion gaps) but the
swap is injected, so the ground truth is exact and precision/recall are
meaningful.

Protocol
--------
1. Find a long contiguous stretch where the cascade calls the 3 anchors
   *and* both target markers CORRECT -- i.e. data everyone already agrees
   is clean.
2. Hold that stretch out of GMM training entirely (train on the rest of
   the recording), so the model never sees the evaluation frames.
3. Swap the Ring1/Pinky1 columns for a block of frames in the middle.
4. Run strategyGMM on the corrupted window and compare its per-frame
   verdict to the known injected mask.
5. Write a before/after animation (.gif + self-contained .html player, in
   the style of scripts/animate_fit.py) and a metrics summary.

Usage
-----
python scripts/validate_gmm_injected_swap.py
"""

from __future__ import annotations

import numpy as np

from _gmm_validation_common import (   # noqa: E402
    OUT_DIR, CASCADE_CORRECT, load_p7_ring1_pinky1, gl,
    DEFAULT_PROCESS_VAR, DEFAULT_OBS_VAR,
)
from animate_gmm_relabel import render_before_after   # noqa: E402

WINDOW = 300          # frames to evaluate
SWAP_LEN = 100        # frames of injected swap, centred in the window


def _longest_clean_run(clean: np.ndarray, need: int) -> tuple[int, int]:
    """Longest contiguous True run in ``clean``; raises if shorter than ``need``."""
    best_lo = best_len = cur_lo = cur_len = 0
    for t, ok in enumerate(clean):
        if ok:
            if cur_len == 0:
                cur_lo = t
            cur_len += 1
            if cur_len > best_len:
                best_lo, best_len = cur_lo, cur_len
        else:
            cur_len = 0
    if best_len < need:
        raise SystemExit(f"Longest clean run is {best_len} frames, need {need}")
    return best_lo, best_lo + best_len


def main() -> None:
    ctx = load_p7_ring1_pinky1()
    idx_ring, idx_pinky = ctx.marker_idxs
    markers, labels = ctx.markers, ctx.labels

    clean = (
        ctx.anchor_valid
        & (ctx.status[:, idx_ring] == CASCADE_CORRECT)
        & (ctx.status[:, idx_pinky] == CASCADE_CORRECT)
        & np.isfinite(markers[:, ctx.marker_idxs]).all(axis=(1, 2))
    )
    run_lo, run_hi = _longest_clean_run(clean, WINDOW)
    lo = run_lo
    hi = lo + WINDOW
    print(f"Clean evaluation window: [{lo}, {hi}) "
          f"(longest clean run was [{run_lo}, {run_hi}), {run_hi - run_lo} frames)")

    # Training frames: cascade-verified, and strictly outside the window.
    train = np.flatnonzero(clean)
    train = train[(train < lo - 200) | (train >= hi + 200)]
    if len(train) > 3000:
        train = train[np.linspace(0, len(train) - 1, 3000).astype(int)]
    print(f"  {len(train)} training frames, all outside the evaluation window")

    # ---- inject the swap ----
    swap_lo = lo + (WINDOW - SWAP_LEN) // 2
    swap_hi = swap_lo + SWAP_LEN
    corrupted = markers.copy()
    blk = slice(swap_lo, swap_hi)
    corrupted[blk, idx_ring], corrupted[blk, idx_pinky] = (
        markers[blk, idx_pinky].copy(), markers[blk, idx_ring].copy())
    truth = np.zeros(WINDOW, dtype=bool)
    truth[swap_lo - lo: swap_hi - lo] = True
    print(f"  injected Ring1<->Pinky1 swap on frames [{swap_lo}, {swap_hi})")

    # ---- run strategyGMM on the corrupted data ----
    labeler = gl.GMMLabeler.fit(corrupted, labels, ctx.anchor_labels, train,
                                ctx.marker_idxs, n_components=3,
                                anchor_valid=ctx.anchor_valid)
    result = labeler.relabel(
        corrupted[lo:hi], labels, n_hypotheses=5,
        process_var=DEFAULT_PROCESS_VAR, obs_var=DEFAULT_OBS_VAR,
        anchor_valid=ctx.anchor_valid[lo:hi], init_frame=0)
    detected = result[:, 0] == 1

    tp = int((detected & truth).sum())
    fp = int((detected & ~truth).sum())
    fn = int((~detected & truth).sum())
    tn = int((~detected & ~truth).sum())
    acc = (tp + tn) / WINDOW
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\nAccuracy {acc:.1%}  precision {prec:.1%}  recall {rec:.1%}  "
          f"(TP {tp}, FP {fp}, FN {fn}, TN {tn})")

    det_edges = np.flatnonzero(np.diff(detected.astype(int)) != 0) + 1 + lo
    print(f"Detected transitions at frames {det_edges.tolist()}; "
          f"true transitions at [{swap_lo}, {swap_hi}]")

    # ---- before/after animation ----
    after = corrupted[lo:hi].copy()
    for k in np.flatnonzero(detected):
        after[k, idx_ring], after[k, idx_pinky] = (
            corrupted[lo + k, idx_pinky].copy(), corrupted[lo + k, idx_ring].copy())

    render_before_after(
        ctx, before=corrupted[lo:hi], after=after, lo=lo,
        is_swapped=detected, truth=truth,
        out_stem=OUT_DIR / "p7_injected_swap_before_after",
        title="strategyGMM — injected Ring1/Pinky1 swap, before vs after",
    )

    summary = OUT_DIR / "p7_injected_swap_metrics.txt"
    with summary.open("w") as f:
        f.write("strategyGMM ground-truth validation (injected swap into clean real data)\n")
        f.write(f"Trial: P7/Trial1_handsonly, markers Ring1 <-> Pinky1\n")
        f.write(f"Evaluation window: [{lo}, {hi}) ({WINDOW} frames), "
              f"cascade-clean (anchors + both targets CORRECT)\n")
        f.write(f"Training frames: {len(train)}, all >=200 frames outside the window\n")
        f.write(f"Injected swap: frames [{swap_lo}, {swap_hi}) ({SWAP_LEN} frames)\n")
        f.write(f"Accuracy {acc:.1%}  precision {prec:.1%}  recall {rec:.1%}\n")
        f.write(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}\n")
        f.write(f"Detected transitions: {det_edges.tolist()}\n")
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
