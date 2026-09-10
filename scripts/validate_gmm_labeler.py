#!/usr/bin/env python3
"""Real-data validation of strategyGMM against a known, documented swap.

P7/Trial1_handsonly has a confirmed Ring1<->Pinky1 label swap in the
vicinity of frame 42789 (see scripts/relabel_with_mano.py's comment above
its verification code). Loading/fitting is shared with
animate_gmm_relabel.py via scripts/_gmm_validation_common.py.

Findings so far (2026-09-10)
-----------------------------
1. The window right at frame 42789 sits inside a much longer cascade-
   flagged run: [41195, 42829] (1635 frames), not a single-frame event.
2. At the start of that run (frame 41195), the raw marker data shows a
   *rolling mislabelling cascade* across Palm1/Palm2/Thumb1/Ring1/Pinky1
   simultaneously -- not a clean pairwise Ring1<->Pinky1 swap. The 3
   anchor markers used for the rigid local frame (Palm2/Palm3/Thumb1) are
   themselves corrupted during this exact episode, which silently wrecks
   every other marker's local coordinate.
3. Fix: gate on a per-frame ``anchor_valid`` mask (cascade CORRECT on all
   3 anchors) via ``vicon2mano.gmm_labeler.mask_untrustworthy_frames`` /
   the ``anchor_valid=`` parameter now threaded through ``GMMLabeler`` and
   ``relabel_sequence``. This script uses that gated path.

Usage
-----
python scripts/validate_gmm_labeler.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gmm_validation_common import (   # noqa: E402
    OUT_DIR, DOCUMENTED_SWAP_FRAME, load_p7_ring1_pinky1, fit_and_relabel_window, gl,
)


def main() -> None:
    ctx = load_p7_ring1_pinky1()
    idx_ring, idx_pinky = ctx.marker_idxs

    lo, hi = max(0, ctx.run_lo - 150), min(ctx.markers.shape[0], ctx.run_lo + 150)
    is_swapped, init_frame_global = fit_and_relabel_window(ctx, lo, hi)
    t_axis = np.arange(lo, hi)
    print(f"  eval window [{lo}, {hi}); init_frame (global): {init_frame_global}")

    flips = t_axis[np.flatnonzero(np.diff(is_swapped.astype(int)) != 0) + 1]
    print(f"GMM verdict transitions within [{lo}, {hi}): frames {flips.tolist()}")
    print(f"Cascade's own transition (start of the flagged run): frame {ctx.run_lo}")

    bad_ring_win = ctx.status[lo:hi, idx_ring] != 2
    agree = (is_swapped == bad_ring_win)
    print(f"Frame-by-frame agreement with cascade's Ring1 verdict: {agree.mean():.1%} "
          f"({agree.sum()}/{len(agree)})")

    name_to_idx = {l: i for i, l in enumerate(ctx.labels)}
    R, o = gl.rigid_frames(ctx.markers, *(name_to_idx[a] for a in ctx.anchor_labels))
    local = gl.to_local(ctx.markers, R, o)
    local = gl.mask_untrustworthy_frames(local, ctx.anchor_valid)
    ring_local, pinky_local = local[lo:hi, idx_ring], local[lo:hi, idx_pinky]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, dim, name in zip(axes[:2], (0, 1), ("local X (mm, anchor-gated)", "local Y (mm)")):
        ax.plot(t_axis, ring_local[:, dim], label="Ring1 (observed)", color="#e67e22")
        ax.plot(t_axis, pinky_local[:, dim], label="Pinky1 (observed)", color="#8e44ad")
        ax.axvline(ctx.run_lo, color="black", linestyle="--", linewidth=1,
                   label="cascade's swap-run start" if dim == 0 else None)
        swapped_frames = t_axis[is_swapped]
        if swapped_frames.size:
            ax.axvspan(swapped_frames.min(), swapped_frames.max(), color="red", alpha=0.12,
                      label="GMM verdict: swapped" if dim == 0 else None)
        ax.set_ylabel(name)
    axes[0].legend(loc="upper right", fontsize=8)

    ax3 = axes[2]
    ax3.step(t_axis, bad_ring_win.astype(int), where="post",
             label="cascade: Ring1 flagged INCORRECT", color="black")
    ax3.step(t_axis, is_swapped.astype(int) - 0.05, where="post",
             label="strategyGMM: verdict SWAPPED", color="crimson", linestyle="--")
    ax3.step(t_axis, (~ctx.anchor_valid[lo:hi]).astype(int) - 0.10, where="post",
             label="anchors untrustworthy this frame", color="gray", linestyle=":")
    ax3.set_ylim(-0.25, 1.2)
    ax3.set_ylabel("flag")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlabel("frame")

    fig.suptitle("strategyGMM vs. cascade (anchor-gated): P7/Trial1_handsonly, Ring1 <-> Pinky1\n"
                "(local-frame position; red band = GMM+Viterbi verdict; "
                "bottom panel compares both verdicts + anchor trust)")
    fig.tight_layout()
    out_png = OUT_DIR / "p7_ring1_pinky1_swap_validation.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nSaved {out_png}")

    summary_path = OUT_DIR / "p7_ring1_pinky1_swap_validation.txt"
    with summary_path.open("w") as f:
        f.write("strategyGMM validation (anchor-gated): P7/Trial1_handsonly, "
              "Ring1 <-> Pinky1 swap\n")
        f.write(f"Documented ground truth frame (relabel_with_mano.py comment): "
              f"{DOCUMENTED_SWAP_FRAME}\n")
        f.write(f"Cascade's own 'Ring1 INCORRECT' run containing it: "
              f"[{ctx.run_lo}, {ctx.run_hi}] ({ctx.run_hi - ctx.run_lo + 1} frames)\n")
        f.write(f"GMM training (reference) frames: {len(ctx.ref_frames)}, "
              f"cascade-verified (targets AND anchors)\n")
        f.write(f"Evaluation window: [{lo}, {hi})\n")
        f.write(f"GMM verdict transition frames: {flips.tolist()}\n")
        f.write(f"Frame-by-frame agreement with cascade's Ring1 verdict: "
              f"{agree.mean():.1%} ({agree.sum()}/{len(agree)})\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
