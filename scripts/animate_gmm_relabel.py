#!/usr/bin/env python3
"""Animate the before/after effect of strategyGMM's relabelling.

Side-by-side 3-D scatter animation of P7/Trial1_handsonly's marker cloud
across the Ring1<->Pinky1 investigation window (see
validate_gmm_labeler.py's module docstring for what that window actually
contains -- a rolling mislabelling cascade, anchor-gated here via
scripts/_gmm_validation_common.py). Left panel is the recording as
labelled on disk ("before"), right panel is after applying strategyGMM's
per-frame identity mapping ("after"). Ring1 is always drawn orange and
Pinky1 always drawn purple, and frames the cascade itself flags as having
untrustworthy anchors are marked -- a frame where "before" and "after"
look the same despite that mark means the GMM (correctly) declined to
guess from garbage-in local coordinates, not that nothing happened.

Usage
-----
python scripts/animate_gmm_relabel.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from _gmm_validation_common import (   # noqa: E402
    OUT_DIR, load_p7_ring1_pinky1, fit_and_relabel_window,
)


def main() -> None:
    ctx = load_p7_ring1_pinky1()
    idx_ring, idx_pinky = ctx.marker_idxs
    markers = ctx.markers

    lo, hi = max(0, ctx.run_lo - 150), min(markers.shape[0], ctx.run_lo + 150)
    is_swapped, init_frame_global = fit_and_relabel_window(ctx, lo, hi)
    print(f"GMM verdict: {is_swapped.sum()}/{len(is_swapped)} frames in [{lo},{hi}) "
          f"marked swapped (init_frame global={init_frame_global})")

    eval_markers = markers[lo:hi]
    after = eval_markers.copy()
    for k in np.flatnonzero(is_swapped):
        after[k, idx_ring], after[k, idx_pinky] = (
            eval_markers[k, idx_pinky].copy(), eval_markers[k, idx_ring].copy())

    anchor_bad_win = ~ctx.anchor_valid[lo:hi]

    stride = max(1, len(eval_markers) // 150)   # cap ~150 animation frames
    frame_ids = np.arange(0, len(eval_markers), stride)

    all_pts = np.concatenate([eval_markers.reshape(-1, 3), after.reshape(-1, 3)])
    finite = np.isfinite(all_pts).all(axis=1)
    mins, maxs = all_pts[finite].min(0), all_pts[finite].max(0)
    pad = 0.05 * (maxs - mins).max()

    fig = plt.figure(figsize=(11, 6.5))
    ax_before = fig.add_subplot(121, projection="3d")
    ax_after = fig.add_subplot(122, projection="3d")

    def style(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    def scatter_frame(ax, pts):
        ax.cla()
        others = np.delete(pts, ctx.marker_idxs, axis=0)
        finite_o = np.isfinite(others).all(axis=1)
        ax.scatter(*others[finite_o].T, color="#bbbbbb", s=12, depthshade=False)
        if np.isfinite(pts[idx_ring]).all():
            ax.scatter(*pts[idx_ring], color="#e67e22", s=70, depthshade=False, label="Ring1")
        if np.isfinite(pts[idx_pinky]).all():
            ax.scatter(*pts[idx_pinky], color="#8e44ad", s=70, depthshade=False, label="Pinky1")

    def update(i):
        k = int(frame_ids[i])
        global_frame = lo + k
        tags = []
        if is_swapped[k]:
            tags.append("SWAPPED")
        if anchor_bad_win[k]:
            tags.append("anchors untrustworthy")
        flag = f"  [{', '.join(tags)}]" if tags else ""
        scatter_frame(ax_before, eval_markers[k])
        style(ax_before, f"BEFORE (as labelled) — frame {global_frame}{flag}")
        scatter_frame(ax_after, after[k])
        style(ax_after, f"AFTER (strategyGMM) — frame {global_frame}"
                        + ("  [corrected]" if is_swapped[k] else ""))
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=80)
    out_gif = OUT_DIR / "p7_ring1_pinky1_before_after.gif"
    print(f"Rendering {len(frame_ids)} frames to {out_gif} ...")
    anim.save(out_gif, writer=PillowWriter(fps=10))
    print(f"Saved {out_gif}")


if __name__ == "__main__":
    main()
