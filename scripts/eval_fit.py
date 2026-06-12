#!/usr/bin/env python3
"""Visualise MANOFitter results: residuals and 3-D marker/skeleton overlays.

Unlike ``eval_labeler.py`` (which evaluates the CNN labeler in isolation), this
script evaluates the *fit pipeline* end-to-end — the residual between each
fitted MANO joint and the Vicon marker assigned to it.

Figures produced
----------------
fit_fig1_residuals.png   Per-joint residual bars + overall residual distribution
fit_fig2_residual_time.png   Mean residual across the recording (temporal)
fit_fig3_overlays.png    3-D frames: Vicon markers (gray) + fitted skeleton

Usage
-----
python scripts/eval_fit.py \\
    --npz   data/Pxh8/Hands_only_Left/mano_fit_left.npz \\
    --csv   data/Pxh8/Hands_only_Left/trajectories_synced.csv \\
    --out-dir vicon2mano/eval/fit \\
    [--n-frames 6]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicon2mano.correspondence import MANO_JOINT_NAMES
from vicon2mano.loader import load_csv

# ─── Visual constants (shared palette with eval_labeler) ──────────────────────

JOINT_LABELS = [n.replace("_", "\n") for n in [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]]

FINGER_COLORS = [
    "#444444",
    "#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c",
    "#27ae60", "#27ae60", "#27ae60", "#27ae60",
    "#2980b9", "#2980b9", "#2980b9", "#2980b9",
    "#e67e22", "#e67e22", "#e67e22", "#e67e22",
    "#8e44ad", "#8e44ad", "#8e44ad", "#8e44ad",
]

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_LEGEND = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_LEGEND_COLORS = ["#444444", "#e74c3c", "#27ae60", "#2980b9", "#e67e22", "#8e44ad"]


# ─── Residual computation ─────────────────────────────────────────────────────

def per_joint_residuals(
    joints: np.ndarray,      # (T, 21, 3) fitted joints, metres
    markers_m: np.ndarray,   # (T, N, 3) markers, metres (may contain NaN)
    assignment: np.ndarray,  # (21,) marker index per joint, -1 if unassigned
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, 21) per-frame residuals in mm (NaN where unassigned/missing)
    and the (21,) marker index per joint."""
    T = joints.shape[0]
    res = np.full((T, 21), np.nan, dtype=np.float32)
    for j, mi in enumerate(assignment):
        if mi < 0:
            continue
        d = np.linalg.norm(joints[:, j] - markers_m[:, mi], axis=-1) * 1000.0
        res[:, j] = d
    return res, assignment


# ─── Figure 1: residual bars + distribution ───────────────────────────────────

def plot_residuals(res: np.ndarray, assignment: np.ndarray, labels, out_dir):
    with np.errstate(invalid="ignore"):
        mean_j = np.nanmean(res, axis=0)   # (21,)
    observed = np.isfinite(mean_j)
    x = np.arange(21)
    clrs = np.array(FINGER_COLORS)
    overall = np.nanmean(res)

    fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"MANO Fit Residuals — overall mean {overall:.1f} mm "
        f"({observed.sum()}/21 joints matched to markers)",
        fontsize=13, fontweight="bold",
    )

    ax_bar.bar(x[observed], mean_j[observed], color=clrs[observed],
               edgecolor="white", linewidth=0.4)
    ax_bar.set_xticks(x[observed])
    ax_bar.set_xticklabels([labels[i] for i in x[observed]], fontsize=7)
    ax_bar.set_ylabel("Mean residual (mm)")
    ax_bar.set_title("Per-joint marker→fitted-joint distance")
    ax_bar.axhline(overall, color="gray", linestyle="--", linewidth=1,
                   label=f"overall {overall:.1f} mm")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    all_res = res[np.isfinite(res)]
    ax_hist.hist(all_res, bins=80, color="#2980b9", edgecolor="white",
                 linewidth=0.3, density=True)
    ax_hist.axvline(np.median(all_res), color="#27ae60", linestyle="--",
                    linewidth=1.5, label=f"median {np.median(all_res):.1f} mm")
    ax_hist.axvline(overall, color="#e74c3c", linestyle="--",
                    linewidth=1.5, label=f"mean {overall:.1f} mm")
    ax_hist.set_xlabel("Residual (mm)"); ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual distribution (all matched joints × all frames)")
    ax_hist.legend(fontsize=8); ax_hist.grid(alpha=0.3)

    handles = [mpatches.Patch(color=FINGER_LEGEND_COLORS[i], label=FINGER_LEGEND[i])
               for i in range(len(FINGER_LEGEND))]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out_dir, "fit_fig1_residuals.png")


# ─── Figure 2: residual across the recording ──────────────────────────────────

def plot_residual_time(res: np.ndarray, out_dir):
    with np.errstate(invalid="ignore"):
        per_frame = np.nanmean(res, axis=1)  # (T,)
    T = len(per_frame)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.suptitle("Fit Residual Across the Recording", fontsize=13, fontweight="bold")
    ax.plot(np.arange(T), per_frame, color="#2980b9", linewidth=0.7)
    m = np.nanmean(per_frame)
    ax.axhline(m, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"mean {m:.1f} mm")
    ax.fill_between(np.arange(T), per_frame, m, where=per_frame > m,
                    color="#e74c3c", alpha=0.08)
    ax.set_xlabel("Frame"); ax.set_ylabel("Mean residual (mm)")
    ax.set_xlim(0, T)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "fit_fig2_residual_time.png")


# ─── Figure 3: 3-D overlays ───────────────────────────────────────────────────

def plot_overlays(joints, markers_m, assignment, n_frames, out_dir):
    T = joints.shape[0]
    idxs = np.linspace(0, T - 1, n_frames).astype(int)
    ncols = min(3, n_frames)
    nrows = (n_frames + ncols - 1) // ncols

    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))
    fig.suptitle(
        "3-D Overlay: Vicon markers (gray) · Fitted MANO skeleton (coloured) · "
        "assignment link (dashed)",
        fontsize=12, fontweight="bold",
    )
    for i, t in enumerate(idxs):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _draw_frame(ax, joints[t], markers_m[t], assignment, title=f"Frame {t}")

    handles = [
        mpatches.Patch(color="lightgray", label="Vicon marker (matched)"),
        mpatches.Patch(color="salmon", label="Vicon marker (unmatched)"),
    ] + [mpatches.Patch(color=FINGER_LEGEND_COLORS[i], label=FINGER_LEGEND[i])
         for i in range(len(FINGER_LEGEND))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out_dir, "fit_fig3_overlays.png")


def _draw_frame(ax, joints_t, markers_t, assignment, title=""):
    jp = joints_t * 1000.0          # (21, 3) mm
    mk = markers_t * 1000.0         # (N, 3) mm

    matched = {int(mi) for mi in assignment if mi >= 0}
    finite = np.isfinite(mk).all(axis=1)
    matched_pts = [i for i in range(len(mk)) if i in matched and finite[i]]
    other_pts = [i for i in range(len(mk)) if i not in matched and finite[i]]

    if matched_pts:
        ax.scatter(*mk[matched_pts].T, c="lightgray", s=28, depthshade=False, zorder=2)
    if other_pts:
        ax.scatter(*mk[other_pts].T, c="salmon", s=26, marker="x",
                   depthshade=False, zorder=3)

    for j0, j1 in SKELETON:
        ax.plot(*zip(jp[j0], jp[j1]), color=FINGER_COLORS[j1], linewidth=2, zorder=4)
    for j in range(21):
        ax.scatter(*jp[j], c=FINGER_COLORS[j], s=45, depthshade=False,
                   edgecolors="white", linewidths=0.5, zorder=5)

    # assignment link: fitted joint → its marker
    for j, mi in enumerate(assignment):
        if mi >= 0 and finite[mi]:
            ax.plot(*zip(jp[j], mk[mi]), color="gray", linewidth=0.7,
                    linestyle=":", alpha=0.6, zorder=6)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X (mm)", fontsize=7); ax.set_ylabel("Y (mm)", fontsize=7)
    ax.set_zlabel("Z (mm)", fontsize=7); ax.tick_params(labelsize=6)

    # Frame on the fitted hand (skeleton + assigned markers) so it fills the
    # axes — unmatched markers from the other hand / forearm are off-frame.
    focus = [jp]
    matched_finite = [mi for mi in assignment if mi >= 0 and finite[mi]]
    if matched_finite:
        focus.append(mk[matched_finite])
    pts = np.vstack(focus)
    ctr = pts.mean(axis=0)
    span = max(np.ptp(pts, axis=0)) / 2 * 1.3 + 1e-3
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visualise MANOFitter fit results.")
    ap.add_argument("--npz", required=True, help="FitResult .npz from vicon2mano")
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--out-dir", default=None,
                    help="save figures here (created if needed); else display")
    ap.add_argument("--n-frames", type=int, default=6,
                    help="frames to draw in the 3-D overlay (evenly spaced)")
    args = ap.parse_args()

    if args.out_dir is None and "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    out_dir = Path(args.out_dir) if args.out_dir else None

    print(f"[eval_fit] loading {args.npz}")
    data = np.load(args.npz)
    joints = data["joints"]            # (T, 21, 3) metres
    assignment = data["assignment"]    # (21,)

    print(f"[eval_fit] loading markers from {args.csv}")
    markers, _labels = load_csv(args.csv)
    markers_m = markers.astype(np.float32) * 1e-3
    T = min(joints.shape[0], markers_m.shape[0])
    joints, markers_m = joints[:T], markers_m[:T]
    print(f"[eval_fit] {T} frames, {markers_m.shape[1]} markers, "
          f"{(assignment >= 0).sum()}/21 joints matched")

    res, _ = per_joint_residuals(joints, markers_m, assignment)

    print("[eval_fit] Fig 1 — residual bars + distribution …")
    plot_residuals(res, assignment, JOINT_LABELS, out_dir)
    print("[eval_fit] Fig 2 — residual across recording …")
    plot_residual_time(res, out_dir)
    print("[eval_fit] Fig 3 — 3-D overlays …")
    plot_overlays(joints, markers_m, assignment, args.n_frames, out_dir)
    print("[eval_fit] done.")


if __name__ == "__main__":
    main()
