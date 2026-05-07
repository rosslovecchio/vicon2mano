#!/usr/bin/env python3
"""
Post-training evaluation: training curves, per-joint accuracy, and 3-D Vicon/MANO overlays.

Figures produced
----------------
Fig 1  Training curves (train/val loss + LR schedule) — requires _log.json from train_labeler.py
Fig 2  Per-joint labeler performance (match rate, error, confidence)
Fig 3  3-D frames: Vicon markers + predicted MANO skeleton overlay
Fig 4  3-D frames: MANO mesh + Vicon markers (requires --mano-dir)

Usage
-----
python scripts/eval_labeler.py \\
    --weights vicon2mano/weights/deep_labeler.pt \\
    --real-data /path/to/Hands_only1_labeled.csv \\
    [--mano-dir /path/to/mano_v1_2/models] \\
    [--n-frames 6] \\
    [--out-dir eval_results]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicon2mano.deep_labeler import (
    MarkerLabeler,
    K_JOINTS,
    SPATIAL_HALF,
    VOXEL_RES,
    assign_from_heatmaps,
    voxelise_with_centroid,
)
from vicon2mano.real_data import MARKER_TO_MANO, RealDataConfig, RealLabeledDataset

# ─── Visual constants ─────────────────────────────────────────────────────────

JOINT_NAMES = [
    "Wrist",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
    "Index_MCP",  "Index_PIP",  "Index_DIP",  "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP",   "Ring_PIP",   "Ring_DIP",   "Ring_TIP",
    "Pinky_MCP",  "Pinky_PIP",  "Pinky_DIP",  "Pinky_TIP",
]

FINGER_COLORS = [
    "#444444",                                          # 0  wrist
    "#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c",        # 1-4  thumb
    "#27ae60", "#27ae60", "#27ae60", "#27ae60",        # 5-8  index
    "#2980b9", "#2980b9", "#2980b9", "#2980b9",        # 9-12 middle
    "#e67e22", "#e67e22", "#e67e22", "#e67e22",        # 13-16 ring
    "#8e44ad", "#8e44ad", "#8e44ad", "#8e44ad",        # 17-20 pinky
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

CORRECT_THRESH_MM = 10.0  # mm — assignment considered correct if within this distance of GT


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(path: str, device: torch.device) -> tuple[MarkerLabeler, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        state, meta = ckpt, {}
    model = MarkerLabeler().to(device)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def load_log(weights_path: str) -> list[dict] | None:
    log_path = Path(weights_path).with_name(Path(weights_path).stem + "_log.json")
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return None


# ─── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def _logits_to_world(logits: torch.Tensor, centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax of each joint heatmap → world position + confidence (no assignment step)."""
    probs = torch.sigmoid(logits)                      # (21, R, R, R)
    flat = probs.reshape(K_JOINTS, -1)
    argmax_flat = flat.argmax(dim=1)                   # (21,)
    conf = flat[torch.arange(K_JOINTS), argmax_flat].numpy()

    iz = (argmax_flat // (VOXEL_RES * VOXEL_RES)).numpy().astype(float)
    rem = argmax_flat % (VOXEL_RES * VOXEL_RES)
    iy = (rem // VOXEL_RES).numpy().astype(float)
    ix = (rem % VOXEL_RES).numpy().astype(float)

    # inverse of: vf = (pt / half + 1.0) * 0.5 * res  →  pt = (vf/res * 2 - 1) * half
    pred_world = centroid + (np.stack([ix, iy, iz], axis=1) / VOXEL_RES * 2.0 - 1.0) * SPATIAL_HALF
    return pred_world, conf  # (21,3), (21,)


@torch.no_grad()
def evaluate_frames(
    model: MarkerLabeler,
    frames: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    max_frames: int | None = None,
) -> list[dict]:
    """Run the labeler on every val frame (or up to max_frames).

    Each result dict contains:
        markers_m     (N, 3) metres
        joint_pos_gt  (21, 3) metres, NaN where unobserved
        assignment    (21,) marker index or -1
        pred_world    (21, 3) heatmap-argmax world positions (before assignment)
        conf          (21,) sigmoid confidence scores
        gt_marker_idx (21,) index into markers_m for GT, -1 if unobserved
    """
    subset = frames if max_frames is None else frames[:max_frames]
    results = []

    for markers_m, joint_pos_gt in subset:
        grid, centroid = voxelise_with_centroid(markers_m)
        logits = model(grid.unsqueeze(0).to(device))[0].cpu()   # (21, R, R, R)

        assignment = assign_from_heatmaps(logits, markers_m, centroid)
        pred_world, conf = _logits_to_world(logits, centroid)

        # Find which marker index corresponds to each GT joint position
        gt_marker_idx = np.full(K_JOINTS, -1, dtype=int)
        for j in range(K_JOINTS):
            if not np.isnan(joint_pos_gt[j]).any():
                dists = np.linalg.norm(markers_m - joint_pos_gt[j], axis=1)
                gt_marker_idx[j] = int(np.argmin(dists))

        results.append(dict(
            markers_m=markers_m,
            joint_pos_gt=joint_pos_gt,
            assignment=assignment,
            pred_world=pred_world,
            conf=conf,
            gt_marker_idx=gt_marker_idx,
        ))

    return results


# ─── Figure 1: Training curves ────────────────────────────────────────────────

def plot_training_curves(log: list[dict], meta: dict, out_dir: Path | None):
    epochs      = [e["epoch"] for e in log]
    train_loss  = [e["train_loss"] for e in log]
    val_loss    = [e["val_loss"] for e in log]
    lr          = [e["lr"] for e in log]
    best_ep     = epochs[int(np.argmin(val_loss))]
    best_val    = min(val_loss)

    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold")

    ax_loss.plot(epochs, train_loss, label="Train", color="#2980b9", linewidth=2)
    ax_loss.plot(epochs, val_loss,   label="Val",   color="#e74c3c", linewidth=2)
    ax_loss.axvline(best_ep, color="#27ae60", linestyle="--", linewidth=1.2,
                    label=f"Best val epoch {best_ep}")
    ax_loss.annotate(
        f"{best_val:.5f}",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(1, len(epochs) * 0.04), best_val * 1.08),
        fontsize=8, color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1),
    )
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Focal BCE loss")
    ax_loss.set_title("Loss curves"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_lr.plot(epochs, lr, color="#8e44ad", linewidth=2)
    ax_lr.set_xlabel("Epoch"); ax_lr.set_ylabel("Learning rate")
    ax_lr.set_title("Cosine LR schedule"); ax_lr.set_yscale("log"); ax_lr.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_training_curves.png")


# ─── Figure 2: Per-joint performance ─────────────────────────────────────────

def plot_joint_performance(results: list[dict], out_dir: Path | None):
    n = len(results)
    match_cnt   = np.zeros(K_JOINTS)
    correct_cnt = np.zeros(K_JOINTS)
    gt_cnt      = np.zeros(K_JOINTS, dtype=int)
    dist_sum    = np.zeros(K_JOINTS)
    all_conf: list[float] = []

    for r in results:
        asgn        = r["assignment"]
        markers_m   = r["markers_m"]
        gt_pos      = r["joint_pos_gt"]
        gt_idx      = r["gt_marker_idx"]

        for j in range(K_JOINTS):
            if asgn[j] >= 0:
                match_cnt[j] += 1

            if gt_idx[j] >= 0:          # joint is observable in this frame
                gt_cnt[j] += 1
                if asgn[j] >= 0:
                    d_mm = np.linalg.norm(markers_m[asgn[j]] - gt_pos[j]) * 1000
                    dist_sum[j] += d_mm
                    if d_mm < CORRECT_THRESH_MM:
                        correct_cnt[j] += 1

        all_conf.extend(r["conf"].tolist())

    match_rate   = match_cnt / n
    correct_rate = np.where(gt_cnt > 0, correct_cnt / gt_cnt, np.nan)
    mean_dist    = np.where(gt_cnt > 0, dist_sum / gt_cnt, np.nan)
    has_gt       = gt_cnt > 0

    short = np.array([nm.replace("_", "\n") for nm in JOINT_NAMES])
    x     = np.arange(K_JOINTS)
    clrs  = np.array(FINGER_COLORS)

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(f"Labeler Performance on Validation Set  (n={n} frames)",
                 fontsize=13, fontweight="bold")

    # 1 – match rate
    ax = axes[0, 0]
    ax.bar(x, match_rate * 100, color=clrs, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("Match rate (%)"); ax.set_ylim(0, 108)
    ax.set_title("Fraction of frames where joint received an assignment")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.7)
    ax.grid(axis="y", alpha=0.3)

    # 2 – correct assignment rate (observable joints only)
    ax = axes[0, 1]
    ax.bar(x[has_gt], correct_rate[has_gt] * 100, color=clrs[has_gt],
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x[has_gt]); ax.set_xticklabels(short[has_gt], fontsize=7)
    ax.set_ylabel("Correct (%)"); ax.set_ylim(0, 108)
    ax.set_title(f"Assignments within {CORRECT_THRESH_MM:.0f} mm of GT (observable joints)")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.7)
    ax.grid(axis="y", alpha=0.3)

    # 3 – mean distance error
    ax = axes[1, 0]
    ax.bar(x[has_gt], mean_dist[has_gt], color=clrs[has_gt],
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x[has_gt]); ax.set_xticklabels(short[has_gt], fontsize=7)
    ax.set_ylabel("Mean error (mm)")
    ax.set_title("Mean distance between assigned marker and GT joint position")
    ax.axhline(CORRECT_THRESH_MM, color="gray", linestyle="--", linewidth=0.8,
               label=f"{CORRECT_THRESH_MM:.0f} mm")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 4 – confidence distribution
    ax = axes[1, 1]
    ax.hist(all_conf, bins=60, color="#2980b9", edgecolor="white",
            linewidth=0.3, density=True)
    ax.axvline(0.10, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Conf threshold 0.10")
    ax.set_xlabel("Sigmoid confidence"); ax.set_ylabel("Density")
    ax.set_title("Heatmap confidence (all 21 joints × all frames)")
    ax.legend(); ax.grid(alpha=0.3)

    handles = [mpatches.Patch(color=FINGER_LEGEND_COLORS[i], label=FINGER_LEGEND[i])
               for i in range(len(FINGER_LEGEND))]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_dir, "fig2_joint_performance.png")


# ─── Figure 3: 3-D Vicon + skeleton overlays ─────────────────────────────────

def plot_3d_overlays(results: list[dict], n_frames: int, out_dir: Path | None):
    sample = results[:n_frames]
    ncols  = min(3, len(sample))
    nrows  = (len(sample) + ncols - 1) // ncols

    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))
    fig.suptitle(
        "3-D Overlay: Vicon markers (gray) · Predicted skeleton (coloured) · GT joints (◆)",
        fontsize=12, fontweight="bold",
    )

    for i, r in enumerate(sample):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _draw_frame(ax, r, title=f"Val frame {i + 1}")

    # Shared legend
    legend_handles = [
        mpatches.Patch(color="lightgray",  label="Vicon marker (matched)"),
        mpatches.Patch(color="salmon",     label="Vicon marker (unmatched)"),
        mpatches.Patch(color="limegreen",  label="GT joint position (◆)"),
        mpatches.Patch(color="#27ae60",    label="Pred–GT link < 10 mm"),
        mpatches.Patch(color="#e74c3c",    label="Pred–GT link ≥ 10 mm"),
    ] + [mpatches.Patch(color=FINGER_LEGEND_COLORS[i], label=FINGER_LEGEND[i])
         for i in range(len(FINGER_LEGEND))]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out_dir, "fig3_3d_overlays.png")


def _draw_frame(ax, r: dict, title: str = ""):
    mk   = r["markers_m"] * 1000          # → mm for display
    gt   = r["joint_pos_gt"] * 1000
    asgn = r["assignment"]

    # Predicted joint positions = the assigned marker positions
    jp = np.full((K_JOINTS, 3), np.nan)
    for j in range(K_JOINTS):
        if asgn[j] >= 0:
            jp[j] = mk[asgn[j]]

    matched_set = set(int(asgn[j]) for j in range(K_JOINTS) if asgn[j] >= 0)
    unmatched   = [i for i in range(len(mk)) if i not in matched_set]

    # All markers in light gray
    ax.scatter(*mk.T, c="lightgray", s=25, depthshade=False, zorder=2)
    # Unmatched in salmon X
    if unmatched:
        ax.scatter(*mk[unmatched].T, c="salmon", s=30, marker="x",
                   depthshade=False, zorder=3)

    # Skeleton lines
    for j0, j1 in SKELETON:
        if not (np.isnan(jp[j0]).any() or np.isnan(jp[j1]).any()):
            ax.plot(*zip(jp[j0], jp[j1]), color=FINGER_COLORS[j1], linewidth=2, zorder=4)

    # Predicted joint dots
    for j in range(K_JOINTS):
        if not np.isnan(jp[j]).any():
            ax.scatter(*jp[j], c=FINGER_COLORS[j], s=55, depthshade=False,
                       edgecolors="white", linewidths=0.5, zorder=5)

    # GT joints as green diamonds
    for j in range(K_JOINTS):
        if not np.isnan(gt[j]).any():
            ax.scatter(*gt[j], c="limegreen", s=35, marker="D",
                       depthshade=False, alpha=0.75, zorder=6)

    # Pred → GT link (green = close, red = far)
    for j in range(K_JOINTS):
        if not (np.isnan(gt[j]).any() or np.isnan(jp[j]).any()):
            d = np.linalg.norm(gt[j] - jp[j])
            c = "#27ae60" if d < CORRECT_THRESH_MM else "#e74c3c"
            ax.plot(*zip(gt[j], jp[j]), color=c, linewidth=0.9, linestyle="--",
                    alpha=0.75, zorder=7)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X (mm)", fontsize=7); ax.set_ylabel("Y (mm)", fontsize=7)
    ax.set_zlabel("Z (mm)", fontsize=7); ax.tick_params(labelsize=6)

    # Equal aspect
    valid = np.vstack([mk, gt[~np.isnan(gt).any(axis=1)]])
    if len(valid):
        ctr  = valid.mean(axis=0)
        span = max(np.ptp(valid, axis=0)) / 2 * 1.3 + 1e-3
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)


# ─── Figure 4: MANO mesh overlay (optional) ──────────────────────────────────

def plot_mano_overlay(results: list[dict], mano_dir: str, n_frames: int, out_dir: Path | None):
    try:
        from vicon2mano.fitter import FitConfig, MANOFitter
    except ImportError:
        print("[eval] smplx not available — skipping Fig 4 (MANO mesh overlay)")
        return

    cfg = FitConfig(mano_model_path=mano_dir, n_iters_shape=30, n_iters_pose=80)
    try:
        fitter = MANOFitter(cfg)
    except Exception as e:
        print(f"[eval] Could not load MANO model: {e}")
        return

    # Try to get triangle faces from the smplx model
    faces = None
    if hasattr(fitter, "_model") and hasattr(fitter._model, "faces"):
        faces = np.array(fitter._model.faces, dtype=int)

    sample = results[:n_frames]
    ncols  = min(3, len(sample))
    nrows  = (len(sample) + ncols - 1) // ncols

    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))
    fig.suptitle("MANO Mesh + Vicon Markers Overlay", fontsize=12, fontweight="bold")

    for i, r in enumerate(sample):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        # Fit MANO to this single frame; fitter expects mm
        markers_mm = r["markers_m"] * 1000   # (N, 3) mm
        try:
            result = fitter.fit(markers_mm[None], verbose=False)   # (1, N, 3)
            verts  = result.vertices[0] * 1000   # (778, 3) mm
            joints = result.joints[0] * 1000     # (21, 3) mm
        except Exception as e:
            print(f"[eval] MANO fit failed for frame {i}: {e}")
            joints, verts = None, None

        # MANO mesh (semi-transparent)
        if verts is not None and faces is not None:
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces, color="#3498db", alpha=0.12, linewidth=0,
            )

        # Vicon markers
        mk = markers_mm
        ax.scatter(*mk.T, c="dimgray", s=30, depthshade=False, zorder=4, label="Vicon")

        # MANO skeleton
        if joints is not None:
            for j0, j1 in SKELETON:
                ax.plot(*zip(joints[j0], joints[j1]),
                        color=FINGER_COLORS[j1], linewidth=2, zorder=5)
            for j in range(K_JOINTS):
                ax.scatter(*joints[j], c=FINGER_COLORS[j], s=50, depthshade=False,
                           edgecolors="white", linewidths=0.5, zorder=6)

            # Connection: Vicon marker → matched MANO joint
            asgn = result.assignment   # (21,)
            for j in range(K_JOINTS):
                if asgn[j] >= 0:
                    ax.plot(*zip(mk[asgn[j]], joints[j]),
                            color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

        ax.set_title(f"Val frame {i + 1}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=7); ax.set_ylabel("Y (mm)", fontsize=7)
        ax.set_zlabel("Z (mm)", fontsize=7); ax.tick_params(labelsize=6)

        valid = np.vstack([mk, joints]) if joints is not None else mk
        ctr   = valid.mean(axis=0)
        span  = max(np.ptp(valid, axis=0)) / 2 * 1.3 + 1e-3
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)

    plt.tight_layout()
    _save(fig, out_dir, "fig4_mano_overlay.png")


# ─── Utility ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path | None, filename: str):
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
    ap = argparse.ArgumentParser(description="Evaluate a trained MarkerLabeler.")
    ap.add_argument("--weights",     default="vicon2mano/weights/deep_labeler.pt")
    ap.add_argument("--real-data",   required=True,
                    help="Labeled Vicon CSV (columns: frame, marker, x_mm, y_mm, z_mm)")
    ap.add_argument("--mano-dir",    default=None,
                    help="MANO model directory — enables Fig 4 (mesh overlay)")
    ap.add_argument("--n-frames",    type=int, default=6,
                    help="Number of frames to visualise in 3-D plots")
    ap.add_argument("--val-frac",    type=float, default=0.02,
                    help="Val fraction — should match what was used during training")
    ap.add_argument("--eval-frames", type=int, default=500,
                    help="Max val frames used for statistical plots (Figs 1-2)")
    ap.add_argument("--out-dir",     default=None,
                    help="Save all figures to this directory (created if needed). "
                         "If omitted, figures are displayed interactively.")
    ap.add_argument("--device",      default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.out_dir is None and "DISPLAY" not in os.environ:
        print("[eval] no DISPLAY detected — add --out-dir to save figures to disk")
        matplotlib.use("Agg")

    out_dir = Path(args.out_dir) if args.out_dir else None
    print(f"[eval] device = {device}")

    # Load model
    print("[eval] loading model …")
    model, meta = load_checkpoint(args.weights, device)
    if meta:
        ep  = meta.get("epoch", "?")
        vl  = meta.get("val_loss")
        vls = f"{vl:.5f}" if vl is not None else "?"
        print(f"       checkpoint: epoch={ep}  val_loss={vls}")

    # Training log
    log = load_log(args.weights)
    if log:
        print(f"[eval] training log: {len(log)} epochs")
    else:
        print("[eval] no training log (_log.json) found — Fig 1 will be skipped")

    # Val dataset
    print("[eval] loading val frames …")
    cfg = RealDataConfig(val_frac=args.val_frac)
    ds  = RealLabeledDataset(args.real_data, split="val", cfg=cfg)
    print(f"       {len(ds)} val frames available")

    # Run labeler on val set
    n_eval = min(args.eval_frames, len(ds._frames))
    print(f"[eval] running labeler on {n_eval} frames …")
    results = evaluate_frames(model, ds._frames, device, max_frames=n_eval)
    print(f"       done — {len(results)} frames evaluated")

    # Fig 1: training curves
    if log:
        print("[eval] Fig 1 — training curves …")
        plot_training_curves(log, meta, out_dir)

    # Fig 2: per-joint performance
    print("[eval] Fig 2 — per-joint performance …")
    plot_joint_performance(results, out_dir)

    # Fig 3: 3-D overlays (no MANO model needed)
    print("[eval] Fig 3 — 3-D Vicon/skeleton overlays …")
    plot_3d_overlays(results, n_frames=args.n_frames, out_dir=out_dir)

    # Fig 4: MANO mesh overlay (optional)
    if args.mano_dir:
        print("[eval] Fig 4 — MANO mesh overlay …")
        plot_mano_overlay(results, args.mano_dir, n_frames=args.n_frames, out_dir=out_dir)

    print("[eval] done.")


if __name__ == "__main__":
    main()
