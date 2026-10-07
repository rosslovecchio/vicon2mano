#!/usr/bin/env python3
"""HTML animation: fitted MANO mesh surface + Vicon marker overlay, at real
elapsed-time playback speed.

First version of this script used Plotly's live Mesh3d frame animation,
which looked right in principle but played back noticeably slower than
real time in the browser: redrawing a 778-vertex/1538-face mesh every
frame is too expensive for the browser to keep up with the requested
per-frame duration, so Plotly silently falls behind instead of honouring
it. Same lesson ``scripts/shared/animate_fit.py`` already learned for the
skeleton-only animation: render every frame to a static image *offline*
(no per-frame interactivity cost at playback time) and drive it with a
fixed-rate JS timer, so playback speed is decoupled from render cost
entirely. This script now follows that exact pattern -- ``_write_html``
imported directly from ``animate_fit.py`` rather than reimplemented.

Usage
-----
python scripts/mano/animate_mesh.py \\
    --npz results/mano/P10_Trial2_handsonly/mano_fit_right.npz \\
    --csv "<raw Vicon CSV>" \\
    --out results/mano/P10_Trial2_handsonly/eval/mesh_animation.html \\
    --n-out 1500 --fps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SHARED_DIR = REPO_ROOT / "scripts" / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

# chumpy (an smplx dependency) uses removed numpy type aliases -- same
# compat shim as vicon2mano.core.fitter / export_joint_angles.py.
import inspect as _inspect
import numpy as _np
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
for _attr, _builtin in {"int": int, "float": float, "bool": bool, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _builtin)

from vicon2mano.core.loader import load_csv
from animate_fit import _write_html  # noqa: E402 -- path inserted above


def resolve_model_path(mano_dir: str, side: str) -> Path:
    model_path = Path(mano_dir)
    if model_path.is_dir() and not (model_path / "mano").exists():
        side_str = "RIGHT" if side == "right" else "LEFT"
        pkl = model_path / f"MANO_{side_str}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"MANO weights not found at {pkl}")
        model_path = pkl
    return model_path


def forward_vertices(model, global_orient, hand_pose, transl, betas) -> np.ndarray:
    """No-grad forward pass -> (n, 778, 3) mesh vertices in metres."""
    import torch
    n = global_orient.shape[0]
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            hand_pose=torch.tensor(hand_pose, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(n, -1),
            return_verts=True,
        )
    return out.vertices.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="FitResult .npz from vicon2mano")
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--out", required=True, help="output .html path")
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--n-out", type=int, default=1500,
                     help="frames sampled into the animation")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--fps", type=float, default=200.0,
                     help="capture rate in Hz (read from the CSV's Trajectories header, "
                          "not assumed) -- used only to derive playback speed")
    ap.add_argument("--dpi", type=int, default=85, help="render resolution (lower = smaller file)")
    args = ap.parse_args()

    import smplx
    model_path = resolve_model_path(args.mano_dir, args.side)
    model = smplx.create(
        str(model_path), model_type="mano",
        is_rhand=(args.side == "right"), use_pca=True, num_pca_comps=6,
        flat_hand_mean=True, batch_size=1,
    )
    faces = model.faces  # (nf, 3) int

    print(f"[animate_mesh] loading {args.npz}")
    fit = np.load(args.npz)
    print(f"[animate_mesh] loading markers from {args.csv}")
    markers_mm, labels = load_csv(args.csv)
    labels = [l.split(":")[-1] for l in labels]

    T = min(fit["global_orient"].shape[0], markers_mm.shape[0])
    end = T if args.end < 0 else min(args.end, T)
    frame_idx = np.unique(np.linspace(args.start, end - 1, min(args.n_out, end - args.start)).astype(int))
    print(f"[animate_mesh] animating {len(frame_idx)} frames from source range "
          f"[{frame_idx[0]}, {frame_idx[-1]}]")

    # Real elapsed time per animation step, from the actual (roughly uniform,
    # after np.unique) gap between sampled source frames -- the playback fps
    # passed to _write_html's fixed-rate JS timer, so total playback time
    # matches the trial's real recording duration regardless of render cost.
    avg_step_frames = (frame_idx[-1] - frame_idx[0]) / max(len(frame_idx) - 1, 1)
    playback_fps = args.fps / avg_step_frames
    print(f"[animate_mesh] {args.fps:.0f} Hz source, {avg_step_frames:.1f} frames/step "
          f"-> {playback_fps:.1f} fps playback (real-time)")

    print("[animate_mesh] running forward pass for sampled frames...")
    verts_m = forward_vertices(
        model,
        fit["global_orient"][frame_idx],
        fit["hand_pose"][frame_idx],
        fit["transl"][frame_idx],
        fit["betas"],
    )
    verts_mm = verts_m * 1000.0  # metres -> mm, to match raw marker scale
    sampled_markers = markers_mm[frame_idx]  # (n, N, 3) mm

    # Fixed camera range across the whole animation, computed once (same
    # pattern as scripts/gmm/relabel_trial.py's build_figure), so the hand
    # doesn't visually jump as the window re-centres frame to frame.
    flat_markers = sampled_markers.reshape(-1, 3)
    all_pts = np.concatenate([
        verts_mm.reshape(-1, 3),
        flat_markers[np.isfinite(flat_markers).all(-1)],
    ], axis=0)
    center = np.nanmean(all_pts, axis=0)
    half = np.nanmax(np.linalg.norm(all_pts - center, axis=-1)) * 1.1

    print("[animate_mesh] rendering frames...")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(k: int) -> None:
        ax.clear()
        v = verts_mm[k]
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=faces,
                         color="#e8a0a0", edgecolor="none", shade=True,
                         antialiased=False, alpha=0.95)
        m = sampled_markers[k]
        finite = np.isfinite(m).all(axis=-1)
        if finite.any():
            ax.scatter(*m[finite].T, c="#2c3e50", s=18, depthshade=False)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"frame {int(frame_idx[k])}", fontsize=10)
        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Y (mm)", fontsize=7)
        ax.set_zlabel("Z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_html(fig, update, len(frame_idx), playback_fps, args.dpi, out_path)
    plt.close(fig)
    print(f"[animate_mesh] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
