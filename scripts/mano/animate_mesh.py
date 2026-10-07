#!/usr/bin/env python3
"""Interactive HTML animation: fitted MANO mesh surface + Vicon marker overlay.

Unlike ``scripts/shared/animate_fit.py`` (skeleton lines only, matplotlib
frames baked into base64 JPEGs), this renders the actual MANO hand
*surface* (778 vertices, ~1538 triangles) with Plotly's native frame
animation -- same Play/Pause + scrub-slider UI already used in
``scripts/gmm/relabel_trial.py``, reused here rather than re-invented.

The fit's ``.npz`` does not store per-frame vertices (``save_npz`` only
keeps joints -- see ``vicon2mano/core/export.py``), so this script re-runs
a cheap, no-grad forward pass through the same MANO model for just the
sampled frames, not all of them.

Usage
-----
python scripts/mano/animate_mesh.py \\
    --npz results/mano/P10_Trial2_handsonly/mano_fit_right.npz \\
    --csv "<raw Vicon CSV>" \\
    --out results/mano/P10_Trial2_handsonly/eval/mesh_animation.html \\
    --n-out 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    ap.add_argument("--n-out", type=int, default=150,
                     help="frames sampled into the animation (mesh rendering is "
                          "heavier than lines, so default is lower than other scripts)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--fps", type=float, default=200.0,
                     help="capture rate in Hz (read from the CSV's Trajectories header, "
                          "not assumed) -- used to make Play advance at real elapsed "
                          "time rather than a fixed per-step duration")
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
    # after np.unique) gap between sampled source frames -- not a fixed
    # guessed duration. A sparse n_out over a long trial still finishes in
    # the trial's real duration, just at coarser visual resolution; use
    # --start/--end to animate a shorter window at full temporal fidelity
    # instead if smoother motion (not just correct timing) is wanted.
    avg_step_frames = (frame_idx[-1] - frame_idx[0]) / max(len(frame_idx) - 1, 1)
    step_ms = (avg_step_frames / args.fps) * 1000.0
    print(f"[animate_mesh] {args.fps:.0f} Hz source, {avg_step_frames:.1f} frames/step "
          f"-> {step_ms:.0f} ms/step (real-time playback)")

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

    # Fixed camera range across the whole animation (same pattern as
    # scripts/gmm/relabel_trial.py's build_figure): compute once from every
    # sampled frame's mesh + markers, not per-frame, so the hand doesn't
    # visually jump as the window re-centres frame to frame.
    all_pts = np.concatenate([
        verts_mm.reshape(-1, 3),
        sampled_markers.reshape(-1, 3)[np.isfinite(sampled_markers.reshape(-1, 3)).all(-1)],
    ], axis=0)
    center = np.nanmean(all_pts, axis=0)
    half = np.nanmax(np.linalg.norm(all_pts - center, axis=-1)) * 1.1

    def marker_xyz(m):
        ok = np.isfinite(m).all(axis=-1)
        return (np.where(ok, m[:, 0], np.nan),
                np.where(ok, m[:, 1], np.nan),
                np.where(ok, m[:, 2], np.nan))

    def mesh_trace(v):
        return go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#e8a0a0", opacity=0.95, flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6),
            lightposition=dict(x=200, y=200, z=400),
            name="MANO mesh", hoverinfo="skip",
        )

    mx0, my0, mz0 = marker_xyz(sampled_markers[0])
    marker_tr = go.Scatter3d(
        x=mx0, y=my0, z=mz0, mode="markers",
        marker=dict(size=5, color="#2c3e50"),
        hovertext=labels, hoverinfo="text", name="Vicon markers",
    )

    frames = []
    for k in range(len(frame_idx)):
        mx, my, mz = marker_xyz(sampled_markers[k])
        frames.append(go.Frame(
            name=str(k),
            data=[mesh_trace(verts_mm[k]), go.Scatter3d(x=mx, y=my, z=mz)],
            layout=go.Layout(title=f"frame {int(frame_idx[k])}"),
        ))

    fig = go.Figure(
        data=[mesh_trace(verts_mm[0]), marker_tr],
        frames=frames,
        layout=go.Layout(
            title=f"frame {int(frame_idx[0])}",
            width=950, height=800,
            scene=dict(
                xaxis=dict(range=[center[0] - half, center[0] + half], title="X (mm)"),
                yaxis=dict(range=[center[1] - half, center[1] + half], title="Y (mm)"),
                zaxis=dict(range=[center[2] - half, center[2] + half], title="Z (mm)"),
                aspectmode="cube"),
            margin=dict(t=80),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.0, y=1.08, buttons=[
                    dict(label="Play", method="animate", args=[None, {
                        "frame": {"duration": step_ms, "redraw": True},
                        "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {
                        "frame": {"duration": 0}, "mode": "immediate"}])])],
            sliders=[dict(currentvalue=dict(prefix="frame: "), steps=[
                dict(method="animate", label=str(int(t)), args=[[str(k)], {
                    "frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                for k, t in enumerate(frame_idx)])],
        ),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, div_id="meshfig")
    print(f"[animate_mesh] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
