#!/usr/bin/env python3
"""Fit MANO to a short window of frames and plot one highlighted frame.

Originally built to check whether a whole-trial fit's residual at one
frame is a temporal-smoothing artifact or a structural limit (e.g. an
unconstrained fingertip -- no Vicon marker exists there at all, so no
amount of per-frame freedom fixes it; confirmed: isolated single-frame
fitting barely moved the thumb/index tip gap, 53.3mm -> 49.9mm, while the
mean *matched-joint* residual barely moved either, 15.4mm -> 14.5mm).

Follow-up use: fit a window of frames (keeps temporal smoothing active,
unlike an isolated single frame) around a highlighted frame, to check
whether the markers that DO exist (not the unconstrained tip) end up
fit more tightly -- a dedicated, local, small-batch fit may do better than
a frame buried inside one 2000-frame chunk of a 42k-frame whole-trial fit.

Writes an interactive, freely-rotatable Plotly HTML for the highlighted
frame only (a single static frame needs no baking/animation). Lighting is
ambient-only (not the default diffuse+specular) -- this session found
diffuse/specular shading on this mesh topology produces dark hatching
artifacts, present even with Plotly's correct WebGL depth-buffering, so
it is a lighting-model issue, not a geometry hole; ambient-only avoids it.

Usage
-----
python scripts/mano/fit_single_frame.py \\
    --csv "<raw Vicon CSV>" \\
    --frame 31163 \\
    --out results/mano/P10_Trial2_handsonly/eval/frame31163_singlefit.html

python scripts/mano/fit_single_frame.py \\
    --csv "<raw Vicon CSV>" \\
    --start 30663 --end 31663 --highlight-frame 31163 \\
    --out results/mano/P10_Trial2_handsonly/eval/frame31163_windowfit.html
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

import inspect as _inspect
import numpy as _np
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
for _attr, _builtin in {"int": int, "float": float, "bool": bool, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _builtin)

from vicon2mano.core.loader import load_csv
from vicon2mano.core.fitter import FitConfig, MANOFitter, mano_output_to_joints21


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--frame", type=int, default=None,
                     help="fit this one frame in isolation (no temporal smoothing)")
    ap.add_argument("--start", type=int, default=None, help="window start frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="window end frame (inclusive)")
    ap.add_argument("--highlight-frame", type=int, default=None,
                     help="which frame within --start/--end to plot and report "
                          "residuals for (default: the window's midpoint)")
    ap.add_argument("--out", required=True, help="output .html path")
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    args = ap.parse_args()

    if args.frame is not None:
        start, end = args.frame, args.frame
    elif args.start is not None and args.end is not None:
        start, end = args.start, args.end
    else:
        raise SystemExit("pass either --frame, or both --start and --end")
    highlight = args.highlight_frame if args.highlight_frame is not None else (start + end) // 2
    if not (start <= highlight <= end):
        raise SystemExit(f"--highlight-frame {highlight} outside window [{start}, {end}]")

    markers_mm, labels = load_csv(args.csv)
    base_labels = [l.split(":")[-1] for l in labels]
    window = markers_mm[start:end + 1]  # (T, N, 3)
    local_idx = highlight - start

    cfg = FitConfig(mano_model_path=args.mano_dir, hand_side=args.side)
    fitter = MANOFitter(cfg)
    result = fitter.fit(window, labels, verbose=True)

    # Mesh vertices for just the highlighted frame.
    import torch
    with torch.no_grad():
        out = fitter._model(
            global_orient=torch.tensor(result.global_orient[local_idx:local_idx + 1]),
            hand_pose=torch.tensor(result.hand_pose[local_idx:local_idx + 1]),
            transl=torch.tensor(result.transl[local_idx:local_idx + 1]),
            betas=torch.tensor(result.betas).unsqueeze(0),
            return_verts=True,
        )
    verts_mm = out.vertices[0].numpy() * 1000.0
    faces = fitter._model.faces

    # Residual: fitted joint vs its assigned marker, for the highlighted frame.
    joints_mm = result.joints[local_idx] * 1000.0
    marker_frame = window[local_idx]
    assign = result.assignment
    print(f"Per-joint marker residual (mm), window [{start},{end}], highlight frame {highlight}:")
    for j, mi in enumerate(assign):
        if mi < 0:
            continue
        d = np.linalg.norm(joints_mm[j] - marker_frame[mi])
        print(f"  joint {j:2d} <- marker {mi:2d} ({base_labels[mi]:10s}): {d:6.2f} mm")
    res = [np.linalg.norm(joints_mm[j] - marker_frame[assign[j]]) for j in range(21) if assign[j] >= 0]
    print(f"Mean residual (matched joints only): {np.mean(res):.2f} mm")

    # Thumb/index tip gap, same vertices used throughout this session.
    thumb_tip, index_tip = verts_mm[744], verts_mm[320]
    print(f"Thumb tip - Index tip distance: {np.linalg.norm(thumb_tip - index_tip):.1f} mm")

    finite = np.isfinite(marker_frame).all(axis=-1)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#e8a0a0", opacity=0.95, flatshading=True,
            # ambient-only lighting: avoids the diffuse/specular shading
            # artifact found this session (dark hatching on this mesh
            # topology, present even with correct WebGL depth-buffering --
            # a lighting-model issue, not a geometry hole).
            lighting=dict(ambient=1.0, diffuse=0.15, specular=0.0),
            name="MANO mesh",
        ),
        go.Scatter3d(
            x=marker_frame[finite, 0], y=marker_frame[finite, 1], z=marker_frame[finite, 2],
            mode="markers+text", marker=dict(size=5, color="#2c3e50"),
            text=[base_labels[i] for i in np.flatnonzero(finite)],
            textposition="top center", name="Vicon markers",
        ),
    ])
    fig.update_layout(
        title=f"MANO fit — window [{start},{end}], frame {highlight}",
        scene=dict(aspectmode="data"),
        width=1000, height=850, margin=dict(t=60, b=10),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"[fit_single_frame] saved {out_path}")


if __name__ == "__main__":
    main()
