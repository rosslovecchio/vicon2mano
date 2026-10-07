#!/usr/bin/env python3
"""Two-pass MANO fit: calibrate per-marker skin offsets, then refit pose
against offset-corrected targets instead of raw marker positions.

Why: ``MANOFitter``'s objective is ``||J_j(theta,beta) - marker||^2`` --
it fits the joint position *directly* to the marker, which assumes the
marker sits at the joint centre. It doesn't: a marker is taped to skin,
and the offset is systematic, not noise -- measured directly on
P10/Trial2 Hands only (single-frame and window fits, both temporal
contexts, same numbers): MCP/wrist markers run 17-28mm, PIP/DIP markers
5-16mm. ``vicon2mano.strategies.mano.relabel`` already has the machinery
to calibrate this (per-marker offset in a bone-*segment* frame that
rotates with the finger, not the palm, since a marker is taped to one
specific bone) -- built for the relabelling strategy, reused here as-is
rather than reimplemented.

Pass 1: fit normally (what MANOFitter already does).
Calibrate: per-marker offset from pass 1's own joints vs the real
  markers, over the whole trial (more samples than any one window).
Pass 2: refit the same window, but the optimiser's target for each
  marker becomes ``marker - R_segment(pass1 joints) @ offset`` instead of
  the raw marker -- i.e. it now aims the joint at the *skin-corrected*
  position, not the marker itself. Segment frames are computed once from
  pass 1's (fixed) joints, not re-differentiated through pass 2, so this
  avoids a circular shape/offset dependency.

Reports three residual numbers, not one, because "lower joint-to-marker
residual" is not the right success metric once a real offset exists (the
objective is *no longer* trying to make them equal):
  raw           : |fitted joint - real marker|            (old metric)
  pass-1 proxy  : |pass-1 joint + R@offset - real marker|  (post-hoc correction
                  of the ORIGINAL fit, no refit)
  pass-2        : |pass-2 joint + R@offset - real marker|  (the refit's own
                  prediction, the number that actually matters)

Usage
-----
python scripts/mano/refit_with_marker_offsets.py \\
    --csv "<raw Vicon CSV>" \\
    --whole-trial-npz results/mano/P10_Trial2_handsonly/mano_fit_right.npz \\
    --start 30663 --end 31663 --highlight-frame 31163 \\
    --out results/mano/P10_Trial2_handsonly/eval/frame31163_offsetfit.html
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
from vicon2mano.core.fitter import FitConfig, MANOFitter
from vicon2mano.strategies.mano.relabel import (
    marker_joint_map, calibrate_marker_offsets, _segment_frames,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--whole-trial-npz", required=True,
                     help="existing whole-trial FitResult .npz, used only to supply "
                          "joints for offset calibration (not refit)")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--highlight-frame", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()

    markers_mm, labels = load_csv(args.csv)
    base_labels = [l.split(":")[-1] for l in labels]

    # ---- calibrate per-marker offsets from the whole trial's own fit ----
    whole = np.load(args.whole_trial_npz)
    T_whole = whole["joints"].shape[0]
    m2j = marker_joint_map(labels, side=args.side)
    print(f"[refit] {len(m2j)} markers map to a MANO joint")
    offsets, spreads = calibrate_marker_offsets(
        markers_mm[:T_whole], np.arange(T_whole), whole["joints"], m2j,
        status=None, min_samples=args.min_samples,
    )
    print("Calibrated offsets (mm magnitude) and pose-invariance spread (mm):")
    for m_idx, off in sorted(offsets.items()):
        print(f"  marker {m_idx:2d} ({base_labels[m_idx]:10s}): "
              f"|offset|={np.linalg.norm(off) * 1000:6.2f}mm  spread={spreads[m_idx]:5.2f}mm")

    # ---- pass 1: normal fit on the window ----
    window = markers_mm[args.start:args.end + 1]
    highlight_local = args.highlight_frame - args.start
    cfg = FitConfig(mano_model_path=args.mano_dir, hand_side=args.side)
    fitter = MANOFitter(cfg)
    print("\n[refit] pass 1: normal fit (raw marker targets)...")
    pass1 = fitter.fit(window, labels, verbose=True)

    # ---- build offset-corrected targets using pass-1's (fixed) joints ----
    assign = pass1.assignment
    R = _segment_frames(pass1.joints)  # (T, 21, 3, 3), from pass-1 joints, held fixed
    window_m = window.astype(np.float32) * cfg.marker_scale
    T = window.shape[0]
    corrected = np.full((T, 21, 3), np.nan, dtype=np.float32)
    for j, mi in enumerate(assign):
        if mi < 0:
            continue
        if mi in offsets:
            corrected[:, j] = window_m[:, mi] - np.einsum("tab,b->ta", R[:, j], offsets[mi])
        else:
            corrected[:, j] = window_m[:, mi]  # no calibrated offset: fall back to raw

    # ---- pass 2: refit against the corrected targets, betas frozen ----
    print("\n[refit] pass 2: refit against offset-corrected targets...")
    go2, hp2, tr2 = fitter._optimise_pose(
        window_m, assign, pass1.betas, verbose=True, target_override=corrected,
    )
    joints2_m = fitter._joints_np(go2, hp2, tr2, pass1.betas)

    # ---- residuals at the highlighted frame ----
    hl = highlight_local
    real_marker_mm = window[hl]
    R_hl = R[hl]

    def residual_table(joints_mm, label):
        print(f"\n{label}:")
        rs = []
        for j, mi in enumerate(assign):
            if mi < 0 or not np.isfinite(real_marker_mm[mi]).all():
                continue
            d = np.linalg.norm(joints_mm[j] - real_marker_mm[mi])
            rs.append(d)
            print(f"  joint {j:2d} <- marker {mi:2d} ({base_labels[mi]:10s}): {d:6.2f} mm")
        print(f"  mean: {np.mean(rs):.2f} mm")
        return np.mean(rs)

    joints1_mm = pass1.joints[hl] * 1000.0
    joints2_mm = joints2_m[hl] * 1000.0
    residual_table(joints1_mm, "RAW: pass-1 joint vs real marker (old metric)")
    residual_table(joints2_mm, "RAW: pass-2 joint vs real marker (expected to be LARGER at "
                                "MCP -- that's the point, joint no longer targets the marker)")

    def proxy_table(joints_mm, label):
        print(f"\n{label}:")
        rs = []
        for j, mi in enumerate(assign):
            if mi < 0 or mi not in offsets or not np.isfinite(real_marker_mm[mi]).all():
                continue
            pred = joints_mm[j] + (R_hl[j] @ offsets[mi]) * 1000.0
            d = np.linalg.norm(pred - real_marker_mm[mi])
            rs.append(d)
            print(f"  joint {j:2d} -> predicted marker {mi:2d} ({base_labels[mi]:10s}): {d:6.2f} mm")
        print(f"  mean: {np.mean(rs):.2f} mm")
        return np.mean(rs)

    proxy_table(joints1_mm, "PROXY: pass-1 joint + offset vs real marker (post-hoc correction, no refit)")
    proxy_table(joints2_mm, "PROXY: pass-2 joint + offset vs real marker (the number that matters)")

    # ---- plot pass-2 mesh + markers for the highlighted frame ----
    import torch
    with torch.no_grad():
        out = fitter._model(
            global_orient=torch.tensor(go2[hl:hl + 1]),
            hand_pose=torch.tensor(hp2[hl:hl + 1]),
            transl=torch.tensor(tr2[hl:hl + 1]),
            betas=torch.tensor(pass1.betas).unsqueeze(0),
            return_verts=True,
        )
    verts_mm = out.vertices[0].numpy() * 1000.0
    faces = fitter._model.faces
    finite = np.isfinite(real_marker_mm).all(axis=-1)
    thumb_tip, index_tip = verts_mm[744], verts_mm[320]
    print(f"\nThumb tip - Index tip distance (pass-2, offset-corrected refit): "
          f"{np.linalg.norm(thumb_tip - index_tip):.1f} mm "
          f"(unchanged from ~50mm in every prior fit would confirm this is structural, "
          f"not a targeting-bias artifact -- the tip still has no marker, corrected or not)")

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#e8a0a0", opacity=0.95, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.15, specular=0.0),
            name="MANO mesh (offset-corrected refit)",
        ),
        go.Scatter3d(
            x=real_marker_mm[finite, 0], y=real_marker_mm[finite, 1], z=real_marker_mm[finite, 2],
            mode="markers+text", marker=dict(size=5, color="#2c3e50"),
            text=[base_labels[i] for i in np.flatnonzero(finite)],
            textposition="top center", name="Vicon markers",
        ),
    ])
    fig.update_layout(
        title=f"Offset-corrected refit — window [{args.start},{args.end}], frame {args.highlight_frame}",
        scene=dict(aspectmode="data"),
        width=1000, height=850, margin=dict(t=60, b=10),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"\n[refit] saved {out_path}")


if __name__ == "__main__":
    main()
