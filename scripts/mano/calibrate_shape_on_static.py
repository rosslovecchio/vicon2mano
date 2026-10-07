#!/usr/bin/env python3
"""Calibrate MANO shape (betas) on a static trial, then fit pose with it frozen.

Why a static: bone length is pose-invariant, so one pose is enough (unlike a
spatial prior, which CLAUDE.md correctly rules statics out for). A static is
the *best* pose for it -- the hand is still (0.04-0.30mm marker sd on P10 vs
1.8-4.2mm during movement) and extended, so consecutive markers' skin offsets
are roughly parallel and partly cancel in the difference.

Two things this has to work around, both measured rather than assumed:

1. ``MANOFitter``'s shape stage is effectively a no-op: ``w_shape=1e-2``
   against a ~1e-3 joint loss, plus ``lr=3e-3`` over 100 iters, leaves betas
   pinned at ~0 (verified: fitted betas come back [0.005, 0, -0, ...]). This
   script therefore runs its own shape fit with ``w_shape``/lr set so betas
   can actually move, instead of calling the stock shape stage.
2. Parts of the static are mislabelled. On P10: Thumb1-Thumb2 reads 90.6mm
   against 39.7mm in the verified movement trial (anatomically impossible for
   a thumb phalanx) and Palm2-Thumb1 reads 129mm against 55mm -- CLAUDE.md's
   documented case. So the thumb is excluded by default. ``Palm2`` is excluded
   too, for a different reason: it is a mid-palm marker standing in as the
   wrist (``_LABEL_HINTS``), and sits ~32mm closer to the MCPs than MANO's
   wrist joint, so including it actively drags the shape fit.

Screen the static before trusting it: ``--report-only`` prints each segment's
static-vs-movement median distance so you can see which are consistent.

Usage
-----
python scripts/mano/calibrate_shape_on_static.py \\
    --static "<static CSV>" --movement "<movement CSV>" \\
    --start 30663 --end 31663 --highlight-frame 31163 \\
    --out results/mano/P10_Trial2_handsonly/eval/frame31163_staticshape.html
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

CHAINS = [("Thumb1", "Thumb2"), ("Thumb2", "Thumb3"),
          ("Index1", "Index2"), ("Index2", "Index3"),
          ("Middle1", "Middle2"), ("Middle2", "Middle3"),
          ("Ring1", "Ring2"), ("Ring2", "Ring3"),
          ("Pinky1", "Pinky2"), ("Pinky2", "Pinky3"),
          ("Palm1", "Palm2"), ("Palm2", "Palm3"), ("Palm1", "Palm3")]


def med_dist(arr, base, a, b):
    if a not in base or b not in base:
        return float("nan")
    d = np.linalg.norm(arr[:, base.index(a)] - arr[:, base.index(b)], axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("nan")


def screen_static(st, bs, mv, bm, tol=5.0):
    """Print static-vs-movement agreement per segment; return the bad names."""
    print(f'{"segment":20s} {"STATIC":>9s} {"MOVEMENT":>10s} {"diff":>8s}')
    bad = set()
    for a, b in CHAINS:
        ds, dm = med_dist(st, bs, a, b), med_dist(mv, bm, a, b)
        if not np.isfinite(ds) or not np.isfinite(dm):
            continue
        flag = ""
        if abs(ds - dm) > tol:
            flag = "  <-- disagrees, static suspect"
            bad.update((a, b))
        print(f'{a + "-" + b:20s} {ds:7.2f}mm {dm:8.2f}mm {ds - dm:7.2f}{flag}')
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", required=True)
    ap.add_argument("--movement", required=True)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--highlight-frame", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--exclude", default="Thumb1,Thumb2,Thumb3,Palm2",
                     help="markers to drop from the STATIC shape fit (see module docstring)")
    ap.add_argument("--n-static-frames", type=int, default=60)
    ap.add_argument("--w-shape", type=float, default=1e-5)
    ap.add_argument("--lr-beta", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    st, lab_s = load_csv(args.static)
    mv, lab_m = load_csv(args.movement)
    bs = [l.split(":")[-1] for l in lab_s]
    bm = [l.split(":")[-1] for l in lab_m]

    print("=== static vs movement segment screening ===")
    screen_static(st, bs, mv, bm)
    if args.report_only:
        return

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    print(f"\nexcluded from the static shape fit: {sorted(excluded)}")

    cfg = FitConfig(mano_model_path=args.mano_dir, hand_side=args.side)
    fitter = MANOFitter(cfg)
    model = fitter._model
    dev = fitter.device

    # ---- which joint each static marker drives (current _LABEL_HINTS) ----
    from vicon2mano.core.correspondence import label_seed
    seed = label_seed(lab_s, side=args.side)
    if seed is None:
        raise SystemExit("could not seed correspondences from the static's labels")
    pairs = [(j, mi) for j, mi in seed.items() if bs[mi] not in excluded]
    print(f"static shape fit uses {len(pairs)} marker->joint pairs")

    from vicon2mano.core.correspondence import MANO_JOINT_NAMES
    jidx = {n: i for i, n in enumerate(MANO_JOINT_NAMES)}

    sf = np.linspace(0, st.shape[0] - 1, min(args.n_static_frames, st.shape[0])).astype(int)
    S = len(sf)
    tgt = torch.full((S, 21, 3), float("nan"), device=dev)
    for jname, mi in pairs:
        tgt[:, jidx[jname]] = torch.tensor(st[sf][:, mi] * 1e-3, dtype=torch.float32, device=dev)
    mask = torch.isfinite(tgt).all(-1)

    betas = nn.Parameter(torch.zeros(10, device=dev))
    # NB: not named `go` -- that is the plotly.graph_objects alias in this module
    g_orient = nn.Parameter(torch.zeros(S, 3, device=dev))
    hp = nn.Parameter(torch.zeros(S, fitter._pose_dim(), device=dev))
    tr = nn.Parameter(torch.tensor(fitter._init_transl(tgt.detach().cpu().numpy()), device=dev))
    opt = torch.optim.Adam([{"params": [betas], "lr": args.lr_beta},
                            {"params": [hp], "lr": 5e-2},
                            {"params": [g_orient], "lr": 1e-2},
                            {"params": [tr], "lr": 3e-3}])
    print(f"\nfitting shape on {S} static frames...")
    for it in range(args.iters):
        opt.zero_grad()
        out = model(global_orient=g_orient, hand_pose=hp, transl=tr,
                    betas=betas.unsqueeze(0).expand(S, -1))
        pred = mano_output_to_joints21(out)
        loss = ((pred - tgt)[mask] ** 2).sum(-1).mean() + args.w_shape * (betas ** 2).sum()
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            print(f"  iter {it+1:4d}  loss={loss.item():.6f}  "
                  f"resid={((pred-tgt)[mask].norm(dim=-1).mean().item()*1000):.2f}mm")
    betas_np = betas.detach().cpu().numpy()
    print(f"\ncalibrated betas = {betas_np.round(3)}")

    # bone lengths under the calibrated shape vs the static's own marker spacing
    with torch.no_grad():
        o0 = model(global_orient=torch.zeros(1, 3, device=dev),
                   hand_pose=torch.zeros(1, fitter._pose_dim(), device=dev),
                   betas=betas.detach().unsqueeze(0))
        J = mano_output_to_joints21(o0)[0].cpu().numpy() * 1000
    print(f'\n{"finger":8s} {"marker 1->3 (static)":>21s} {"MANO mcp->tip":>14s}  ratio')
    for f, (m_, t_) in {"Index": (5, 8), "Middle": (9, 12),
                         "Ring": (13, 16), "Pinky": (17, 20)}.items():
        k = med_dist(st, bs, f + "1", f + "3")
        n = np.linalg.norm(J[m_] - J[t_])
        print(f"{f:8s} {k:18.2f}mm {n:11.2f}mm  {k/n:5.2f}")

    if args.start is None or args.out is None:
        return

    # ---- fit pose on the movement window with these betas FROZEN ----
    window = mv[args.start:args.end + 1]
    hl = (args.highlight_frame if args.highlight_frame is not None
          else (args.start + args.end) // 2) - args.start
    assign_full = fitter.fit(window[:2], lab_m, verbose=False).assignment  # cheap: get assignment
    window_m = window.astype(np.float32) * cfg.marker_scale
    mk = window[hl]

    def fit_and_score(betas_use, tag):
        g2, h2, t2 = fitter._optimise_pose(window_m, assign_full, betas_use, verbose=False)
        jm = fitter._joints_np(g2, h2, t2, betas_use)[hl] * 1000
        r = [np.linalg.norm(jm[j] - mk[mi]) for j, mi in enumerate(assign_full)
             if mi >= 0 and np.isfinite(mk[mi]).all()]
        print(f"  {tag:34s} mean residual at frame {args.highlight_frame}: {np.mean(r):6.2f} mm")
        return (g2, h2, t2), float(np.mean(r))

    print(f"\nfitting pose on movement window [{args.start},{args.end}], betas frozen:")
    _, r_base = fit_and_score(np.zeros(10, dtype=np.float32),
                               "betas=0 (current default)")
    (go2, hp2, tr2), r_static = fit_and_score(betas_np, "static-calibrated betas")
    print(f"  -> {100 * (1 - r_static / r_base):+.1f}% change from static shape calibration")
    res = [r_static]

    with torch.no_grad():
        out = model(global_orient=torch.tensor(go2[hl:hl+1], device=dev),
                    hand_pose=torch.tensor(hp2[hl:hl+1], device=dev),
                    transl=torch.tensor(tr2[hl:hl+1], device=dev),
                    betas=torch.tensor(betas_np, device=dev).unsqueeze(0), return_verts=True)
    V = out.vertices[0].cpu().numpy() * 1000
    faces = model.faces
    fin = np.isfinite(mk).all(-1)
    fig = go.Figure(data=[
        go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                  color="#e8a0a0", opacity=0.95, flatshading=True,
                  lighting=dict(ambient=1.0, diffuse=0.15, specular=0.0),
                  name="MANO (static-calibrated shape)"),
        go.Scatter3d(x=mk[fin, 0], y=mk[fin, 1], z=mk[fin, 2], mode="markers+text",
                     marker=dict(size=5, color="#2c3e50"),
                     text=[bm[i] for i in np.flatnonzero(fin)],
                     textposition="top center", name="Vicon markers"),
    ])
    fig.update_layout(title=f"Static-calibrated shape — frame {args.highlight_frame} "
                            f"(mean residual {np.mean(res):.1f} mm)",
                      scene=dict(aspectmode="data"), width=1000, height=850)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
