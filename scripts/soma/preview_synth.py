#!/usr/bin/env python3
"""Does the synthetic data look like the real data? Numbers + a visual preview.

Compares synthetic clouds with real P10/Trial2 frames (manual labels, real
observations only) on: per-pair marker distances, nearest-neighbour spacing
(how close the labeller's confusable markers sit) and missing-marker counts.
Writes an HTML with real and synthetic clouds side by side, coloured by label.

    python scripts/soma/preview_synth.py --out results/soma/preview_synth.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT
from vicon2mano.core.fitter import FitConfig, MANOFitter
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.library import MarkerLibrary
from vicon2mano.strategies.soma.synth import SynthConfig, SynthGenerator

FINGER_COL = {"Forearm": "#7f7f7f", "Palm": "#2b2b2b", "Thumb": "#d62728", "Index": "#ff7f0e",
              "Middle": "#2ca02c", "Ring": "#1f77b4", "Pinky": "#9467bd"}


def colour(name):
    return next((c for k, c in FINGER_COL.items() if name.startswith(k)), "#000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(REPO / "results/soma/library_P10_Trial2.npz"))
    ap.add_argument("--mano-dir", default=str(REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=str(REPO / "results/soma/preview_synth.html"))
    args = ap.parse_args()

    lib = MarkerLibrary.load(args.library)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MANOFitter(FitConfig(mano_model_path=args.mano_dir, hand_side="right", use_pca=False))._model
    gen = SynthGenerator(lib, model, SynthConfig(), device=dev, seed=0)
    M = gen.M

    gen.sample(256)                                   # warm-up
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time(); pts, valid, lab = gen.sample(args.n)
    if dev == "cuda": torch.cuda.synchronize()
    print(f"generated {args.n} clouds in {time.time()-t:.2f}s on {dev}")
    pts, valid, lab = pts.cpu().numpy(), valid.cpu().numpy(), lab.cpu().numpy()
    S = np.full((args.n, M, 3), np.nan)
    b, i = np.nonzero(lab >= 0); S[b, lab[b, i]] = pts[b, i]

    # real: manual positions at real observations only (library rule)
    man, nm = load_csv(str(Path(DATA_ROOT) / "P10/Trial2_handsonly_manuallylabelled_filled.csv"))[:2]
    labels = [n.split(":")[-1] for n in nm]
    assert labels == lib.labels
    Rl = np.full((len(lib.frames), M, 3), np.nan)
    ok = np.isfinite(lib.residual).all(-1)
    Rl[ok] = man[lib.frames][ok]

    def pair_d(X):
        return np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)

    rs = np.random.default_rng(0).choice(len(Rl), min(len(Rl), args.n), replace=False)
    Dr, Ds = pair_d(Rl[rs]), pair_d(S)
    iu = np.triu_indices(M, 1)
    med_r, med_s = np.nanmedian(Dr, 0)[iu], np.nanmedian(Ds, 0)[iu]
    iqr = lambda D: (np.nanpercentile(D, 75, 0) - np.nanpercentile(D, 25, 0))[iu]
    iqr_r, iqr_s = iqr(Dr), iqr(Ds)
    print(f"\npair distances over {len(iu[0])} marker pairs (synthetic vs real):")
    print(f"  median: mean |diff| {np.mean(np.abs(med_s-med_r)):.1f} mm, worst {np.max(np.abs(med_s-med_r)):.1f} mm "
          f"({lib.labels[iu[0][np.argmax(np.abs(med_s-med_r))]]}-{lib.labels[iu[1][np.argmax(np.abs(med_s-med_r))]]})")
    print(f"  spread (IQR): real median {np.median(iqr_r):.1f} mm, synthetic {np.median(iqr_s):.1f} mm, "
          f"ratio synth/real median {np.median(iqr_s/np.maximum(iqr_r,1e-6)):.2f}")
    for nm_ in ("Index3", "Middle3", "Ring3", "Pinky3"):
        m = lib.labels.index(nm_)
        nn_r = np.nanmin(np.where(np.eye(M, dtype=bool)[m], np.nan, Dr[:, m]), 1)
        nn_s = np.nanmin(np.where(np.eye(M, dtype=bool)[m], np.nan, Ds[:, m]), 1)
        print(f"  {nm_} nearest other marker: real median {np.nanmedian(nn_r):5.1f} p10 {np.nanpercentile(nn_r,10):5.1f} | "
              f"synth median {np.nanmedian(nn_s):5.1f} p10 {np.nanpercentile(nn_s,10):5.1f} mm")
    miss_s = M - (lab >= 0).sum(1)
    print(f"\nmissing markers per synthetic cloud: " +
          ", ".join(f"{k}:{100*(miss_s==k).mean():.0f}%" for k in range(0, 8)) +
          f"; ghosts/cloud mean {(lab == -1).sum(1).mean():.2f}")

    # visual: 4 real frames (re-centred) and 8 synthetic clouds
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels = [("real", Rl[k] - np.nanmedian(Rl[k], 0)) for k in rs[:4]]
    for k in range(8):
        X = np.full((M + 1, 3), np.nan); v = valid[k]
        panels.append(("synthetic", (pts[k][v], lab[k][v])))
    fig = make_subplots(rows=3, cols=4, specs=[[{"type": "scene"}] * 4] * 3,
                        subplot_titles=[f"{p[0]} {i+1}" for i, p in enumerate(panels)])
    for pi, (kind, data) in enumerate(panels):
        r, c = pi // 4 + 1, pi % 4 + 1
        if kind == "real":
            X, L = data, np.arange(M)
            keep = np.isfinite(X).all(-1); X, L = X[keep], L[keep]
        else:
            X, L = data
        txt = [lib.labels[l] if l >= 0 else "ghost" for l in L]
        col = [colour(lib.labels[l]) if l >= 0 else "#e377c2" for l in L]
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers", text=txt,
                                   marker=dict(size=4, color=col), showlegend=False,
                                   hovertemplate="%{text}<extra></extra>"), row=r, col=c)
    fig.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(height=1100, width=1400, title="strategySOMA synthetic clouds vs real P10/Trial2 frames "
                      "(colour = finger; pink = ghost)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
