#!/usr/bin/env python3
"""Score fit_trials.py runs on the markers that were really observed, so runs with different gap fills compare fairly.

Per trial and run: median over fitted frames of the largest and the median distance (mm) between an observed marker
and its attached vertex, and of the largest distance over all markers (filled ones included). Only batches every
run fitted are scored.

    python scripts/soma_all/score_observed.py P16 rigid=results/soma/movement real=results/soma/real_fill/movement
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vicon2mano.core.loader import load_csv  # noqa: E402

p, runs = sys.argv[1], dict(r.split("=", 1) for r in sys.argv[2:])
trials = sorted({d.name for r in runs.values() for d in (Path(r) / p).iterdir() if d.is_dir()})
for trial in trials:
    common = set.intersection(*({q.name for q in (Path(r) / p / trial).iterdir() if (q / "pred.npz").exists()} for r in runs.values()))
    row = []
    for k, r in runs.items():
        d = Path(r) / p / trial
        filled = d / "soma_labelled_filled.csv"
        how = np.load(filled.with_suffix(".npz"))["how"]
        cols = [n.split(":")[-1] for n in load_csv(str(filled))[1]]
        mx, md, am = [], [], []
        for b in sorted(common):
            z = np.load(d / b / "pred.npz")
            obs = how[z["frames"]][:, [cols.index(str(n)) for n in z["names"]]] == 0
            D = np.where(obs, z["dist_mm"], np.nan)
            mx.append(np.nanmax(D, 1)), md.append(np.nanmedian(D, 1)), am.append(np.nanmax(z["dist_mm"], 1))
        f = lambda v: np.nanmedian(np.concatenate(v))
        row.append(f"{k} {f(mx):.1f} / {f(md):.1f} / {f(am):.1f}")
    print(f"{trial:20s}", " | ".join(row))
