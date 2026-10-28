#!/usr/bin/env python3
"""Blank the gap-filled values of a labelled export: keep a value only where the raw Vicon export has a point there.

Used to train the learned filler on P10's manual relabel, whose gap-fills are unreliable.

    python scripts/soma/keep_observed.py --labelled P10/Trial2_handsonly_manuallylabelled_filled.csv \
        --raw P10/Trial2_handsonly.csv --out P10_trial2_manual_observed.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vicon2mano.core.loader import load_csv  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--labelled", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=0.01, help="mm")
a = ap.parse_args()
X, names = load_csv(a.labelled)[:2]
R = load_csv(a.raw)[0]
names = [n.split(":")[-1] for n in names]
keep = np.zeros(X.shape[:2], bool)
for t in range(len(X)):
    r = R[t][np.isfinite(R[t]).all(1)]
    if len(r):
        keep[t] = cKDTree(r).query(np.nan_to_num(X[t], nan=1e9))[0] < a.tol
X[~keep] = np.nan
print(f"kept {keep.mean():.1%} of values; per marker min {keep.mean(0).min():.1%}")
with open(a.out, "w") as f:
    f.write("Frame,Sub Frame," + ",".join(f"{n}_{c}" for n in names for c in "XYZ") + "\n")
    for t in range(len(X)):
        f.write(f"{t + 1},0," + ",".join("" if not np.isfinite(v) else f"{v:.4f}" for v in X[t].ravel()) + "\n")
