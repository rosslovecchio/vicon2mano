#!/usr/bin/env python3
"""Gap-fill accuracy: cut artificial gaps where gap_fill.py's output has observed markers, fill them again, compare.

    python scripts/soma/gap_fill_check.py --filled FILLED.csv [--lengths 1000,5000 --filler filler.pt --max-gap 100000]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vicon2mano.core.loader import load_csv  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--filled", required=True, help="gap_fill.py output CSV (its .npz says which values were observed)")
ap.add_argument("--hz", type=float, default=200.0)
ap.add_argument("--lengths", default="10,50,200,1000", help="artificial gap lengths, frames")
ap.add_argument("--filler", default=None, help="passed to gap_fill.py")
ap.add_argument("--max-gap", default=None, help="passed to gap_fill.py")
args = ap.parse_args()
src, hz = Path(args.filled), args.hz
tmp = Path(tempfile.mkdtemp())
how = np.load(src.with_suffix(".npz"))["how"]
X, names = load_csv(str(src))[:2]
names = [n.split(":")[-1] for n in names]
lines = src.read_text().splitlines()
rng = np.random.default_rng(0)
cuts = []
for nm in ("Palm1", "Index3", "Middle3", "Ring2", "Ring3", "Pinky1", "Pinky3", "Thumb3"):
    m = names.index(nm)
    obs = (how[:, m] == 0) & np.isfinite(X[:, m]).all(1)
    for L in map(int, args.lengths.split(",")):
        n = 0
        for _ in range(200):                                             # up to 6 cuts where the marker was seen
            s = int(rng.integers(L + 100, len(X) - L - 100))
            if n < 6 and obs[s - 1:s + L + 1].all():
                cuts.append((m, s, L))
                n += 1
Y = X.copy()
for m, s, L in cuts:
    Y[s:s + L, m] = np.nan
out = [lines[0]] + [",".join(lines[1 + t].split(",")[:2] + ["" if not np.isfinite(v) else f"{v:.6f}" for v in Y[t].ravel()])
                    for t in range(len(Y))]
(tmp / "in.csv").write_text("\n".join(out) + "\n")
subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).with_name("gap_fill.py")), "--csv", str(tmp / "in.csv"),
                "--out", str(tmp / "out.csv")] + [x for k in ("filler", "max_gap") if getattr(args, k)
                                                  for x in (f"--{k.replace('_', '-')}", str(getattr(args, k)))],
               check=True, capture_output=True)
F = load_csv(str(tmp / "out.csv"))[0]
res = {}
for m, s, L in cuts:
    res.setdefault(L, []).append(np.nanmedian(np.linalg.norm(F[s:s + L, m] - X[s:s + L, m], axis=1)))
for L, v in sorted(res.items()):
    v = np.array(v)
    ok = v[np.isfinite(v)]                                               # a cut the method left empty has no error
    print(f"gap {L:5d} frames ({L / hz:.2f} s): {len(v)} cuts, {len(v) - len(ok)} left empty; filled: median error "
          + (f"{np.median(ok):.1f} mm, worst {np.max(ok):.1f} mm" if len(ok) else "-"))
