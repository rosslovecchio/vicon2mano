#!/usr/bin/env python3
"""Gap filling for a labelled marker CSV: spline for short gaps, rigid-body (pattern) fill for longer ones.

Labelled positions that jump more than --jump mm are dropped first (label switches).
A gap of at most --spline frames with both edges observed is filled by a cubic spline. A longer gap, up to
--max-gap frames (also at the ends of the trial, from its one observed edge), is filled frame by frame from three
donor markers: among the six whose distance to the missing marker is steadiest around the gap, the steadiest
triplet visible in that frame; the marker's coordinates in the donors' frame are taken from the nearest frames
before and after the gap where marker and triplet were all seen, blended linearly across the gap. Writes the CSV
in the same layout, a JSON report and an .npz of what was filled.

With --filler (scripts/soma_all/train_filler.py) the longer gaps are filled by the learned filler instead: the marker
predicted from the other markers seen in each frame, shifted to meet its observed position at both gap edges.
"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vicon2mano.core.loader import load_csv

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--spline", type=int, default=10, help="frames: spline fill up to this gap length")
ap.add_argument("--max-gap", type=int, default=2000, help="frames: rigid fill up to this gap length")
ap.add_argument("--jump", type=float, default=8.0, help="mm: a labelled position this far from its last one starts a new piece")
ap.add_argument("--window", type=int, default=50, help="frames either side used to pick donors")
ap.add_argument("--filler", default=None, help="learned filler .pt: replaces the rigid fill for gaps longer than --spline")
a = ap.parse_args()

X, names = load_csv(a.csv)[:2]
X = X.astype(np.float64)
names = [n.split(":")[-1] for n in names]
T, M = X.shape[:2]
seen = np.isfinite(X).all(-1)
# label switches (a marker jumping onto another point): drop the short side of each jump, so only continuous
# stretches remain and the filling does not interpolate between two different points
dropped, dropmask = {}, np.zeros(X.shape[:2], bool)
for m in range(M):
    idx = np.nonzero(seen[:, m])[0]
    if len(idx) < 2:
        continue
    cut = np.nonzero(np.linalg.norm(np.diff(X[idx, m], axis=0), axis=1) > a.jump)[0] + 1
    pieces = np.split(idx, cut)
    longest = max(len(q) for q in pieces)
    bad = np.concatenate([q for q in pieces if len(q) < min(longest, 200)] or [np.zeros(0, int)])
    X[bad, m] = np.nan
    dropmask[bad, m] = True
    dropped[names[m]] = int(len(bad))
seen = np.isfinite(X).all(-1)
Y = X.copy()
how = np.zeros((T, M), np.int8)                                  # 0 observed/empty, 1 spline, 2 rigid, 3 jump left empty, 4 learned
if a.filler:
    import torch
    from vicon2mano.strategies.soma import fill
    filler = fill.load(a.filler, "cuda" if torch.cuda.is_available() else "cpu")
    order = [names.index(n) for n in filler.labels]


def learned(m, fr):
    """The filler's prediction of marker ``m`` at frames ``fr`` from the other markers observed there."""
    Z = X[fr][:, order]
    Z[:, order.index(m)] = np.nan
    return fill.predict(filler, Z)[:, order.index(m)]


def frame(P):
    """Origin and orthonormal axes from three points (None if they are nearly collinear)."""
    e1 = P[1] - P[0]; e1 /= np.linalg.norm(e1)
    e2 = P[2] - P[0]; e2 = e2 - e1 * (e2 @ e1)
    if np.linalg.norm(e2) < 5.0:                                             # mm off the line through the first two
        return None
    e2 /= np.linalg.norm(e2)
    return P[0], np.stack([e1, e2, np.cross(e1, e2)])


def reference(m, tri, start, step, limit):
    """Nearest frame from ``start`` going ``step`` where the marker and the triplet were all seen: (frame, coords)."""
    for r in range(start, start + step * limit, step):
        if not 0 <= r < T:
            return None
        if seen[r, m] and seen[r, tri].all():
            f = frame(X[r, tri])
            return None if f is None else (r, f[1] @ (X[r, m] - f[0]))
    return None


report = {}
for m, nm in enumerate(names):
    miss = ~seen[:, m]
    d = np.diff(np.r_[0, miss.astype(int), 0])
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]           # gap = [s, e)
    cnt = dict(gaps=int(len(starts)), missing=int(miss.sum()), spline=0, rigid=0, learned=0, left=0)
    for s, e in zip(starts, ends):
        L = e - s
        if L > a.max_gap or (s == 0 and e == T):
            cnt["left"] += L
            continue
        if L <= a.spline and 0 < s and e < T:
            k = np.r_[max(0, s - 4):s, e:min(T, e + 4)]
            k = k[seen[k, m]]
            if len(k) >= 2:
                Y[s:e, m] = CubicSpline(k, X[k, m])(np.arange(s, e))
                how[s:e, m] = 1
                cnt["spline"] += L
                continue
        if a.filler:
            edges = [t for t in (s - 1, e) if 0 <= t < T]
            P = learned(m, np.r_[s:e, edges])
            off = {t: X[t, m] - p for t, p in zip(edges, P[L:]) if np.isfinite(p).all()}
            w = (np.arange(s, e) - (s - 1)) / (e - s + 1)
            if len(off) == 2:
                P[:L] += (1 - w)[:, None] * off[s - 1] + w[:, None] * off[e]
            elif off:
                P[:L] += next(iter(off.values()))
            ok = np.isfinite(P[:L]).all(1)
            Y[s:e][ok, m] = P[:L][ok]
            how[s:e][ok, m] = 4
            cnt["learned"] += int(ok.sum())
            cnt["left"] += int(L - ok.sum())
            continue
        lo, hi = max(0, s - a.window), min(T, e + a.window)
        edge = np.r_[lo:s, e:hi]
        edge = edge[seen[edge, m]]
        spread = np.full(M, np.inf)
        for j in range(M):
            k = edge[seen[edge, j]] if j != m else []
            if len(k) >= 3:
                spread[j] = np.std(np.linalg.norm(X[k, j] - X[k, m], axis=1))
        D = [j for j in np.argsort(spread)[:6] if np.isfinite(spread[j])]
        tris = sorted(combinations(D, 3), key=lambda c: spread[list(c)].sum())
        refs = {c: (reference(m, list(c), s - 1, -1, 4 * a.window), reference(m, list(c), e, 1, 4 * a.window)) for c in tris}
        done = 0
        for t in range(s, e):
            for c in tris:
                b, f = refs[c]
                if (b is None and f is None) or not seen[t, list(c)].all():
                    continue
                fr_ = frame(X[t, list(c)])
                if fr_ is None:
                    continue
                if b is not None and f is not None:
                    w = (t - b[0]) / (f[0] - b[0])
                    q = (1 - w) * b[1] + w * f[1]
                else:
                    q = (b or f)[1]
                Y[t, m] = fr_[0] + fr_[1].T @ q
                how[t, m] = 2
                done += 1
                break
        cnt["rigid"] += done
        cnt["left"] += L - done
    report[nm] = {k: int(v) for k, v in cnt.items()} | dict(dropped_as_jumps=dropped.get(nm, 0))

src = Path(a.csv).read_text().splitlines()
hdr = src[0]
out = [hdr]
for t in range(T):
    pre = src[1 + t].split(",")[:2]
    vals = ["" if not np.isfinite(v) else f"{v:.6f}" for v in Y[t].ravel()]
    out.append(",".join(pre + vals))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(out) + "\n")
tot = {k: sum(r[k] for r in report.values()) for k in ("missing", "spline", "rigid", "learned", "left")}
how[dropmask & (how == 0)] = 3
np.savez_compressed(Path(a.out).with_suffix(".npz"), how=how, names=np.array(names))
Path(a.out).with_suffix(".json").write_text(json.dumps(dict(total=tot, per_marker=report), indent=1))
print("total:", tot)
print({k: (v["missing"], v["spline"] + v["rigid"] + v["learned"]) for k, v in report.items() if v["missing"]})
