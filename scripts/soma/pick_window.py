#!/usr/bin/env python3
"""Pick the stillest, flattest 200 frames of a mapped trial to stand in for a
missing static (what was done for P1/P2: ``calibrate_static.py --session ...
--window a:b``).

Candidates are windows in which every hand marker is present in at least
``--min-present`` of the frames. Each is scored on

  motion  median over frames of the median marker speed (mm/frame)
  bend    mean over index/middle/ring/pinky of the angle at the X2 marker between
          X1->X2 and X2->X3 (deg; 0 = straight finger), median over frames

and the window with the lowest sum of both after dividing by their medians over
all candidates is chosen. Because calibration then uses *Vicon's labels* in that
window, they are checked against a SOMA model's labels (``--model``, the base
P10 model by default), counting only points SOMA labelled; a low agreement means
the window cannot be trusted and the script exits non-zero.

    python scripts/soma/pick_window.py --participant P5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HAND = [f"{f}{i}" for f in ("Palm", "Thumb", "Index", "Middle", "Ring", "Pinky") for i in (1, 2, 3)]
FINGERS = ("Index", "Middle", "Ring", "Pinky")


def window_stats(X: np.ndarray, names: list[str], length: int, stride: int, min_present: float):
    """[(start, motion_mm_per_frame, bend_deg, min_marker_presence)] over complete-enough windows."""
    ix = {n: names.index(n) for n in HAND}
    H = X[:, [ix[n] for n in HAND]]                                   # (T, 18, 3)
    pres = np.isfinite(H).all(-1)
    step = np.linalg.norm(np.diff(H, axis=0), axis=-1)                # (T-1, 18)
    with np.errstate(all="ignore"):
        speed = np.r_[np.nanmedian(step, axis=1), np.nan]
    bend = np.full(len(X), np.nan)
    ang = []
    for f in FINGERS:
        a, b, c = (X[:, ix[f"{f}{k}"]] for k in (1, 2, 3))
        u, v = b - a, c - b
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    with np.errstate(all="ignore"):
        bend = np.nanmean(np.stack(ang), axis=0)
    out = []
    for s in range(0, len(X) - length + 1, stride):
        p = pres[s:s + length].mean(0)
        if p.min() < min_present:
            continue
        sp, bd = speed[s:s + length], bend[s:s + length]
        if np.isfinite(sp).sum() < length // 2 or np.isfinite(bd).sum() < length // 2:
            continue
        out.append((s, float(np.nanmedian(sp)), float(np.nanmedian(bd)), float(p.min())))
    return out


def choose(stats):
    mo = np.median([s[1] for s in stats])
    be = np.median([s[2] for s in stats])
    score = [s[1] / mo + s[2] / be for s in stats]
    order = np.argsort(score)
    return [(stats[i], score[i]) for i in order]


def label_agreement(raw_window: np.ndarray, names: list[str], model_path: str, device: str):
    """(agreement among labelled points, share left unlabelled) of Vicon's column labels vs SOMA's.

    An unlabelled point is an abstention, not a disagreement: the base model
    declines some markers on some participants (P5 Palm1/Pinky1, P7 Forearm4/
    Thumb1) without ever assigning a conflicting label."""
    from vicon2mano.strategies.soma.infer import label_trial, load_model
    net, labels, N = load_model(model_path, device)
    keep = [i for i, n in enumerate(names) if n in labels]
    raw = raw_window[:, keep]
    col_label = np.array([labels.index(names[i]) for i in keep])
    _, _, voted, _ = label_trial(net, raw, N, device)
    pres = np.isfinite(raw).all(-1)
    lab = pres & (voted >= 0)
    agree = (voted == col_label[None, :]) & lab
    return float(agree.sum() / max(lab.sum(), 1)), float((pres & (voted < 0)).sum() / max(pres.sum(), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", required=True)
    ap.add_argument("--session", default="Trial 1 Hands only")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--min-present", type=float, default=0.95,
                    help="every hand marker must be present in at least this fraction of the window")
    ap.add_argument("--model", default=str(REPO / "results/soma/models/labeler_P10.pt"))
    ap.add_argument("--min-agreement", type=float, default=0.95)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import torch
    from vicon2mano.core.dataset import find_trial_csv
    from vicon2mano.core.loader import load_csv
    path, *_ = find_trial_csv(args.participant, args.session)
    if path is None:
        raise SystemExit(f"manual_frames.csv maps no file to {args.participant}/{args.session}")
    X, names = load_csv(str(path))[:2]
    names = [n.split(":")[-1] for n in names]
    missing = [n for n in HAND if n not in names]
    if missing:
        raise SystemExit(f"{path.name} lacks hand columns {missing}")
    stats = window_stats(X, names, args.length, args.stride, args.min_present)
    if not stats:
        raise SystemExit(f"{args.participant}: no {args.length}-frame window with every hand marker present in "
                         f">= {args.min_present:.0%} of its frames")
    ranked = choose(stats)
    print(f"{args.participant} / {args.session} -> {path.name}: {len(X)} frames, {len(stats)} complete windows")
    print(f"{'window':>14s} {'mm/frame':>9s} {'bend deg':>9s} {'min presence':>13s}  score")
    for (s, mo, be, pm), sc in ranked[:3]:
        print(f"{s:>6d}:{s + args.length:<7d} {mo:9.3f} {be:9.1f} {pm:13.2f}  {sc:.2f}")
    (s, mo, be, pm), _ = ranked[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ag, abst = label_agreement(X[s:s + args.length], names, args.model, dev)
    ok = ag >= args.min_agreement
    print(f"chosen {s}:{s + args.length}  motion {mo:.3f} mm/frame, mean finger bend {be:.1f} deg; "
          f"Vicon labels agree with {Path(args.model).name} on {100 * ag:.1f}% of the points it labelled "
          f"({100 * abst:.1f}% left unlabelled) "
          f"({'ok' if ok else 'BELOW ' + format(args.min_agreement, '.0%') + ' - do not use'})")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(dict(
            participant=args.participant, session=args.session, file=path.name, window=f"{s}:{s + args.length}",
            motion_mm_per_frame=mo, bend_deg=be, label_agreement=ag, unlabelled=abst, ok=ok)), encoding="utf-8")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
