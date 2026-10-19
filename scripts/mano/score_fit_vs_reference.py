#!/usr/bin/env python3
"""Score a fit against a reference fit (e.g. the manual-label fit), split by
whether Vicon's labels were right or wrong *for that finger* in each frame.

Angle error is the geodesic rotation difference (deg) between the two fits for
each finger joint (the 3 joints MCP/PIP/DIP of the finger, per frame). A
joint-frame belongs to a finger-frame that is

  wrong  Vicon's label disagreed with the manual export on at least one of that
         finger's markers (a raw point that sits on a manual-labelled position,
         < 1 mm, but carries a different Vicon label), counting both the finger
         the point was given to and the finger it truly belongs to;
  right  at least one of the finger's markers was judgeable (matched to a manual
         position) and none was wrong;
  other  no judgeable marker of that finger (excluded from the split).

Gap-filled manual positions never match a raw point, so a frame is only called
right/wrong where the manual reference is a real observation.

    python scripts/mano/score_fit_vs_reference.py \\
        --reference results/mano/P10_Trial2_handsonly/mano_fit_right_calibrated.npz \\
        --raw  <P10/Trial2_handsonly.csv> --manual <..._manuallylabelled_filled.csv> \\
        --fit pin=results/perf/palm1/pin/fit.npz --fit nopin=results/perf/palm1/nopin/fit.npz
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LABELS = ["Forearm1", "Forearm2", "Forearm3", "Forearm4", "Palm1", "Palm2", "Palm3", "Thumb1", "Thumb2",
          "Thumb3", "Index1", "Index2", "Index3", "Middle1", "Middle2", "Middle3", "Ring1", "Ring2", "Ring3",
          "Pinky1", "Pinky2", "Pinky3"]
FINGERS = ["Index", "Middle", "Pinky", "Ring", "Thumb"]            # smplx hand_pose joint order, 3 joints each


def finger_of(label: str) -> str | None:
    for f in FINGERS:
        if label.startswith(f):
            return f
    return None


def finger_frame_class(raw, manual, raw_names, man_names):
    """{finger: 'right'|'wrong'|'other' per frame} from Vicon-vs-manual labels."""
    from vicon2mano.strategies.soma.realdata import raw_identities
    keep = [i for i, n in enumerate(raw_names) if n in LABELS]
    raw = raw[:, keep]
    vic = np.array([LABELS.index(raw_names[i]) for i in keep])               # Vicon's label per column
    ident = raw_identities(raw, manual[:, [man_names.index(l) for l in LABELS]])
    judge = ident >= 0
    wrong = judge & (ident != vic[None, :])
    right = judge & (ident == vic[None, :])
    fvic = np.array([finger_of(LABELS[v]) for v in vic], dtype=object)
    out = {}
    T = raw.shape[0]
    lut = np.array([finger_of(l) for l in LABELS] + [None], dtype=object)    # label index -> finger (last = none)
    true_f = lut[np.where(ident >= 0, ident, len(LABELS))]
    for f in FINGERS:
        w = np.zeros(T, bool)
        r = np.zeros(T, bool)
        w |= (wrong & (fvic[None, :] == f)).any(1)                           # a marker Vicon gave to f was wrong
        w |= (wrong & (true_f == f)).any(1)                                  # ... or one that truly is f's
        r |= (right & (fvic[None, :] == f)).any(1)
        cls = np.full(T, "other", dtype=object)
        cls[r & ~w] = "right"
        cls[w] = "wrong"
        out[f] = cls
    return out


def load_pose(path, model_dir=None):
    z = np.load(path)
    hp = z["hand_pose"]
    fr = z["frames_fitted"] if "frames_fitted" in z.files else np.arange(len(hp))
    if hp.shape[1] != 45:                                                    # PCA coefficients -> 45-dim
        import torch
        import smplx
        m = smplx.create(str(Path(model_dir) / "MANO_RIGHT.pkl"), model_type="mano", is_rhand=True,
                         use_pca=True, num_pca_comps=hp.shape[1], flat_hand_mean=True, batch_size=1)
        hp = (torch.tensor(hp) @ m.hand_components + m.hand_mean).numpy()
    return hp.reshape(len(hp), 15, 3), fr


def geodesic_deg(a, b):
    from scipy.spatial.transform import Rotation as R
    ra, rb = R.from_rotvec(a.reshape(-1, 3)), R.from_rotvec(b.reshape(-1, 3))
    return np.degrees((ra.inv() * rb).magnitude()).reshape(a.shape[:-1])


def fingers_residual(log: Path | None):
    if log is None or not Path(log).exists():
        return float("nan")
    m = re.findall(r"FINGERS\s+([\d.]+) mm", Path(log).read_text(encoding="utf-8", errors="replace"))
    return float(m[-1]) if m else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="the manual-label fit .npz")
    ap.add_argument("--raw", required=True, help="raw Vicon csv of the trial")
    ap.add_argument("--manual", required=True, help="manually labelled export of the same trial")
    ap.add_argument("--fit", action="append", required=True, metavar="NAME=FIT.npz[,LOG]",
                    help="fit to score; optional ',fit.log' for its finger residual")
    ap.add_argument("--mano-dir", default=None, help="only needed if the reference stores PCA poses")
    args = ap.parse_args()

    from vicon2mano.core.dataset import DATA_ROOT
    from vicon2mano.core.loader import load_csv
    raw, rn = load_csv(args.raw)[:2]
    man, mn = load_csv(args.manual)[:2]
    rn, mn = [n.split(":")[-1] for n in rn], [n.split(":")[-1] for n in mn]
    cls = finger_frame_class(raw, man, rn, mn)
    ref, rfr = load_pose(args.reference, args.mano_dir or str(DATA_ROOT / "mano_v1_2" / "models"))
    print("finger-frames by Vicon's labels (right / wrong / other): " + ", ".join(
        f"{f} {int((cls[f] == 'right').sum())}/{int((cls[f] == 'wrong').sum())}/{int((cls[f] == 'other').sum())}"
        for f in FINGERS))
    print(f"\n{'fit':10s} {'split':6s} {'joint-frames':>12s} {'mean err':>9s} {'p95':>7s} {'>10 deg':>8s}   per finger mean err (deg)")
    for spec in args.fit:
        name, rest = spec.split("=", 1)
        path, _, log = rest.partition(",")
        hp, fr = load_pose(path, args.mano_dir or str(DATA_ROOT / "mano_v1_2" / "models"))
        common = np.intersect1d(fr, rfr)
        e = geodesic_deg(hp[np.searchsorted(fr, common)], ref[np.searchsorted(rfr, common)])    # (n, 15)
        for split in ("right", "wrong", "all"):
            errs, per = [], {}
            for fi, f in enumerate(FINGERS):
                m = (cls[f][common] != "other") if split == "all" else (cls[f][common] == split)
                v = e[m][:, 3 * fi:3 * fi + 3].ravel()
                errs.append(v)
                per[f] = v.mean() if len(v) else float("nan")
            v = np.concatenate(errs)
            print(f"{name:10s} {split:6s} {len(v):12d} {v.mean():8.2f}d {np.percentile(v, 95):6.1f}d {100 * (v > 10).mean():7.1f}%   "
                  + "  ".join(f"{f[:3]} {per[f]:5.2f}" for f in FINGERS))
        print(f"{name:10s} finger residual (fit.log, against the labels it was fitted to): "
              f"{fingers_residual(Path(log) if log else None):.2f} mm\n")


if __name__ == "__main__":
    main()
