#!/usr/bin/env python3
"""Per-frame quality flags for a SOMA-labelled, surface-fitted trial.

Wrong labels give wrong joint angles, so every fitted frame is graded from
three independent signals; none is trusted alone (on P10, 60% of the wrong
labels still carry confidence >= 0.9, so confidence by itself misses most):

* label  - the labeller's own doubt about a marker in that frame: low
           confidence, per-frame label differing from the piece vote, or left
           unlabelled by the vote.
* fit    - a marker the fitted hand cannot reach: its distance to the fitted
           attachment point far above that marker's own trial median (robust,
           MAD-scaled, so it adapts to each participant).
* angle  - physically implausible output: a joint rotated past ``--max-joint``
           degrees, or a frame-to-frame jump larger than ``--max-step``.

``flag`` is 0 = no signal, 1 = review (one signal), 2 = unreliable (two or
more signals, or fewer than ``--min-present`` of the fitted markers observed:
gaps are missing data, not label doubt, so they only count through that rule). Thresholds are starting
points, not calibrated: check them against reviewed frames before relying on
them (``eval_sample.py`` has the 500-frame reviewed sample).

    python scripts/soma/flag_frames.py \\
        --labels results/x/labels.npz   (eval_trial.py --out) \\
        --csv    results/x/soma.csv     (eval_trial.py --csv-out; the fit's input) \\
        --fit    results/x/fit.npz      (fit_surface_markers.py --npz-out) \\
        --pred   results/x/pred.npz     (fit_surface_markers.py --pred-out) \\
        --out    results/x/flags.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def marker_label_signals(per_frame, voted, conf, label_idx, frames):
    """Per fitted frame and marker: (conf, disagree, unlabelled) of the raw
    column that the vote assigned to that label.

    ``per_frame``/``voted``/``conf`` are (T, C) per raw column; ``label_idx``
    the label index of each fitted marker. A marker with no column in a frame
    (gap) gets conf 0 and is reported through ``present`` instead.
    """
    T = len(frames)
    K = len(label_idx)
    c = np.zeros((T, K))
    dis = np.zeros((T, K), bool)
    present = np.zeros((T, K), bool)
    ar = np.arange(T)
    for k, l in enumerate(label_idx):
        m = voted[frames] == l
        has = m.any(1)
        col = m.argmax(1)
        c[:, k] = np.where(has, conf[frames][ar, col], 0.0)
        dis[:, k] = has & (per_frame[frames][ar, col] != l)
        present[:, k] = has
    return c, dis, present


def robust_excess(res):
    """Per-marker excess over its own median in robust sigmas (MAD * 1.4826)."""
    med = np.nanmedian(res, axis=0)
    mad = np.nanmedian(np.abs(res - med), axis=0) * 1.4826
    mad = np.maximum(mad, 0.5)                      # mm; guards a near-constant marker
    return (res - med) / mad


def grade(low_conf, disagree, excess, max_joint_deg, step_deg,
          n_present, min_present, args):
    label_sig = (low_conf | disagree).any(1)
    fit_sig = np.nan_to_num(excess, nan=0.0).max(1) > args.fit_sigma
    angle_sig = (max_joint_deg > args.max_joint) | (step_deg > args.max_step)
    n_sig = label_sig.astype(int) + fit_sig + angle_sig
    flag = np.where((n_sig >= 2) | (n_present < min_present), 2, n_sig)
    return flag.astype(np.int8), label_sig, fit_sig, angle_sig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.5, help="label confidence below this is doubt")
    ap.add_argument("--fit-sigma", type=float, default=6.0,
                    help="marker residual this many robust sigmas over its median is a fit signal")
    ap.add_argument("--max-joint", type=float, default=150.0,
                    help="deg; axis-angle magnitude of any joint beyond this (a loose plausibility bound)")
    ap.add_argument("--max-step", type=float, default=5.0, help="deg/frame; joint jump beyond this")
    ap.add_argument("--min-present", type=int, default=14, help="fewer fitted markers observed = unreliable")
    args = ap.parse_args()

    from scipy.spatial.transform import Rotation as R

    from vicon2mano.core.loader import load_csv

    z = np.load(args.labels)
    per_frame, voted, conf, labels = z["per_frame"], z["voted"], z["conf"], list(z["labels"])
    p = np.load(args.pred)
    frames, names, pred = p["frames"], [str(n) for n in p["names"]], p["pred_mm"]
    fit = np.load(args.fit)
    hp = fit["hand_pose"][frames].reshape(-1, 15, 3)

    mk, lab = load_csv(args.csv)[:2]
    lab = [n.split(":")[-1] for n in lab]
    obs = mk[frames][:, [lab.index(n) for n in names]]
    res = np.linalg.norm(pred - obs, axis=-1)                       # (T, K) mm, NaN = absent

    label_idx = [labels.index(n) for n in names]
    c, dis, present = marker_label_signals(per_frame, voted, conf, label_idx, frames)
    seen = present & np.isfinite(res)
    rot = np.degrees(np.linalg.norm(hp, axis=-1))                  # (T, 15) joint magnitudes
    max_joint = rot.max(1)
    d = np.zeros(len(frames))
    if len(frames) > 1:                                            # jump to the previous fitted frame
        rel = R.from_rotvec(hp[1:].reshape(-1, 3)).inv() * R.from_rotvec(hp[:-1].reshape(-1, 3))
        d[1:] = np.degrees(rel.magnitude()).reshape(-1, 15).max(1)
        d[1:] /= np.maximum(np.diff(frames), 1)                    # per frame, even when strided
    excess = robust_excess(np.where(seen, res, np.nan))

    flag, ls, fs, as_ = grade(seen & (c < args.conf), dis & seen, excess,
                              max_joint, d, seen.sum(1), args.min_present, args)
    import pandas as pd
    df = pd.DataFrame({
        "frame": frames,
        "flag": flag,
        "label_signal": ls, "fit_signal": fs, "angle_signal": as_,
        "n_markers": seen.sum(1),
        "min_conf": np.where(seen, c, 1.0).min(1),
        "n_disagree": (dis & seen).sum(1),
        "max_resid_mm": np.nanmax(np.where(seen, res, np.nan), axis=1),
        "max_resid_sigma": np.nan_to_num(excess, nan=0.0).max(1),
        "max_joint_deg": max_joint,
        "max_step_deg": d,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n = len(df)
    print(f"{n} frames: ok {100*(flag == 0).mean():.1f}% | review {100*(flag == 1).mean():.1f}% "
          f"| unreliable {100*(flag == 2).mean():.1f}%   (label {100*ls.mean():.1f}% "
          f"fit {100*fs.mean():.1f}% angle {100*as_.mean():.1f}%)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
