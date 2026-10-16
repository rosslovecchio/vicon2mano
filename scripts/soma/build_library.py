#!/usr/bin/env python3
"""Build the strategySOMA marker library from a trusted labelled trial.

Default inputs are P10/Trial2 Hands only: the calibrated surface fit and the
manually labelled CSV. A manual position counts as a real observation only
where it coincides (<1mm) with a point in the raw Vicon export -- elsewhere it
was gap-filled and is not copied (the manual fill folds the pinky to 180 deg
around frames 12600-13300, for one).

    python scripts/soma/build_library.py --out results/soma/libraries/library_P10_Trial2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT
from vicon2mano.core.fitter import FitConfig, MANOFitter
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.library import build_library


def main():
    ap = argparse.ArgumentParser()
    p10 = Path(DATA_ROOT) / "P10"
    ap.add_argument("--fit", default=str(REPO / "results/mano/P10_Trial2_handsonly/mano_fit_right_calibrated.npz"))
    ap.add_argument("--labelled", default=str(p10 / "Trial2_handsonly_manuallylabelled_filled.csv"))
    ap.add_argument("--raw", default=str(p10 / "Trial2_handsonly.csv"))
    ap.add_argument("--mano-dir", default=str(REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--min-real", type=int, default=20,
                    help="library frames need at least this many real (unfilled) markers")
    ap.add_argument("--out", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    args = ap.parse_args()

    man, nm = load_csv(args.labelled)[:2]
    raw, nr = load_csv(args.raw)[:2]
    labels = [n.split(":")[-1] for n in nm]
    nr = [n.split(":")[-1] for n in nr]
    raw_l = raw[:, [nr.index(n) for n in labels]]
    # real = the manual position is one of the raw points this frame (any column)
    d = np.linalg.norm(man[:, :, None] - raw[:, None], axis=-1)
    real = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2) < 1.0
    z = np.load(args.fit)
    fitted = np.zeros(len(man), bool); fitted[z["frames_fitted"]] = True
    frames = np.flatnonzero(fitted & (real.sum(1) >= args.min_real))
    print(f"{len(frames)} library frames (of {len(man)}), real markers/frame "
          f"median {np.median(real[frames].sum(1)):.0f}")

    fitter = MANOFitter(FitConfig(mano_model_path=args.mano_dir, hand_side="right", use_pca=False))
    lib = build_library(fitter._model, {k: z[k] for k in ("global_orient", "hand_pose", "transl", "betas")},
                        man, real, labels, np.isfinite(raw_l).all(-1), frames=frames,
                        source=f"{Path(args.labelled).name} + {Path(args.fit).name}")
    ok = np.isfinite(lib.residual).all(-1)
    mag = np.linalg.norm(np.nan_to_num(lib.residual), axis=-1)
    print(f"{'marker':9s} {'face':>5s} {'kept':>6s} {'|res| med':>9s} {'p95':>6s} {'raw miss':>8s}")
    for m, n in enumerate(labels):
        v = mag[ok[:, m], m]
        print(f"{n:9s} {lib.face_idx[m]:5d} {100*ok[:, m].mean():5.1f}% {np.median(v):9.1f} "
              f"{np.percentile(v, 95):6.1f} {100*lib.miss_rate[m]:7.1f}%")
    lib.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
