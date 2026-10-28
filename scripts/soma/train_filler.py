#!/usr/bin/env python3
"""Train the learned marker filler (``strategies/soma/fill.py``) on synthetic clouds, real labelled recordings, or both.

    python scripts/soma/train_filler.py --library results/soma/libraries/library_P16.npz --out results/soma/models/filler_P16.pt
    python scripts/soma/train_filler.py --csv A.csv B.csv [--library L.npz] --out filler.pt

With both, each batch is half synthetic, half real. Real CSVs are read in the library's marker order (or the first
CSV's); blank values count as unobserved.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vicon2mano.core.dataset import DATA_ROOT  # noqa: E402
from vicon2mano.core.fitter import FitConfig, MANOFitter  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.strategies.soma import fill  # noqa: E402
from vicon2mano.strategies.soma.library import MarkerLibrary  # noqa: E402
from vicon2mano.strategies.soma.synth import SynthGenerator  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--library", help="participant library: synthetic clouds")
ap.add_argument("--csv", nargs="*", default=[], help="labelled recordings: real frames")
ap.add_argument("--mano-dir", default=str(DATA_ROOT / "mano_v1_2" / "models"))
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
srcs, labels = [], None
if a.library:
    lib = MarkerLibrary.load(a.library)
    model = MANOFitter(FitConfig(mano_model_path=a.mano_dir, hand_side="right", use_pca=False))._model
    gen = SynthGenerator(lib, model, device=dev, seed=a.seed)
    srcs.append(lambda B: gen.sample(B, complete=True))
    labels = lib.labels
if a.csv:
    frames = []
    for c in a.csv:
        X, names = load_csv(c)[:2]
        names = [n.split(":")[-1] for n in names]
        labels = labels or names
        frames.append(X[:, [names.index(n) for n in labels]])
    srcs.append(fill.frame_sampler(np.concatenate(frames), dev))
sample = srcs[0] if len(srcs) == 1 else lambda B: tuple(torch.cat(z) for z in zip(*(s(B // 2) for s in srcs)))
f = fill.train_filler(sample, list(labels), a.steps, log=lambda s: print(s, flush=True))
fill.save(f, a.out)
print("saved", a.out)
