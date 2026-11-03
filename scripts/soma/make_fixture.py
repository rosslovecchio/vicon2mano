#!/usr/bin/env python3
"""Cut a small real-data example out of one participant's recordings, for the tutorial. NOT FOR RELEASE by itself:
the folder is local, and sharing it needs release authorisation for that participant's data.

Writes, under --out: the static (as the pipeline used it), a movement excerpt (raw Vicon labels) containing reviewed
frames, a second excerpt from another trial to train the filler on, the reviewed labels of the movement excerpt, the
accepted static's params, a fixed artificial-gap mask, and MANIFEST.json (sources, frame ranges, sha256 of every file).

    python scripts/soma/make_fixture.py --participant P3 --trial "Trial2 HOI" --first 4000 --frames 6000 \\
        --train-trial "Trial2 Hands only" --out results/soma/fixture_P3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma")]
import static_pose_check as spc  # noqa: E402
from vicon2mano.core import dataset as ds  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.strategies.soma.realdata import write_wide_csv  # noqa: E402

STATICS = ds.DATA_ROOT / "soma_transfer/static_final"


def excerpt(trial: str, p: str, first: int, n: int, out: Path) -> dict:
    """Write frames [first, first + n) of ``p``'s raw ``trial`` to ``out``; return its source record."""
    src = ds.find_trial_csv(p, trial)[0]
    X, names = load_csv(str(src))[:2]
    write_wide_csv(out, X[first:first + n], [m.split(":")[-1] for m in names])
    return dict(source=str(src), first=first, frames=int(min(n, len(X) - first)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--trial", required=True, help="movement trial (Session name in manual_frames.csv)")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--train-trial", required=True, help="another trial: its excerpt trains the filler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = a.participant, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(spc.sources(p)[p][0], out / "static.csv")
    shutil.copy(STATICS / p / f"{p}_static_final_params.npz", out / "static_params.npz")
    src = dict(static=str(spc.sources(p)[p][0]), static_params=str(STATICS / p / f"{p}_static_final_params.npz"),
               movement=excerpt(a.trial, p, a.first, a.frames, out / "movement.csv"),
               train=excerpt(a.train_trial, p, 0, a.frames, out / "train.csv"))
    ann = pd.read_csv(REPO / "results/shared/frame_sample/manual_annotations.csv")
    ann = ann[(ann.participant == p) & (ann.trial_id == a.trial) & ann.frame.between(a.first, a.first + a.frames - 1)]
    ann.assign(frame=ann.frame - a.first).to_csv(out / "reviewed_labels.csv", index=False)
    X, names = load_csv(str(out / "movement.csv"))[:2]
    ok = np.isfinite(X).all(-1)
    rng, mask = np.random.default_rng(a.seed), []
    for m in ("Index3", "Middle3", "Ring3", "Thumb3"):                    # one 1 s and one 5 s cut per marker, observed
        for n in (200, 1000):
            starts = [s for s in range(0, len(X) - n) if ok[s:s + n, names.index(m)].all()]
            if starts:
                s = int(rng.choice(starts))
                mask.append(dict(marker=m, first=s, frames=n))
    (out / "gap_mask.json").write_text(json.dumps(mask, indent=1))
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(dict(participant=p, release="NOT AUTHORISED", sources=src,
                                                       reviewed_frames=sorted(int(f) for f in ann.frame.unique()),
                                                       sha256=files), indent=1))
    print(f"{out}: {len(files)} files, reviewed frames {sorted(set(ann.frame - a.first))}, {len(mask)} gap cuts")


if __name__ == "__main__":
    main()
