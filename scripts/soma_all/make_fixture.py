#!/usr/bin/env python3
"""Cut a small real-data example out of one participant's recordings, for the tutorial. NOT FOR RELEASE by itself:
the folder is local, and sharing it needs release authorisation for that participant's data.

Writes, under --out: the static (as the pipeline used it), a movement excerpt (raw Vicon labels) containing reviewed
frames, a second excerpt from another trial to train the filler on, the reviewed labels of the movement excerpt, the
accepted static's params, a fixed artificial-gap mask, the frame-ID maps, the rigid-fill regression reference
(expected.json) and MANIFEST.json (sources, frame ranges, sha256 of every file). tests/soma/test_fixture.py checks it.

Frame IDs: an excerpt's ``_Frame`` column is the source export's Vicon frame number. frame_map_<excerpt>.csv lists,
per excerpt row, ``row`` (0-based in the excerpt), ``source_row`` (0-based in the source, the index
results/shared/frame_sample/manual_annotations.csv uses) and ``source_frame`` (Vicon frame number). reviewed_labels.csv
keeps the annotation's ``source_row`` and adds ``row``.

    python scripts/soma_all/make_fixture.py --participant P3 --trial "Trial2 HOI" --first 4000 --frames 6000 \\
        --train-trial "Trial2 Hands only" --out results/soma/fixture_P3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma_all")]
from vicon2mano.core import dataset as ds  # noqa: E402
from vicon2mano.core.loader import csv_frame_numbers, load_csv  # noqa: E402
from vicon2mano.strategies.soma.realdata import write_wide_csv  # noqa: E402

STATICS = ds.DATA_ROOT / "soma_transfer/static_final"


def excerpt(trial: str, p: str, first: int, n: int, out: Path) -> dict:
    """Write rows [first, first + n) of ``p``'s raw ``trial`` to ``out`` (Vicon frame numbers kept) and its frame map."""
    src = ds.find_trial_csv(p, trial)[0]
    X, names = load_csv(str(src))[:2]
    rows = np.arange(first, min(first + n, len(X)))
    frames = csv_frame_numbers(str(src))[rows]
    write_wide_csv(out, X[rows], [m.split(":")[-1] for m in names], frames=frames)
    pd.DataFrame(dict(row=np.arange(len(rows)), source_row=rows, source_frame=frames)).to_csv(
        out.with_name(f"frame_map_{out.stem}.csv"), index=False)
    return dict(source=str(src), trial=trial, first_row=int(first), rows=int(len(rows)))


def rigid_check(fixture: Path, work: Path) -> dict:
    """Cut the gap mask out of movement.csv, refill with gap_fill.py's rigid fill, and score against the cut values."""
    X, names = load_csv(str(fixture / "movement.csv"))[:2]
    cut = X.copy()
    mask = json.loads((fixture / "gap_mask.json").read_text())
    for c in mask:
        cut[c["first"]:c["first"] + c["frames"], names.index(c["marker"])] = np.nan
    work.mkdir(parents=True, exist_ok=True)
    write_wide_csv(work / "cut.csv", cut, names, frames=csv_frame_numbers(str(fixture / "movement.csv")))
    subprocess.run([sys.executable, "-W", "ignore", str(REPO / "scripts/soma_all/gap_fill.py"), "--csv", str(work / "cut.csv"),
                    "--out", str(work / "filled.csv")], check=True, capture_output=True)
    F = load_csv(str(work / "filled.csv"))[0]
    res = {}
    for c in mask:
        s, i = slice(c["first"], c["first"] + c["frames"]), names.index(c["marker"])
        e = np.linalg.norm(F[s, i] - X[s, i], axis=-1)
        res[f"{c['marker']}@{c['first']}+{c['frames']}"] = dict(
            filled=int(np.isfinite(e).sum()), median=round(float(np.nanmedian(e)), 3) if np.isfinite(e).any() else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--trial", required=True, help="movement trial (Session name in manual_frames.csv)")
    ap.add_argument("--first", type=int, required=True, help="first source row (0-based)")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--train-trial", required=True, help="another trial: its excerpt trains the filler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import static_pose_check as spc
    p, out = a.participant, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    static = spc.sources(p)[p][0]
    shutil.copy(static, out / "static.csv")
    shutil.copy(STATICS / p / f"{p}_static_final_params.npz", out / "static_params.npz")
    src = dict(static=str(static), static_params=str(STATICS / p / f"{p}_static_final_params.npz"),
               movement=excerpt(a.trial, p, a.first, a.frames, out / "movement.csv"),
               train=excerpt(a.train_trial, p, 0, a.frames, out / "train.csv"))
    ann = pd.read_csv(REPO / "results/shared/frame_sample/manual_annotations.csv")
    ann = ann[(ann.participant == p) & (ann.trial_id == a.trial) & ann.frame.between(a.first, a.first + a.frames - 1)]
    ann.rename(columns={"frame": "source_row"}).assign(row=ann.frame - a.first).to_csv(out / "reviewed_labels.csv", index=False)
    X, names = load_csv(str(out / "movement.csv"))[:2]
    ok = np.isfinite(X).all(-1)
    rng, mask = np.random.default_rng(a.seed), []
    for m in ("Index3", "Middle3", "Ring3", "Thumb3"):                    # one 1 s and one 5 s cut per marker, observed
        for n in (200, 1000):
            starts = [s for s in range(0, len(X) - n) if ok[s:s + n, names.index(m)].all()]
            if starts:
                mask.append(dict(marker=m, first=int(rng.choice(starts)), frames=n))
    (out / "gap_mask.json").write_text(json.dumps(mask, indent=1))
    (out / "expected.json").write_text(json.dumps(dict(rigid_fill_mm=rigid_check(out, out / "_work"), tolerance_mm=0.01), indent=1))
    shutil.rmtree(out / "_work")
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(dict(participant=p, release="NOT AUTHORISED", sources=src,
                                                       reviewed_source_rows=sorted(int(f) for f in ann.frame.unique()),
                                                       sha256=files), indent=1))
    print(f"{out}: {len(files)} files, reviewed rows {sorted(set(ann.frame - a.first))}, {len(mask)} gap cuts")


if __name__ == "__main__":
    main()
