#!/usr/bin/env python3
"""Score SOMA on the manually reviewed 500-frame sample (all participants).

Ground truth: for each reviewed trial, the hand-corrected Nexus export
(``<stem>_manuallylabelled.csv``) **at the frames you reviewed only** -- the
frames listed in ``results/shared/frame_sample/manual_annotations.csv``.
Elsewhere those exports are not checked (and are demonstrably wrong in
places), so nothing outside the reviewed frames is scored.

A raw point's true identity is the label whose corrected position coincides
with it (< 1 mm); a raw point with no corrected counterpart was deleted in
review: a ghost, which should be left unlabelled. Each trial is labelled in
full (the piece vote needs the whole trial), then scored on its reviewed
frames.

    python scripts/soma/eval_sample.py --model results/soma/models/labeler_P10.pt \\
        --model-for P12=results/soma/models/labeler_P12.pt --out results/soma/eval/sample500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core import dataset as ds
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.infer import label_trial, load_model
from vicon2mano.strategies.soma.realdata import raw_identities

SAMPLE = REPO / "results/shared/frame_sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="default model for every participant")
    ap.add_argument("--model-for", action="append", default=[],
                    help="participant-specific model, e.g. P12=results/soma/models/labeler_P12.pt")
    ap.add_argument("--participants", default="", help="comma list to restrict to")
    ap.add_argument("--out", required=True, help="output stem: <out>_points.csv and <out>_summary.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    per = dict(s.split("=", 1) for s in args.model_for)
    nets = {}
    ann = pd.read_csv(SAMPLE / "manual_annotations.csv")
    man_fr = pd.read_csv(SAMPLE / "sampling_manifest.csv")
    cat = {(r.participant, r.trial_id, int(r.frame)): r.category for r in man_fr.itertuples()}
    want = {p for p in args.participants.split(",") if p}

    rows = []
    for (p, trial), g in ann.groupby(["participant", "trial_id"]):
        if want and p not in want:
            continue
        mpath = per.get(p, args.model)
        if mpath not in nets:
            nets[mpath] = load_model(mpath, dev)
        net, labels, N = nets[mpath]
        trial_path, *_ = ds.find_trial_csv(p, trial)
        corr = ds.find_manual_csv(trial_path) if trial_path else None
        if corr is None:
            print(f"[skip] {p}/{trial}: no trial or corrected export"); continue
        raw, nr = load_csv(str(trial_path))[:2]
        nr = [n.split(":")[-1] for n in nr]
        cor, nc = load_csv(str(corr))[:2]
        nc = [n.split(":")[-1] for n in nc]
        if not set(labels) <= set(nr) or not set(labels) <= set(nc):
            print(f"[skip] {p}/{trial}: label set differs from the model's"); continue
        raw = raw[:, [nr.index(l) for l in labels]]
        frames = np.array(sorted(g.frame.unique()), int)
        frames = frames[frames < len(raw)]
        ident = raw_identities(raw, cor[:, [nc.index(l) for l in labels]])
        lab, conf, voted, _ = label_trial(net, raw, N, dev)
        pres = np.isfinite(raw).all(-1)
        for t in frames:
            for c in np.flatnonzero(pres[t]):
                rows.append(dict(participant=p, trial_id=trial, frame=int(t), category=cat.get((p, trial, int(t)), ""),
                                 model=Path(mpath).stem, vicon_label=labels[c],
                                 true_label=labels[ident[t, c]] if ident[t, c] >= 0 else "GHOST",
                                 soma_frame=labels[lab[t, c]] if lab[t, c] >= 0 else "",
                                 soma_voted=labels[voted[t, c]] if voted[t, c] >= 0 else "",
                                 n_missing=int(len(labels) - pres[t].sum())))
        d = pd.DataFrame([r for r in rows if r["participant"] == p and r["trial_id"] == trial])
        real = d[d.true_label != "GHOST"]
        print(f"{p:4s} {trial:20s} {len(frames):3d} fr | Vicon {100*(real.vicon_label == real.true_label).mean():5.1f}%  "
              f"SOMA {100*(real.soma_voted == real.true_label).mean():5.1f}%  ({Path(mpath).stem})", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}_points.csv", index=False)

    def summ(d):
        real = d[d.true_label != "GHOST"]; gh = d[d.true_label == "GHOST"]
        return pd.Series(dict(frames=d[["trial_id", "frame"]].drop_duplicates().shape[0], points=len(real),
                              vicon=100 * (real.vicon_label == real.true_label).mean(),
                              soma_frame=100 * (real.soma_frame == real.true_label).mean(),
                              soma_voted=100 * (real.soma_voted == real.true_label).mean(),
                              soma_unlabelled=100 * (real.soma_voted == "").mean(),
                              ghosts=len(gh), ghosts_left_unlabelled=100 * (gh.soma_voted == "").mean() if len(gh) else np.nan))
    by_p = df.groupby("participant").apply(summ)
    by_c = df.groupby("category").apply(summ)
    df["missing_bin"] = np.where(df.n_missing <= 4, "<=4 missing", ">4 missing")
    by_m = df.groupby("missing_bin").apply(summ)
    tot = summ(df).to_frame("ALL").T
    held = summ(df[df.participant != "P10"]).to_frame("ALL except P10 (held out)").T
    summary = pd.concat([by_p, by_c, by_m, tot, held])
    summary.round(2).to_csv(f"{out}_summary.csv")
    pd.set_option("display.width", 200)
    print("\n", summary.round(2).to_string())
    print(f"\nsaved {out}_points.csv, {out}_summary.csv")


if __name__ == "__main__":
    main()
