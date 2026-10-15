#!/usr/bin/env python3
"""Label a whole raw trial with the trained SOMA labeller and score it.

Per frame: the unlabelled cloud -> one-to-one labels. Per piece (raw column
cut at gaps / >5mm jumps): summed-confidence vote, clashes resolved.

Label-free metrics are always printed (how decisive the piece votes are,
what is left unlabelled, where SOMA and Vicon disagree). With ``--truth`` the
labels are also compared to a labelled export; ``--truth-name`` says how much
to trust it -- a partly trusted file prints as "agreement", not "correct".

    python scripts/soma/eval_trial.py --model results/soma/labeler.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.model import SomaLabeler, hard_assign
from vicon2mano.strategies.soma.realdata import (clouds, pieces, raw_identities, relabelled_markers,
                                                 vote_pieces, write_wide_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "results/soma/labeler.pt"))
    ap.add_argument("--raw", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly.csv"))
    ap.add_argument("--truth", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly_manuallylabelled_filled.csv"),
                    help="labelled export to compare with; '' for none")
    ap.add_argument("--truth-name", default="manual",
                    help="how to call the truth; anything but 'manual' reports agreement, not correctness")
    ap.add_argument("--out", default=None, help="npz of per-frame and voted labels per raw slot")
    ap.add_argument("--csv-out", default=None, help="relabelled trial as a wide CSV (voted labels)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location=dev)
    labels, N = ck["labels"], ck["n_points"]
    M = len(labels)
    net = SomaLabeler(M, d_model=ck["d_model"], n_layers=ck["layers"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()

    raw, nr = load_csv(args.raw)[:2]
    nr = [n.split(":")[-1] for n in nr]
    T, C = raw.shape[:2]
    if args.truth:
        man, nm = load_csv(args.truth)[:2]
        nm = [n.split(":")[-1] for n in nm]
        ident = raw_identities(raw, man[:, [nm.index(l) for l in labels]])
    else:
        ident = np.where(np.isfinite(raw).all(-1), -1, -2)
    n_pres = np.isfinite(raw).all(-1).sum(1)
    frames = np.flatnonzero((n_pres > 0) & (n_pres <= N))
    pts, valid, _, col = clouds(raw, np.where(np.isfinite(raw).all(-1), 0, -2), frames, N)

    lab = np.full((T, C), -1); conf = np.zeros((T, C))
    with torch.no_grad():
        for s in range(0, len(frames), 2048):
            lp = net(torch.tensor(pts[s:s + 2048], device=dev),
                     torch.tensor(valid[s:s + 2048], device=dev)).float().cpu().numpy()
            for b in range(lp.shape[0]):
                k = s + b; nv = int(valid[k].sum()); t = frames[k]
                h = hard_assign(lp[b], nv)
                c = col[k, :nv]
                lab[t, c] = h
                conf[t, c] = np.where(h >= 0, np.exp(lp[b, np.arange(nv), np.maximum(h, 0)]), 0)
    pid = pieces(raw)
    voted = vote_pieces(lab, conf, pid, M)

    vicon = np.where(np.isfinite(raw).all(-1), np.array([labels.index(n) if n in labels else -1 for n in nr])[None], -1)
    pres = np.isfinite(raw).all(-1)
    few = (C - n_pres) <= 4
    print(f"{Path(args.raw).name}: {T} frames, {int(pres.sum())} points; "
          f"{100*few.mean():.1f}% of frames have <=4 markers missing; model {Path(args.model).name}")

    # ---- label-free: how decisive are the piece votes, what is left over
    P = pid.max() + 1
    m = pid >= 0
    votes = np.zeros((P, M + 1))
    np.add.at(votes, (pid[m], np.where(lab[m] >= 0, lab[m], M)), conf[m])
    share = votes.max(1) / np.maximum(votes.sum(1), 1e-12)
    sl = share[pid[m]]                                   # per slot, its piece's vote share
    print(f"  piece vote share (per point): >=0.9 {100*(sl >= .9).mean():5.2f}% | 0.5-0.9 "
          f"{100*((sl >= .5) & (sl < .9)).mean():5.2f}% | <0.5 {100*(sl < .5).mean():5.2f}%")
    print(f"  unlabelled after vote: {100*(voted[pres] == -1).mean():.2f}% of points")
    dv = pres & (voted != vicon) & (voted >= 0)
    print(f"  SOMA label differs from Vicon's: {100*dv.sum()/pres.sum():.2f}% of points; by Vicon column: " +
          ", ".join(f"{nr[c]} {100*dv[:, c].sum()/max(pres[:, c].sum(), 1):.1f}%"
                    for c in np.argsort(-dv.sum(0))[:6] if dv[:, c].any()))

    # ---- against a labelled export, if given
    if args.truth:
        judge = ident >= 0
        word = "correct" if args.truth_name == "manual" else f"agree w/ {args.truth_name}"
        print(f"  vs {Path(args.truth).name}: {int(judge.sum())} judgeable points "
              f"({100*judge.sum()/pres.sum():.1f}% of raw points match a {args.truth_name} position)")
        for name, L in (("raw Vicon labels", vicon), ("SOMA per frame", lab), ("SOMA + piece vote", voted)):
            row = f"    {name:18s}"
            for sub, msk in (("all", judge), ("<=4 missing", judge & few[:, None])):
                row += f" | {sub}: {100*(L == ident)[msk].mean():6.2f}% {word}, {100*(L == -1)[msk].mean():4.2f}% unlabelled"
            print(row)
        d = judge & (voted != vicon) & (voted >= 0)
        print(f"    where SOMA and Vicon differ ({int(d.sum())} points): {args.truth_name} sides with SOMA "
              f"{100*(voted==ident)[d].mean():.1f}%, with Vicon {100*(vicon==ident)[d].mean():.1f}%")
    if args.csv_out:
        X = relabelled_markers(raw, voted, M)
        write_wide_csv(args.csv_out, X, labels)
        print(f"saved {args.csv_out}")
    if args.out:
        np.savez_compressed(args.out, per_frame=lab, voted=voted, conf=conf, labels=np.array(labels))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
