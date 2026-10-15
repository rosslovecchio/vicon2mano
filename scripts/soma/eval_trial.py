#!/usr/bin/env python3
"""Label a whole raw trial with the trained SOMA labeller and score it.

Per frame: the unlabelled cloud -> one-to-one labels. Per piece (raw column
cut at gaps / >5mm jumps): summed-confidence vote, clashes resolved. Scored
against a trusted labelled export where one exists, side by side with the
raw Vicon labels.

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
from vicon2mano.strategies.soma.realdata import clouds, pieces, raw_identities, vote_pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "results/soma/labeler.pt"))
    ap.add_argument("--raw", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly.csv"))
    ap.add_argument("--truth", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly_manuallylabelled_filled.csv"))
    ap.add_argument("--out", default=None, help="npz of per-frame and voted labels per raw slot")
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
        assert [n.split(":")[-1] for n in nm] == labels
        ident = raw_identities(raw, man)
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
    judge = ident >= 0
    few = (C - n_pres) <= 4
    print(f"{T} frames, {int(judge.sum())} judgeable slots; {100*few.mean():.1f}% of frames have <=4 missing")
    for name, L in (("raw Vicon labels", vicon), ("SOMA per frame", lab), ("SOMA + piece vote", voted)):
        row = f"  {name:18s}"
        for sub, msk in (("all", judge), ("<=4 missing", judge & few[:, None])):
            ok = (L == ident)[msk]; un = (L == -1)[msk]
            row += f" | {sub}: {100*ok.mean():6.2f}% correct, {100*un.mean():4.2f}% unlabelled"
        print(row)
    # where do the labeller and Vicon disagree, and who is right?
    d = judge & (voted != vicon) & (voted >= 0)
    print(f"  voted labels that differ from Vicon: {int(d.sum())} slots -> SOMA right {100*(voted==ident)[d].mean():.1f}%, "
          f"Vicon right {100*(vicon==ident)[d].mean():.1f}%")
    if args.out:
        np.savez_compressed(args.out, per_frame=lab, voted=voted, conf=conf, labels=np.array(labels))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
