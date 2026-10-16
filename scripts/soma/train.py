#!/usr/bin/env python3
"""Train the strategySOMA labeller on synthetic clouds only.

Every --eval-every steps it reports accuracy on held-out synthetic clouds and,
for monitoring only, on real raw frames (<= 4 markers missing) against a
labelled export -- by default P10/Trial2 and its manual relabel. For a
partly trusted export, --monitor-name makes the log say "agree" instead of
"REAL". Real data is never trained on and never used to pick the checkpoint
-- the saved model is simply the last one.

    python scripts/soma/train.py --steps 30000 --out results/soma/models/labeler_P10.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT
from vicon2mano.core.fitter import FitConfig, MANOFitter
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.library import MarkerLibrary
from vicon2mano.strategies.soma.model import SomaLabeler, assignment_nll, hard_assign
from vicon2mano.strategies.soma.realdata import clouds, raw_identities
from vicon2mano.strategies.soma.synth import SynthConfig, SynthGenerator


def accuracy(net, pts, valid, label, bs=512):
    """(hard one-to-one acc, best-label argmax acc) over real points (label >= 0)."""
    net.eval(); hard, arg, n = 0, 0, 0
    with torch.no_grad():
        for s in range(0, len(pts), bs):
            lp = net(pts[s:s + bs], valid[s:s + bs]).float().cpu().numpy()
            for b in range(lp.shape[0]):
                v = valid[s + b].cpu().numpy(); nv = int(v.sum()); t = label[s + b].cpu().numpy()[:nv]
                M = lp.shape[2] - 1
                h = hard_assign(lp[b], nv); a = lp[b, :nv, :M].argmax(1)
                real = t >= 0
                hard += (h[real] == t[real]).sum(); arg += (a[real] == t[real]).sum(); n += real.sum()
    net.train()
    return hard / n, arg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    ap.add_argument("--mano-dir", default=str(REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--real-frames", type=int, default=2000)
    ap.add_argument("--monitor-raw", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly.csv"))
    ap.add_argument("--monitor-truth", default=str(Path(DATA_ROOT) / "P10/Trial2_handsonly_manuallylabelled_filled.csv"))
    ap.add_argument("--monitor-name", default="REAL", help="log label for the monitor accuracy")
    ap.add_argument("--out", default=str(REPO / "results/soma/models/labeler_P10.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = MarkerLibrary.load(args.library)
    mano = MANOFitter(FitConfig(mano_model_path=args.mano_dir, hand_side="right", use_pca=False))._model
    gen = SynthGenerator(lib, mano, SynthConfig(), device=dev, seed=args.seed)
    M, N = gen.M, gen.n_points
    net = SomaLabeler(M, d_model=args.d_model, n_layers=args.layers).to(dev)
    print(f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params on {dev}; {M} labels, {N} points max")

    # held-out synthetic set (different generator seed)
    vgen = SynthGenerator(lib, mano, SynthConfig(), device=dev, seed=args.seed + 999)
    sv = vgen.sample(4000)
    # real monitoring set: raw frames, identities from the labelled export
    man, nm = load_csv(args.monitor_truth)[:2]
    raw, nr = load_csv(args.monitor_raw)[:2]
    nm = [n.split(":")[-1] for n in nm]
    ident = raw_identities(raw, man[:, [nm.index(l) for l in lib.labels]])
    n_pres = (ident > -2).sum(1)
    ok = np.flatnonzero((n_pres >= M - 4) & (n_pres <= N))
    rf = np.random.default_rng(0).choice(ok, min(args.real_frames, len(ok)), replace=False)
    rp, rv, rl, _ = clouds(raw, ident, rf, N)
    rset = [torch.tensor(a, device=dev) for a in (rp, rv, rl)]
    print(f"monitor ({args.monitor_name}): {Path(args.monitor_raw).name}, {len(rf)} frames with <=4 missing "
          f"(of {len(ok)} eligible)")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))
    log = []
    t0 = time.time(); run = 0.0
    for step in range(1, args.steps + 1):
        pts, valid, lab = gen.sample(args.batch)
        loss = assignment_nll(net(pts, valid), lab, valid)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            sh, sa = accuracy(net, *sv)
            rh, ra = accuracy(net, *rset)
            rec = dict(step=step, loss=run, synth_hard=sh, synth_arg=sa, real_hard=rh, real_arg=ra,
                       min=(time.time() - t0) / 60)
            log.append(rec)
            print(f"step {step:6d}  loss {run:.4f}  synth {100*sh:5.2f}% (argmax {100*sa:5.2f}%)  "
                  f"{args.monitor_name} {100*rh:5.2f}% (argmax {100*ra:5.2f}%)  {rec['min']:.1f} min", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=net.state_dict(), labels=lib.labels, n_points=N,
                            d_model=args.d_model, layers=args.layers, step=step), args.out)
            Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
