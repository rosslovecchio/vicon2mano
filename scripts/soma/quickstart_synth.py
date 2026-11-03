#!/usr/bin/env python3
"""Synthetic quick-start: a made-up participant generated from MANO alone, then gap-filled and fitted.

SYNTHETIC DATA, NOT A RECORDING. It checks that the installation, file formats and commands work end to end, and
that the steps recover a known answer; it says nothing about accuracy on real Vicon recordings. No recorded data is
read: the hand is MANO's mean shape (scale 1) with markers on the released pins
(configs/surface_fit/static_final_pins.json), 3 mm out along the normal; the movement is smooth random finger poses
inside the joint limits with a slow wrist motion, 0.5 mm noise, and fixed gaps. MANO itself is not included: it is
read from core.dataset.MANO_DIR (obtain it under its licence).

    python scripts/soma/quickstart_synth.py --out results/soma/quickstart          # generate + check
Writes static.csv, static_params.npz, movement.csv (with gaps), truth.npz, then gap_fill.py -> filled.csv,
then a filler trained on movement.csv itself -> filled_learned.csv, movement_fit.py on the learned fill -> fit/, and
check.json; exits 1 if a learned-fill or fit result is outside the tolerances. The rigid fill's error on the long Ring3
gap is the intended failure: rigid filling degrades on long finger gaps, which is why Step 3 trains a filler.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma"), str(REPO / "scripts/mano")]
import fit_surface_markers as fsm  # noqa: E402
import static_search as ss  # noqa: E402
from vicon2mano.strategies.soma.realdata import write_wide_csv  # noqa: E402

GAPS = {"Middle2": (100, 108), "Index3": (400, 900), "Thumb3": (1200, 1350), "Ring3": (1500, 1900)}   # frames


def markers(J, att: torch.Tensor, go, hp, tr, standoff=3.0) -> np.ndarray:
    """Marker positions (T, M, 3) mm: attached vertices of the mean-shape MANO, ``standoff`` mm along the normal."""
    sm = fsm.ScaledModel(J.model, J.dev)
    t = lambda x: torch.tensor(np.asarray(x, np.float32), device=J.dev)
    with torch.no_grad():
        V = sm(global_orient=t(go), hand_pose=t(hp), transl=t(tr), betas=torch.zeros(len(go), 10, device=J.dev),
               return_verts=True).vertices
        F = torch.tensor(np.asarray(J.model.faces, np.int64), device=J.dev)
        fn = torch.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]], dim=-1)
        N = torch.zeros_like(V).index_add_(1, F[:, 0], fn).index_add_(1, F[:, 1], fn).index_add_(1, F[:, 2], fn)
        N = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return ((V[:, att] + standoff / 1000 * N[:, att]) * 1000).cpu().numpy()


def poses(J, T: int, rng, every: int = 250, frac: float = 0.4) -> np.ndarray:
    """Smooth random hand poses (T, 45): keyframes inside ``frac`` of the joint limits, no twist or PIP/DIP abduction."""
    lo, hi = J.lo.copy(), J.hi.copy()                                    # (15, 3) deg in the joint-axis basis
    lo[:12, 2] = hi[:12, 2] = 0
    hinge = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] in "23"]
    lo[hinge, 1] = hi[hinge, 1] = 0
    keys = rng.uniform(np.minimum(lo, 0), np.maximum(hi, 0), (T // every + 2, 15, 3)) * frac
    keys[0] = 0
    s = np.arange(T) / every
    i, w = s.astype(int), (1 - np.cos(np.pi * (s % 1))) / 2
    c = np.radians(keys[i] * (1 - w)[:, None, None] + keys[i + 1] * w[:, None, None])
    return np.einsum("tjc,jck->tjk", c, np.linalg.inv(J.B)).reshape(T, 45)


def run(args: list, log: Path):
    """Run a repo script with this interpreter, output to ``log``; raise if it fails."""
    subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], check=True, cwd=REPO,
                   stdout=open(log, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/soma/quickstart"))
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filler-steps", type=int, default=5000, help="train_filler.py steps")
    ap.add_argument("--tol-fill-mm", type=float, default=10.0, help="median learned gap-fill error allowed per gap")
    ap.add_argument("--tol-angle-deg", type=float, default=10.0, help="median joint-angle error allowed")
    a = ap.parse_args()
    out, rng, T = Path(a.out), np.random.default_rng(a.seed), a.frames
    out.mkdir(parents=True, exist_ok=True)

    J = ss.Judge()
    pins = json.loads((REPO / "configs/surface_fit/static_final_pins.json").read_text())
    mk = list(pins)
    att = torch.tensor([pins[m] for m in mk], device=J.dev)
    z3 = np.zeros((1, 3), np.float32)
    S = markers(J, att, z3, np.zeros((1, 45)), z3)
    write_wide_csv(out / "static.csv", S + rng.normal(0, 0.1, (300, *S.shape[1:])), mk)
    np.savez(out / "static_params.npz", hand_pose=np.zeros((1, 45), np.float32), global_orient=z3, transl=z3,
             betas=np.zeros(10, np.float32), scale=1.0, names=np.array(mk), att=att.cpu().numpy())

    tt = np.arange(T) / 200
    go = np.stack([0.3 * np.sin(2 * np.pi * f * tt + p) for f, p in ((0.13, 0), (0.07, 1), (0.11, 2))], 1)
    tr = np.stack([0.05 * np.sin(2 * np.pi * f * tt + p) for f, p in ((0.05, 0), (0.08, 2), (0.06, 4))], 1)
    hp = poses(J, T, rng)
    X = markers(J, att, go, hp, tr)
    Xg = X + rng.normal(0, 0.5, X.shape)
    for m, (s, e) in GAPS.items():
        Xg[s:e, mk.index(m)] = np.nan
    write_wide_csv(out / "movement.csv", Xg, mk)
    np.savez(out / "truth.npz", markers=X, hand_pose=hp, global_orient=go, transl=tr, names=np.array(mk))
    print(f"synthetic participant written to {out} ({T} frames, gaps in {', '.join(GAPS)})", flush=True)

    from vicon2mano.core.loader import load_csv

    def errors(csv: Path) -> dict:
        Fm, fn = load_csv(str(csv))[:2]
        Fm = Fm[:, [[n.split(":")[-1] for n in fn].index(m) for m in mk]]
        e = {m: np.linalg.norm(Fm[s:e, mk.index(m)] - X[s:e, mk.index(m)], axis=-1) for m, (s, e) in GAPS.items()}
        return {m: dict(median=round(float(np.nanmedian(v)), 2), max=round(float(np.nanmax(v)), 2),
                        all_filled=bool(np.isfinite(v).all())) for m, v in e.items()}

    run(["scripts/soma/gap_fill.py", "--csv", out / "movement.csv", "--out", out / "filled.csv"], out / "gap_fill.log")
    run(["scripts/soma/train_filler.py", "--csv", out / "movement.csv", "--steps", a.filler_steps, "--seed", a.seed,
         "--out", out / "filler.pt"], out / "train_filler.log")
    run(["scripts/soma/gap_fill.py", "--csv", out / "movement.csv", "--out", out / "filled_learned.csv",
         "--filler", out / "filler.pt", "--max-gap", 100000], out / "gap_fill_learned.log")
    rigid, learned = errors(out / "filled.csv"), errors(out / "filled_learned.csv")
    run(["scripts/soma/movement_fit.py", "--participant", "SYNTH", "--static", out / "static_params.npz",
         "--static-csv", out / "static.csv", "--csv", out / "filled_learned.csv", "--out", out / "fit", "--step", 10,
         "--chunk", 1000, "--iters", 800, "--w-filled", 0.1], out / "movement_fit.log")
    f = np.load(out / "fit" / "fit.npz")
    fr = f["frames_fitted"]
    rot = lambda h: Rotation.from_rotvec(np.asarray(h).reshape(-1, 3))
    ang = np.degrees((rot(f["hand_pose"]).inv() * rot(hp[fr])).magnitude()).reshape(len(fr), 15)
    res = dict(synthetic=True, frames=T, fitted_frames=int(len(fr)), gap_fill_mm=dict(rigid=rigid, learned=learned),
               joint_angle_error_deg=dict(median=round(float(np.median(ang)), 2), p95=round(float(np.percentile(ang, 95)), 2)),
               fit_summary=json.loads((out / "fit" / "summary.json").read_text()).get("max_dist_mm"),
               tolerances=dict(fill_median_mm=a.tol_fill_mm, angle_median_deg=a.tol_angle_deg))
    res["passed"] = bool(all(v["all_filled"] and v["median"] <= a.tol_fill_mm for v in learned.values())
                         and res["joint_angle_error_deg"]["median"] <= a.tol_angle_deg)
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
