#!/usr/bin/env python3
"""Per-trial summary of a run_fullset.py output: how extreme the fitted angles
are, how well the fingers were fitted, and which flag signals fired.

Columns
  frames        fitted frames
  >120 b2b      frames where any finger joint's bone-to-bone flexion exceeds 120 deg
                (angle between the bone entering and the bone leaving the joint,
                0 = straight; the definition scripts/shared/export_finger_angles.py
                and the earlier "frames with any finger joint > 120 deg" figures use)
  >120 |aa|     same threshold on the axis-angle magnitude in angles.csv
  >120 flex     (with --flexion-axis) frames where any of the 12 finger joints index1..ring3
                (no thumb) has a flexion COMPONENT above 120 deg: the projection of the joint's
                rotation on its measured flexion axis, joint_axis_basis(...)[:, :, 0]. This is the
                measure behind the earlier docs' "176 frames"; it is not the same as either
                column above, and those figures came from a 400/400/600-iteration fit.
  fingers mm    mean marker-to-surface distance of the finger markers (fit.log)
  overall mm    same over all fitted markers (fit.log)
  label/fit/angle  share of frames in which each flag signal fired (flags.csv)

    python scripts/soma/trial_report.py --participants P12,P10
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FINGER_JOINTS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                 "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def flexion_b2b(joints: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T, 15) degrees; same definition as export_finger_angles.flexion_angles."""
    cols = []
    for mcp, pip, dip, tip in FINGER_JOINTS.values():
        chain = [0, mcp, pip, dip, tip]
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            vi, vo = joints[:, b] - joints[:, a], joints[:, c] - joints[:, b]
            vi /= np.linalg.norm(vi, axis=-1, keepdims=True)
            vo /= np.linalg.norm(vo, axis=-1, keepdims=True)
            cols.append(np.degrees(np.arccos(np.clip((vi * vo).sum(-1), -1, 1))))
    return np.stack(cols, 1)


def fit_log_residuals(log: Path) -> tuple[float, float]:
    t = log.read_text(encoding="utf-8", errors="replace")
    o = re.findall(r"OVERALL\s+([\d.]+) mm", t)
    f = re.findall(r"FINGERS\s+([\d.]+) mm", t)
    return (float(f[-1]) if f else float("nan"), float(o[-1]) if o else float("nan"))


def joints_from_fit(fit, model, device, frames, batch=20000) -> np.ndarray:
    import torch
    from vicon2mano.core.fitter import mano_output_to_joints21
    out = []
    betas = torch.tensor(fit["betas"], dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        for s in range(0, len(frames), batch):
            f = frames[s:s + batch]
            o = model(global_orient=torch.tensor(fit["global_orient"][f], device=device),
                      hand_pose=torch.tensor(fit["hand_pose"][f], device=device),
                      transl=torch.tensor(fit["transl"][f], device=device),
                      betas=betas.expand(len(f), -1), return_verts=True)
            out.append(mano_output_to_joints21(o).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", required=True)
    ap.add_argument("--out-root", default=str(REPO / "results/soma/fullset"))
    ap.add_argument("--mano-dir", default=None)
    ap.add_argument("--threshold", type=float, default=120.0)
    ap.add_argument("--flexion-axis", action="store_true",
                    help="also count frames with a flexion component above the threshold (index1..ring3)")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from vicon2mano.core.dataset import DATA_ROOT
    from vicon2mano.core.fitter import FitConfig, MANOFitter
    mano = args.mano_dir or str(DATA_ROOT / "mano_v1_2" / "models")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MANOFitter(FitConfig(mano_model_path=mano, hand_side="right", use_pca=False))._model.to(dev)

    Bmat = None
    if args.flexion_axis:
        sys.path.insert(0, str(REPO / "scripts" / "mano"))
        import fit_surface_markers as fsm
        Bmat = fsm.joint_axis_basis(model, dev, 45)
    print(f"{'trial':26s} {'frames':>7s} {'>120 b2b':>9s} {'>120 |aa|':>10s} "
          + (f"{'>120 flex':>10s} " if Bmat is not None else "")
          + f"{'fingers mm':>10s} {'overall mm':>10s} {'label':>6s} {'fit':>6s} {'angle':>6s}")
    for p in [x.strip() for x in args.participants.split(",") if x.strip()]:
        for tdir in sorted((Path(args.out_root) / p).iterdir()):
            ang = tdir / "angles.csv"
            if not ang.exists():
                continue
            a = pd.read_csv(ang)
            fit = np.load(tdir / "fit.npz")
            fr = a["frame"].to_numpy()
            b2b = flexion_b2b(joints_from_fit(fit, model, dev, fr))
            mag = a[[c for c in a.columns if c.endswith("_deg")]].to_numpy()
            fing, overall = fit_log_residuals(tdir / "fit.log")
            pct = lambda m: 100 * m.mean()
            flex = ""
            if Bmat is not None:
                hpj = fit["hand_pose"][fr].reshape(len(fr), 15, 3)
                comp = np.degrees(np.einsum("tjk,jk->tj", hpj, Bmat[:, :, 0]))[:, :12]
                flex = f"{(comp.max(1) > args.threshold).sum():10d} "
            print(f"{p + ' ' + tdir.name:26s} {len(a):7d} {(b2b.max(1) > args.threshold).sum():9d} "
                  f"{(mag.max(1) > args.threshold).sum():10d} {flex}{fing:10.2f} {overall:10.2f} "
                  f"{pct(a['label_signal']):5.1f}% {pct(a['fit_signal']):5.1f}% {pct(a['angle_signal']):5.1f}%")


if __name__ == "__main__":
    main()
