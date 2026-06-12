#!/usr/bin/env python3
"""Synthetic ground-truth evaluation of the MANO fitting pipeline.

Generates marker recordings from MANO poses that are *known*, runs the
production ``MANOFitter`` on them, and reports finger joint angle error
against ground truth — the number a clinical methods paper needs.

Realism choices (each makes the test harder, not easier):

* GT poses are sampled in **PCA-15** space while the fitter uses PCA-6, so
  the fitted model cannot represent the truth exactly (model mismatch).
* GT betas are random per sequence (subject shape mismatch; fitter must
  estimate shape itself).
* Markers mimic the clinical protocol (palm + MCP/PIP/DIP per finger): each
  marker has a constant "skin-mount" offset (default 8 mm, random direction,
  rotating with the hand), per-frame Gaussian noise (1 mm) and short NaN
  dropout gaps (~2 %).
* Motion: smooth keyframe-interpolated grasps (2 keyframes/s) + global
  rotation/translation drift.

Usage
-----
python scripts/eval_synth_fit.py [--n-seq 3] [--n-frames 1000] \\
    [--offset-mm 8] [--noise-mm 1] [--seed 0] [--out-dir vicon2mano/eval/fit]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vicon2mano import fitter as F
from vicon2mano.fitter import MANOFitter, FitConfig, mano_output_to_joints21
from export_finger_angles import flexion_angles, FINGER_JOINTS

# marker slots: label → GT joint index (repo 21-joint order).
# The clinical sparse protocol: palm + MCP/PIP/DIP per finger, no fingertip.
MARKER_PROTOCOL = {
    "Palm_Left2": 0,
    "Thumb_Left1": 1, "Thumb_Left2": 2, "Thumb_Left3": 3,
    "Index_Left1": 5, "Index_Left2": 6, "Index_Left3": 7,
    "Middle_Left1": 9, "Middle_Left2": 10, "Middle_Left3": 11,
    "Ring_Left1": 13, "Ring_Left2": 14, "Ring_Left3": 15,
    "Pinky_Left1": 17, "Pinky_Left2": 18, "Pinky_Left3": 19,
}
# Tip-augmented protocol: adds a marker per fingertip (4 extra slots,
# diagnostic — tests whether distal joints are tip-limited).
TIP_MARKERS = {"Thumb_Left4": 4, "Index_Left4": 8, "Middle_Left4": 12,
               "Ring_Left4": 16, "Pinky_Left4": 20}

GT_PCA = 15


def _smoothstep(x):
    return x * x * (3 - 2 * x)


def _interp_keyframes(keys: np.ndarray, n_per_seg: int) -> np.ndarray:
    """(K, D) keyframes → ((K-1)*n_per_seg, D) smooth trajectory."""
    out = []
    w = _smoothstep(np.linspace(0, 1, n_per_seg, endpoint=False))[:, None]
    for k in range(len(keys) - 1):
        out.append((1 - w) * keys[k] + w * keys[k + 1])
    return np.concatenate(out, axis=0)


def generate_sequence(rng, model, T, offset_m, noise_m, device, gt_pca=GT_PCA,
                      protocol=None):
    """Return (markers_mm (T, 16, 3), labels, gt_joints (T, 21, 3) metres)."""
    from scipy.spatial.transform import Rotation, Slerp

    n_per_seg = 50                                  # 0.5 s per keyframe @100Hz
    K = T // n_per_seg + 2                          # keys span past frame T-1
    pose_keys = np.clip(rng.normal(0, 1.0, (K, gt_pca)), -2.5, 2.5)
    hand_pose = _interp_keyframes(pose_keys, n_per_seg)[:T]

    rot_keys = Rotation.random(K, random_state=rng.integers(1 << 31))
    # random walk: each key partially blends toward the previous one
    rvs = rot_keys.as_rotvec()
    for k in range(1, K):
        rvs[k] = rvs[k - 1] + 0.5 * (rvs[k] - rvs[k - 1])
    slerp = Slerp(np.arange(K) * n_per_seg,
                  Rotation.from_rotvec(rvs))
    global_orient = slerp(np.arange(T)).as_rotvec()

    tr_keys = rng.uniform(-0.25, 0.25, (K, 3)) + np.array([1.0, -0.1, 1.1])
    transl = _interp_keyframes(tr_keys, n_per_seg)[:T]
    betas = rng.normal(0, 0.5, 10).astype(np.float32)

    # GT joints via PCA-15 model
    js = []
    bs = 512
    with torch.no_grad():
        for s in range(0, T, bs):
            sl = slice(s, min(s + bs, T))
            out = model(
                global_orient=torch.tensor(global_orient[sl], dtype=torch.float32, device=device),
                hand_pose=torch.tensor(hand_pose[sl], dtype=torch.float32, device=device),
                transl=torch.tensor(transl[sl], dtype=torch.float32, device=device),
                betas=torch.tensor(betas, device=device).unsqueeze(0).expand(sl.stop - sl.start, -1),
            )
            js.append(mano_output_to_joints21(out).cpu().numpy())
    gt_joints = np.concatenate(js, axis=0)          # (T, 21, 3) metres

    # markers: constant skin offset (rotates with the hand) + noise + gaps
    protocol = protocol if protocol is not None else MARKER_PROTOCOL
    labels = list(protocol.keys())
    jidx = np.array(list(protocol.values()))
    off_local = rng.normal(0, 1, (len(jidx), 3))
    off_local *= offset_m / np.linalg.norm(off_local, axis=1, keepdims=True)
    R = Rotation.from_rotvec(global_orient).as_matrix()      # (T, 3, 3)
    off_world = np.einsum("tij,mj->tmi", R, off_local)       # (T, 16, 3)
    markers = gt_joints[:, jidx] + off_world
    markers += rng.normal(0, noise_m, markers.shape)

    # ~2% dropout in runs of 5–20 frames
    n_gap_frames = int(0.02 * T * len(jidx))
    dropped = 0
    while dropped < n_gap_frames:
        mi = rng.integers(len(jidx))
        t0 = rng.integers(T)
        run = rng.integers(5, 21)
        markers[t0:t0 + run, mi] = np.nan
        dropped += min(run, T - t0)

    return markers * 1000.0, labels, gt_joints      # mm for the fitter


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-seq", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=1000)
    ap.add_argument("--offset-mm", type=float, default=8.0)
    ap.add_argument("--noise-mm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-pca", type=int, default=GT_PCA,
                    help="PCA components for the ground-truth pose model "
                         "(set to 6 to remove model mismatch vs the fitter)")
    ap.add_argument("--tips", action="store_true",
                    help="add fingertip markers (diagnostic)")
    ap.add_argument("--fit-pca", type=int, default=6,
                    help="PCA components for the fitter (production = 6)")
    ap.add_argument("--out-dir", default="vicon2mano/eval/fit")
    args = ap.parse_args()
    protocol = dict(MARKER_PROTOCOL)
    if args.tips:
        protocol.update(TIP_MARKERS)

    cfg = FitConfig(hand_side="left", n_pca_comps=args.fit_pca)
    fitter = MANOFitter(cfg)
    device = fitter.device

    # separate richer GT model (PCA-15)
    model_path = Path(cfg.mano_model_path) / "MANO_LEFT.pkl"
    gt_model = F.smplx.create(
        str(model_path), model_type="mano", is_rhand=False,
        use_pca=True, num_pca_comps=args.gt_pca, flat_hand_mean=True, batch_size=1,
    ).to(device)

    angle_names = [f"{f}_{j}" for f in FINGER_JOINTS for j in ("mcp", "pip", "dip")]
    err_all, corr_all, jpos_all = [], [], []
    for s in range(args.n_seq):
        rng = np.random.default_rng(args.seed + s)
        markers_mm, labels, gt_joints = generate_sequence(
            rng, gt_model, args.n_frames,
            args.offset_mm * 1e-3, args.noise_mm * 1e-3, device,
            gt_pca=args.gt_pca, protocol=protocol)
        print(f"[eval_synth_fit] sequence {s}: fitting {args.n_frames} frames …")
        res = fitter.fit(markers_mm, labels, verbose=False)

        gt_ang = flexion_angles(gt_joints)
        fit_ang = flexion_angles(res.joints)
        err = np.stack([np.abs(fit_ang[n] - gt_ang[n]) for n in angle_names])   # (15, T)
        corr = np.array([np.corrcoef(fit_ang[n], gt_ang[n])[0, 1]
                         for n in angle_names])
        jpos = np.linalg.norm(res.joints - gt_joints, axis=-1).mean() * 1000
        err_all.append(err)
        corr_all.append(corr)
        jpos_all.append(jpos)
        print(f"  seq {s}: angle MAE {err.mean():.2f} deg, "
              f"median corr {np.median(corr):.3f}, joint pos err {jpos:.1f} mm")

    err = np.concatenate(err_all, axis=1)           # (15, n_seq*T)
    corr = np.stack(corr_all).mean(axis=0)
    print("\n=== Synthetic ground-truth results "
          f"({args.n_seq} seq × {args.n_frames} frames, "
          f"{args.offset_mm:.0f} mm skin offset, {args.noise_mm:.0f} mm noise) ===")
    print(f"{'angle':12s} {'MAE deg':>8s} {'P95 deg':>8s} {'corr':>6s}")
    for i, n in enumerate(angle_names):
        print(f"{n:12s} {err[i].mean():8.2f} "
              f"{np.percentile(err[i], 95):8.2f} {corr[i]:6.3f}")
    print("-" * 38)
    print(f"{'OVERALL':12s} {err.mean():8.2f} "
          f"{np.percentile(err, 95):8.2f} {np.median(corr):6.3f}")
    print(f"mean joint position error: {np.mean(jpos_all):.1f} mm")


if __name__ == "__main__":
    main()
