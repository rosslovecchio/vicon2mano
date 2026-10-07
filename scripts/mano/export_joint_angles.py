#!/usr/bin/env python3
"""Export per-finger joint angles from a completed MANOFitter result.

Decodes the fitted ``hand_pose`` (PCA coefficients, or full 45-dim if the
fit used ``--no-pca``) into the 15 finger joints' axis-angle rotations, via
the same ``hand_components``/``hand_mean`` decode MANO applies internally
(see ``MANOFitter._decode_full_pose`` in an earlier session -- this script
is standalone, not a dependency on that removed helper).

Joint angle here means each joint's rotation *magnitude* in degrees
(||axis-angle||), not a per-DOF flexion/abduction decomposition -- MANO's
three axis-angle components per joint are not individually labelled
anatomically, so a single magnitude is the honest, simple readout: ~0deg
at rest, larger with more flexion, not sign-distinguishing
flexion-vs-hyperextension. Good enough for "is this finger bending and by
how much"; not a clinical goniometer reading.

Output
------
<out>.csv   one row per frame: frame, time_s, then one column per joint
            (e.g. thumb_mcp_deg, thumb_pip_deg, ..., pinky_dip_deg)
<out>_<finger>.png   (only with --plot) one figure per finger, 3 stacked
            subplots (MCP/PIP/DIP) of that joint's angle over time

Usage
-----
python scripts/mano/export_joint_angles.py \\
    --npz results/mano/P10_Trial2_handsonly/mano_fit_right.npz \\
    --out results/mano/P10_Trial2_handsonly/joint_angles \\
    --fps 200 \\
    --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# smplx's chumpy dependency uses removed numpy type aliases; same compat
# shim as vicon2mano.core.fitter, needed before importing smplx here too.
import inspect as _inspect
import numpy as _np
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
for _attr, _builtin in {"int": int, "float": float, "bool": bool, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _builtin)

# Finger/joint order in smplx's native 45-dim hand_pose layout (see
# vicon2mano.core.fitter's module docstring): index, middle, pinky, ring,
# thumb, each as (mcp, pip, dip).
FINGER_ORDER = ["index", "middle", "pinky", "ring", "thumb"]
JOINT_SUFFIXES = ["mcp", "pip", "dip"]
JOINT_COLUMNS = [f"{finger}_{suffix}_deg" for finger in FINGER_ORDER for suffix in JOINT_SUFFIXES]

FINGER_COLORS = {
    "thumb": "#e74c3c", "index": "#27ae60", "middle": "#2980b9",
    "ring": "#e67e22", "pinky": "#8e44ad",
}


def decode_full_pose(hand_pose: np.ndarray, model) -> np.ndarray:
    """(T, n_pca) or (T, 45) hand_pose -> (T, 15, 3) axis-angle per joint."""
    import torch
    hp = torch.tensor(hand_pose, dtype=torch.float32)
    if hand_pose.shape[1] == 45:
        full = hp + model.hand_mean
    else:
        full = hp @ model.hand_components + model.hand_mean
    return full.view(-1, 15, 3).numpy()


def joint_angles_deg(hand_pose: np.ndarray, model) -> pd.DataFrame:
    full = decode_full_pose(hand_pose, model)       # (T, 15, 3)
    mag_deg = np.degrees(np.linalg.norm(full, axis=-1))  # (T, 15)
    return pd.DataFrame(mag_deg, columns=JOINT_COLUMNS)


def plot_finger(df: pd.DataFrame, finger: str, time_s: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [f"{finger}_{s}_deg" for s in JOINT_SUFFIXES]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    color = FINGER_COLORS[finger]
    for ax, col, suffix in zip(axes, cols, JOINT_SUFFIXES):
        ax.plot(time_s, df[col], color=color, linewidth=0.8)
        ax.set_ylabel(f"{suffix.upper()} (deg)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{finger.capitalize()} joint angles — P10/Trial2 Hands only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="FitResult .npz from vicon2mano")
    ap.add_argument("--out", required=True, help="output path stem (no extension)")
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--fps", type=float, default=200.0,
                     help="capture rate in Hz, for the time_s column and plot x-axis "
                          "(read from the CSV's Trajectories header, not assumed)")
    ap.add_argument("--plot", action="store_true", help="also write one PNG per finger")
    args = ap.parse_args()

    import smplx
    model_path = Path(args.mano_dir)
    if model_path.is_dir() and not (model_path / "mano").exists():
        side_str = "RIGHT" if args.side == "right" else "LEFT"
        pkl = model_path / f"MANO_{side_str}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"MANO weights not found at {pkl}")
        model_path = pkl
    model = smplx.create(
        str(model_path), model_type="mano",
        is_rhand=(args.side == "right"), use_pca=True, num_pca_comps=6,
        flat_hand_mean=True, batch_size=1,
    )

    data = np.load(args.npz)
    hand_pose = data["hand_pose"]
    print(f"[export_joint_angles] {hand_pose.shape[0]} frames, hand_pose dim {hand_pose.shape[1]}")

    df = joint_angles_deg(hand_pose, model)
    df.insert(0, "frame", np.arange(len(df)))
    df.insert(1, "time_s", df["frame"] / args.fps)

    out_csv = Path(f"{args.out}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[export_joint_angles] saved {out_csv}")

    print("Per-joint angle summary (mean / max, deg):")
    for col in JOINT_COLUMNS:
        print(f"  {col:16s} mean {df[col].mean():5.1f}  max {df[col].max():5.1f}")

    if args.plot:
        for finger in FINGER_ORDER:
            plot_finger(df, finger, df["time_s"].to_numpy(),
                        Path(f"{args.out}_{finger}.png"))


if __name__ == "__main__":
    main()
