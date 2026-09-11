#!/usr/bin/env python3
"""Export per-frame finger flexion angles to CSV.

For each finger (thumb, index, middle, ring, pinky) computes the flexion
angle at the MCP, PIP and DIP joints from the fitted MANO skeleton: the
angle between the bone entering the joint and the bone leaving it. 0 deg
means the two bones are collinear (finger fully straight); larger values
mean more bend.

Usage
-----
python scripts/export_finger_angles.py \\
    --npz data/Pxh8/Hands_only_Left/mano_fit_left.npz left \\
    --npz data/Pxh8/Hands_only_Left/mano_fit_right.npz right \\
    --out vicon2mano/eval/fit/finger_angles.csv \\
    [--rate 100]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# (mcp, pip, dip, tip) joint indices per finger, in repo 21-joint order.
FINGER_JOINTS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
WRIST = 0


def flexion_angles(joints: np.ndarray) -> dict[str, np.ndarray]:
    """joints: (T, 21, 3) metres -> {f"{finger}_{joint}": (T,) degrees}."""
    out = {}
    for finger, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        chain = [WRIST, mcp, pip, dip, tip]
        for name, (a, b, c) in zip(
            ("mcp", "pip", "dip"), zip(chain[:-2], chain[1:-1], chain[2:])
        ):
            v_in = joints[:, b] - joints[:, a]
            v_out = joints[:, c] - joints[:, b]
            v_in = v_in / np.linalg.norm(v_in, axis=-1, keepdims=True)
            v_out = v_out / np.linalg.norm(v_out, axis=-1, keepdims=True)
            cos = np.clip((v_in * v_out).sum(-1), -1.0, 1.0)
            out[f"{finger}_{name}"] = np.degrees(np.arccos(cos))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", action="append", nargs=2, required=True,
                    metavar=("FILE", "HAND_NAME"),
                    help="FitResult .npz and a label for its columns "
                         "(e.g. --npz mano_fit_left.npz left); repeatable")
    ap.add_argument("--out", required=True, help="output .csv path")
    ap.add_argument("--rate", type=float, default=100.0,
                    help="capture rate in Hz, for the time_s column")
    args = ap.parse_args()

    hands = {}
    T = None
    for path, name in args.npz:
        print(f"[export_finger_angles] loading {path} as '{name}'")
        joints = np.load(path)["joints"]  # (T, 21, 3) metres
        hands[name] = flexion_angles(joints)
        T = joints.shape[0] if T is None else min(T, joints.shape[0])

    fieldnames = ["frame", "time_s"]
    for name in hands:
        for finger in FINGER_JOINTS:
            for j in ("mcp", "pip", "dip"):
                fieldnames.append(f"{name}_{finger}_{j}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export_finger_angles] writing {T} rows -> {out}")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for t in range(T):
            row = [t, t / args.rate]
            for name, angles in hands.items():
                for finger in FINGER_JOINTS:
                    for j in ("mcp", "pip", "dip"):
                        row.append(f"{angles[f'{finger}_{j}'][t]:.2f}")
            writer.writerow(row)
    print(f"[export_finger_angles] done -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
