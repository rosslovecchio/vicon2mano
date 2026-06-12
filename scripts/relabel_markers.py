#!/usr/bin/env python3
"""Label an unlabeled (but complete) Vicon marker cloud — positions untouched.

Some recordings have every marker present but no usable labels. This assigns
each marker a name by matching it to a labeled *template* recording via a
rigid-/scale-invariant geometric descriptor (see
``correspondence.relabel_by_template``), then writes the recording back out in
wide-Nexus CSV format with **identical coordinates** and the new column names.

It moves nothing; it only names columns. Use it as the first step before
fitting, to turn an unlabeled recording into one the label-seed path accepts.

Usage
-----
python scripts/relabel_markers.py \\
    --unlabeled  path/to/unlabeled.csv \\
    --template   data/Pxh8/Hands_only_Left/trajectories_synced.csv \\
    --out        path/to/relabeled.csv \\
    [--stride 20]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicon2mano.correspondence import relabel_by_template, relabel_two_hands
from vicon2mano.loader import load_c3d, load_csv


def _load(path: Path):
    if path.suffix.lower() == ".c3d":
        markers, labels, _ = load_c3d(str(path))
        return markers, labels
    return load_csv(str(path))


def write_wide_csv(path: Path, markers: np.ndarray, labels: list[str]):
    """Write (T, N, 3) mm markers + labels as a single-header wide Nexus CSV.

    Blank cells for NaN, matching what loader.load_csv expects to read back."""
    T, N, _ = markers.shape
    header = ["_Frame", "_Sub Frame"]
    for lab in labels:
        header += [f"{lab}_X", "_Y", "_Z"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(T):
            row = [t + 1, 0]
            for n in range(N):
                for c in range(3):
                    v = markers[t, n, c]
                    row.append("" if not np.isfinite(v) else f"{v:.6f}")
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unlabeled", required=True, help="recording to label")
    ap.add_argument("--template", required=True,
                    help="labeled reference recording (same marker protocol)")
    ap.add_argument("--out", required=True, help="output relabeled CSV")
    ap.add_argument("--stride", type=int, default=20,
                    help="frame subsampling for the descriptor")
    ap.add_argument("--cost-warn", type=float, default=0.5,
                    help="flag matches with cost above this as low-confidence")
    ap.add_argument("--two-hands", action="store_true",
                    help="cloud has both hands: cluster + relabel each side "
                         "(chirality-safe; required for two-hand recordings)")
    args = ap.parse_args()

    markers, old_labels = _load(Path(args.unlabeled))
    tmpl_markers, tmpl_labels = _load(Path(args.template))
    print(f"[relabel] unlabeled: {markers.shape[1]} markers, {markers.shape[0]} frames")
    print(f"[relabel] template:  {tmpl_markers.shape[1]} markers ({len(set(tmpl_labels))} labels)")

    relabel = relabel_two_hands if args.two_hands else relabel_by_template
    new_labels, cost = relabel(
        markers, tmpl_markers, tmpl_labels, stride=args.stride)

    # position invariant: assert we are not touching coordinates
    write_wide_csv(Path(args.out), markers, new_labels)

    print("\n[relabel] marker → assigned label (cost):")
    flagged = 0
    for i, (lab, c) in enumerate(zip(new_labels, cost)):
        old = old_labels[i] if i < len(old_labels) else ""
        mark = ""
        if c > args.cost_warn:
            mark = "  ⚠ low-confidence"; flagged += 1
        old_str = f"  (was '{old}')" if old and old != lab else ""
        print(f"  [{i:2d}] {lab:16s} cost={c:6.3f}{old_str}{mark}")

    # if the input had real labels, report agreement as a sanity check
    if old_labels and not all(l.startswith("Unlabeled") for l in old_labels):
        agree = sum(1 for a, b in zip(old_labels, new_labels) if a == b)
        print(f"\n[relabel] agreement with existing labels: {agree}/{len(new_labels)}")
    if flagged:
        print(f"[relabel] {flagged} low-confidence matches (cost > {args.cost_warn}) "
              "— check these or supply a closer template")
    print(f"[relabel] wrote {args.out} (positions unchanged)")


if __name__ == "__main__":
    main()
