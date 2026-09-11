#!/usr/bin/env python3
"""Triage a directory of Vicon hand recordings before fitting.

Scans every recording (CSV / C3D), measures how complete and how labelable
it is, assigns a difficulty tier, and writes:

* ``triage_summary.csv`` — one row per recording with all metrics + tier.
* ``triage_clean_for_labeler.csv`` — (file, side) of clean recordings usable
  as (marker → label) supervision for training the deep labeler.

No fitting is performed; this is a cheap pre-pass that decides the
reference/clean subset and gates out unsalvageable files before any GPU work.
See COHORT_STRATEGY.md for how the tiers feed the cohort plan.

Usage
-----
python scripts/triage_dataset.py --data-dir /path/to/recordings \\
    [--out-dir vicon2mano/eval/triage] [--pattern "*.csv" "*.c3d"]
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vicon2mano.core.correspondence import _LABEL_HINTS, _normalise_label
from vicon2mano.core.loader import load_c3d, load_csv, load_h5_trajectory

# ─── Tier thresholds (tunable) ────────────────────────────────────────────────
# A recording is rated per hand-side it contains. Tiers key on *well-observed*
# joints — labels resolved by the seed hints AND whose marker is present in at
# least MIN_MARKER_PRESENCE of frames. A recognized label whose marker is
# (nearly) always NaN cannot constrain the fit, so it does not count. The
# secondary check is the median presence across those observed joints.
#   clean        — label-seed reference quality, full marker complement
#   usable       — label-seed works, minor gaps
#   degraded     — label-seed unreliable; needs the trained labeler / Hungarian
#   unsalvageable— too few observed joints to constrain a hand fit
MIN_MARKER_PRESENCE = 0.5                              # a joint "observed" if ≥ this
TIER_CLEAN = dict(min_obs=15, min_presence=0.95)
TIER_USABLE = dict(min_obs=10, min_presence=0.70)
TIER_DEGRADED = dict(min_obs=6, min_presence=0.40)     # else unsalvageable

SIDES = ("left", "right")


def _count_label_matches(labels: list[str], side: str) -> dict[str, int]:
    """For one side, return {joint_name: marker_index} resolved by hints.

    Mirrors correspondence.label_seed but without the ≥10 cutoff, and
    restricted to markers naming this side (or side-agnostic labels)."""
    opposite = {"left": "right", "right": "left"}[side]
    norm: list[str | None] = []
    for l in labels:
        ll = l.lower()
        if opposite in ll:
            norm.append(None)            # other hand's marker
        else:
            norm.append(_normalise_label(l))
    matched: dict[str, int] = {}
    for joint, hints in _LABEL_HINTS.items():
        for idx, lbl in enumerate(norm):
            if lbl is not None and any(h in lbl for h in hints):
                matched[joint] = idx
                break
    return matched


def _side_present(labels: list[str], side: str) -> bool:
    return any(side in l.lower() for l in labels)


def _tier(n_obs: int, presence: float) -> str:
    if n_obs >= TIER_CLEAN["min_obs"] and presence >= TIER_CLEAN["min_presence"]:
        return "clean"
    if n_obs >= TIER_USABLE["min_obs"] and presence >= TIER_USABLE["min_presence"]:
        return "usable"
    if n_obs >= TIER_DEGRADED["min_obs"] and presence >= TIER_DEGRADED["min_presence"]:
        return "degraded"
    return "unsalvageable"


def _load(path: Path):
    """Return (markers (T,N,3) mm, labels, rate_hz or None)."""
    if path.suffix.lower() == ".c3d":
        markers, labels, rate = load_c3d(str(path))
        return markers, labels, rate
    markers, labels = load_csv(str(path))
    return markers, labels, None


def analyse(path: Path) -> list[dict]:
    """Return one record per hand-side present in the file (≥1)."""
    markers, labels, rate = _load(path)            # (T, N, 3) mm
    return analyse_loaded(str(path), markers, labels, rate)


def analyse_loaded(source: str, markers, labels, rate) -> list[dict]:
    """Per-hand-side QA records for an already-loaded recording."""
    T, N, _ = markers.shape
    finite = np.isfinite(markers).all(axis=-1)     # (T, N)

    sides = [s for s in SIDES if _side_present(labels, s)]
    if not sides:
        sides = ["unknown"]                        # no side designators in labels

    records = []
    for side in sides:
        if side == "unknown":
            matched = {}
            for s in SIDES:                        # best of either hint set
                m = _count_label_matches(labels, s)
                if len(m) > len(matched):
                    matched = m
            side_label = "unknown"
        else:
            matched = _count_label_matches(labels, side)
            side_label = side

        midx = sorted(matched.values())
        if midx:
            per_joint = finite[:, midx].mean(axis=0)        # presence per matched joint
            observed = per_joint >= MIN_MARKER_PRESENCE
            n_obs = int(observed.sum())
            # median presence across the observed joints drives the tier
            presence = float(np.median(per_joint[observed])) if n_obs else 0.0
            worst = float(per_joint.min())
        else:
            n_obs = 0
            presence = worst = 0.0

        records.append({
            "file": source,
            "side": side_label,
            "n_frames": T,
            "n_markers": N,
            "overall_presence": round(float(finite.mean()), 4),
            "labels_recognized": len(matched),
            "joints_observed": n_obs,
            "observed_presence_median": round(presence, 4),
            "matched_presence_worst": round(worst, 4),
            "rate_hz": "" if rate is None else round(float(rate), 2),
            "tier": _tier(n_obs, presence),
            "error": "",
        })
    return records


def _error_row(source: str, e: Exception) -> dict:
    return {
        "file": source, "side": "", "n_frames": 0, "n_markers": 0,
        "overall_presence": 0.0, "labels_recognized": 0,
        "joints_observed": 0, "observed_presence_median": 0.0,
        "matched_presence_worst": 0.0, "rate_hz": "", "tier": "error",
        "error": f"{type(e).__name__}: {e}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="directory of CSV/C3D files to scan")
    src.add_argument("--h5", help="consolidated HDF5 (scans every participant/session)")
    ap.add_argument("--pattern", nargs="+", default=["*.csv", "*.c3d"],
                    help="glob patterns for recordings (--data-dir mode)")
    ap.add_argument("--out-dir", default="vicon2mano/eval/triage")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    rows = []

    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as f:
            tasks = [(px, sess) for px in sorted(f.keys())
                     for sess in sorted(f[px].keys())]
        print(f"[triage] scanning {len(tasks)} (participant, session) datasets "
              f"in {args.h5}")
        for px, sess in tasks:
            source = f"{px}/{sess}"
            try:
                markers, labels, rate = load_h5_trajectory(args.h5, px, sess)
                rows.extend(analyse_loaded(source, markers, labels, rate))
            except Exception as e:
                print(f"  ! {source}: {e}")
                rows.append(_error_row(source, e))
    else:
        data_dir = Path(args.data_dir)
        files = sorted({
            p for pat in args.pattern for p in data_dir.rglob(pat)
            if out_dir not in p.resolve().parents     # don't scan our own output
        })
        if not files:
            print(f"[triage] no files matching {args.pattern} under {data_dir}")
            return
        print(f"[triage] scanning {len(files)} files under {data_dir}")
        for p in files:
            try:
                rows.extend(analyse(p))
            except Exception as e:                 # never let one bad file abort the scan
                print(f"  ! {p.name}: {e}")
                rows.append(_error_row(str(p), e))
                traceback.print_exc(limit=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["file", "side", "n_frames", "n_markers", "overall_presence",
              "labels_recognized", "joints_observed", "observed_presence_median",
              "matched_presence_worst", "rate_hz", "tier", "error"]
    summary = out_dir / "triage_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # clean (file, side) pairs → labeler training supervision
    clean = [r for r in rows if r["tier"] == "clean"]
    clean_csv = out_dir / "triage_clean_for_labeler.csv"
    with open(clean_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "side", "joints_observed",
                                          "observed_presence_median"])
        w.writeheader()
        for r in clean:
            w.writerow({k: r[k] for k in ["file", "side", "joints_observed",
                                          "observed_presence_median"]})

    # console summary
    from collections import Counter
    tiers = Counter(r["tier"] for r in rows)
    n_sources = len({r["file"] for r in rows})
    print(f"\n[triage] {len(rows)} (recording, side) records from {n_sources} recordings")
    for t in ["clean", "usable", "degraded", "unsalvageable", "error"]:
        if tiers.get(t):
            print(f"  {t:14s} {tiers[t]}")
    print(f"\n  clean reference set (labeler training): {len(clean)} records")
    print(f"  summary  → {summary}")
    print(f"  clean    → {clean_csv}")


if __name__ == "__main__":
    main()
