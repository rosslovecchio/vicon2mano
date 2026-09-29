#!/usr/bin/env python3
"""One row per trial, pulling together geometric self-consistency metrics
that are otherwise spread across ``results/shared/geometric_consistency/``.

Deliberately NOT a combined score -- each column is kept as its own
separately-meaningful number (see CLAUDE.md's fitting philosophy on this
repo generally avoiding premature combination), so a reader can sort by
whichever dimension matters for the next step (e.g. frame sampling) without
the choice of weighting being baked in here.

Reads ``results/shared/geometric_consistency/dataset/dataset_trial_summary.csv``
(written by ``geometric_self_consistency.py --all``) -- run that first if
this errors with a missing-file message.

Usage
-----
python scripts/shared/build_trial_overview.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATASET_CSV = REPO_ROOT / "results" / "shared" / "geometric_consistency" / "dataset" / "dataset_trial_summary.csv"
OUT_CSV = REPO_ROOT / "results" / "shared" / "trial_overview.csv"


def classify_trial_type(trial_id: str) -> str:
    """"Hands only" vs "HOI" (hand-object interaction), read off the
    session name exactly as manual_frames.csv spells it -- same two
    categories CLAUDE.md's own state-of-the-data notes use ("hands only
    trials are consistently 2-10x cleaner than HOI")."""
    low = trial_id.lower()
    if "hoi" in low:
        return "HOI"
    if "hand" in low:
        return "Hands only"
    return "unknown"


def build_overview(trial_summary: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "trial_id": trial_summary["trial_id"],
        "participant": trial_summary["subject_id"],
        "trial_type": trial_summary["trial_id"].map(classify_trial_type),
        "reference_source_type": trial_summary["reference_source_type"],
        "reference_status": trial_summary["reference_status"],
        "n_valid_reference_bones": trial_summary["n_valid_reference_bones"],
        # "anomaly count" = number of distinct suspicious geometric-deviation
        # events (vicon2mano.core.geometric_consistency.geometric_events),
        # not the larger "affected observations" count -- see that module
        # for the event/observation distinction. Rate is per frame, so
        # trials of different length are comparable.
        "geometric_anomaly_count": trial_summary["total_suspicious_events"],
        "geometric_anomaly_rate": trial_summary["total_suspicious_events"] / trial_summary["n_frames"],
        # Sum of each marker's own missing-frame count -- a frame where
        # several markers are simultaneously missing is counted once per
        # marker, consistent with how total_suspicious_events/
        # total_affected_observations are already marker-instance sums
        # elsewhere in this pipeline, not distinct-frame counts.
        "total_gap_frames": trial_summary["total_affected_missing_frames"],
        "mean_marker_availability": trial_summary["mean_availability_pct"],
    })
    return out.sort_values(["participant", "trial_id"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-csv", default=str(DATASET_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.exists():
        raise SystemExit(f"{dataset_csv} not found -- run "
                          "scripts/shared/geometric_self_consistency.py --all first")
    trial_summary = pd.read_csv(dataset_csv)
    overview = build_overview(trial_summary)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(out_csv, index=False)
    print(f"Wrote {len(overview)} trial(s) to {out_csv}")


if __name__ == "__main__":
    main()
