#!/usr/bin/env python3
"""Build ``sampling_manifest.csv`` (deliverable #1 of the manual-validation
plan): turn the 500-frame anchor selection into a MATLAB-ready review list,
resolving each frame's actual trial CSV, and where a frame was selected
*because of* a specific marker/bone (largest_anomaly,
geometry_availability_disagreement), carrying that marker through as a
reviewing hint. This is also the record of "how frames were selected" the
validation report cites.

Reads ``results/shared/frame_sample/selected_anchors.csv`` (built by
``select_frames_for_review.py``) plus each trial's own ``trial_summary.csv``
and ``events.csv`` under ``results/shared/geometric_consistency/``.

Frame numbers here are **0-indexed** (matching this repo's own array
convention throughout -- see ``vicon2mano.core.dataset.load_ref_ranges_csv``'s
docstring on the same +1 offset). The MATLAB side converts to Nexus's
1-indexed "Vicon Frame" numbering; this export does not.

Usage
-----
python scripts/shared/build_sampling_manifest.py
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

from vicon2mano.core import dataset as ds  # noqa: E402

GEOM_DIR = REPO_ROOT / "results" / "shared" / "geometric_consistency"
ANCHORS_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "selected_anchors.csv"
WINDOWS_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "selected_windows.csv"
OUT_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "sampling_manifest.csv"

MARKER_CATEGORIES = ("largest_anomaly", "geometry_availability_disagreement")


def trial_dir_name(participant: str, trial_id: str) -> str:
    return f"{participant}_{ds.slug(trial_id)}"


def load_trial_metadata(participant: str, trial_id: str) -> dict | None:
    d = GEOM_DIR / trial_dir_name(participant, trial_id) / "trial_summary.csv"
    if not d.exists():
        return None
    row = pd.read_csv(d).iloc[0]
    return {"analysis_trial": row["analysis_trial"], "n_frames": int(row["n_frames"])}


def load_events_lookup(participant: str, trial_id: str) -> pd.DataFrame | None:
    d = GEOM_DIR / trial_dir_name(participant, trial_id) / "events.csv"
    if not d.exists():
        return None
    return pd.read_csv(d)


def build_review_list(anchors: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    window_bounds = (windows.groupby(["participant", "trial_id"])["frame"]
                     .agg(window_min="min", window_max="max"))

    rows = []
    trial_meta_cache: dict[tuple, dict] = {}
    events_cache: dict[tuple, pd.DataFrame] = {}

    for _, a in anchors.iterrows():
        key = (a["participant"], a["trial_id"])
        if key not in trial_meta_cache:
            trial_meta_cache[key] = load_trial_metadata(*key)
        meta = trial_meta_cache[key]
        if meta is None:
            print(f"[warn] no trial_summary.csv for {key} -- skipping from review list")
            continue

        marker_of_interest = None
        associated_bone = None
        if a["category"] in MARKER_CATEGORIES:
            if key not in events_cache:
                events_cache[key] = load_events_lookup(*key)
            events = events_cache[key]
            if events is not None:
                match = events[events["peak_frame"] == a["frame"]]
                if len(match):
                    hit = match.iloc[0]
                    marker_of_interest = hit["marker"]
                    associated_bone = hit["associated_bone"]

        wb = window_bounds.loc[key] if key in window_bounds.index else None
        rows.append({
            "participant": a["participant"],
            "trial_id": a["trial_id"],
            "analysis_trial": meta["analysis_trial"],
            "n_frames": meta["n_frames"],
            "frame": int(a["frame"]),
            "window_start": int(wb["window_min"]) if wb is not None else int(a["frame"]),
            "window_end": int(wb["window_max"]) if wb is not None else int(a["frame"]),
            "category": a["category"],
            "selection_metric": a["selection_metric"],
            "marker_of_interest": marker_of_interest,
            "associated_bone": associated_bone,
        })
    return (pd.DataFrame(rows)
            .sort_values(["participant", "trial_id", "frame"])
            .reset_index(drop=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchors-csv", default=str(ANCHORS_CSV))
    ap.add_argument("--windows-csv", default=str(WINDOWS_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    anchors_csv = Path(args.anchors_csv)
    windows_csv = Path(args.windows_csv)
    if not anchors_csv.exists() or not windows_csv.exists():
        raise SystemExit(f"{anchors_csv} / {windows_csv} not found -- run "
                          "scripts/shared/select_frames_for_review.py first")
    anchors = pd.read_csv(anchors_csv)
    windows = pd.read_csv(windows_csv)
    review_list = build_review_list(anchors, windows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review_list.to_csv(out_csv, index=False)
    print(f"Wrote {len(review_list)} row(s) to {out_csv} "
          f"({review_list['participant'].nunique()} participants, "
          f"{review_list.groupby(['participant', 'trial_id']).ngroups} trials)")


if __name__ == "__main__":
    main()
