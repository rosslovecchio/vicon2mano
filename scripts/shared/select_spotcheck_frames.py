#!/usr/bin/env python3
"""Select, per trial, the frames for a direct Nexus spot-check of the
manual_identity_validation method:

- up to 5 `wrong_label`/`ghost` frames with SMALL anchor deviation (the
  palm triangle itself looked geometrically consistent -- a wrong_label
  verdict there is probably a genuine finger/marker swap)
- up to 5 with LARGE anchor deviation (the palm triangle itself differed --
  the verdict might be an artifact of a corrupted local reference frame,
  not a real labelling error)
- 3-5 `correct` frames
- 2-3 `ambiguous` frames

"Anchor deviation" is the max of the 3 palm-triangle bone-length deltas
(vicon2mano.core.validation.palm_triangle_bone_deltas), split at
--deviation-threshold-mm (default 10mm -- comfortably above the ~0.005mm
round-trip noise floor documented in build_annotations_from_correction and
below genuine mislabelling distances, consistent with this repo's other
bone-length tolerances).

Reads results/shared/frame_sample/manual_annotations.csv (built by
build_annotations_from_correction.py). Writes
results/shared/validation/spotcheck_manifest.csv, in the same shape
sampling_manifest.csv uses, so label_selected_frames.m's navigation pattern
can be reused directly by spotcheck_frames.m.

Usage
-----
python scripts/shared/select_spotcheck_frames.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv            # noqa: E402
from vicon2mano.core import dataset as ds              # noqa: E402
from vicon2mano.core import validation as val          # noqa: E402

ANNOTATIONS_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "manual_annotations.csv"
OUT_CSV = REPO_ROOT / "results" / "shared" / "validation" / "spotcheck_manifest.csv"

QUOTAS = {"wrong_small_anchor_dev": 5, "wrong_large_anchor_dev": 5,
         "correct": 5, "ambiguous": 3}


def run(annotations_csv: Path, out_csv: Path, deviation_threshold_mm: float, seed: int) -> None:
    annotations = pd.read_csv(annotations_csv)
    rng = np.random.default_rng(seed)

    rows = []
    for (participant, trial_id), g in annotations.groupby(["participant", "trial_id"]):
        trial_path, *_ = ds.find_trial_csv(participant, trial_id)
        if trial_path is None:
            continue
        corrected_path = ds.find_manual_csv(trial_path)
        if corrected_path is None:
            continue

        wrong = g[g["status"].isin(["wrong_label", "ghost"])]
        correct = g[g["status"] == "correct"]
        ambiguous = g[g["status"] == "ambiguous"]

        wrong_frames = sorted(wrong["frame"].unique().tolist())
        deviation_by_frame = {}
        if wrong_frames:
            mo, lo = load_csv(str(trial_path))
            mc, lc = load_csv(str(corrected_path))
            for f in wrong_frames:
                deltas = val.palm_triangle_bone_deltas(mo, lo, mc, lc, int(f))
                deviation_by_frame[f] = max(deltas.values()) if deltas else np.nan

        wrong_annotated = wrong.copy()
        wrong_annotated["anchor_deviation_mm"] = wrong_annotated["frame"].map(deviation_by_frame)
        small = (wrong_annotated[wrong_annotated["anchor_deviation_mm"] <= deviation_threshold_mm]
                .drop_duplicates("frame").sort_values("anchor_deviation_mm"))
        large = (wrong_annotated[wrong_annotated["anchor_deviation_mm"] > deviation_threshold_mm]
                .drop_duplicates("frame").sort_values("anchor_deviation_mm", ascending=False))

        def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
            return df if len(df) <= n else df.iloc[rng.choice(len(df), size=n, replace=False)]

        picks = []
        for df, n, cat in [
            (small, QUOTAS["wrong_small_anchor_dev"], "wrong_small_anchor_dev"),
            (large, QUOTAS["wrong_large_anchor_dev"], "wrong_large_anchor_dev"),
            (correct.drop_duplicates("frame"), QUOTAS["correct"], "correct"),
            (ambiguous.drop_duplicates("frame"), QUOTAS["ambiguous"], "ambiguous"),
        ]:
            if df.empty:
                continue
            chosen = sample(df, n)
            for _, r in chosen.iterrows():
                picks.append({
                    "participant": participant, "trial_id": trial_id,
                    "analysis_trial_path": str(trial_path),
                    "frame": int(r["frame"]), "marker_name": r["marker_name"],
                    "status": r["status"],
                    "anchor_deviation_mm": r.get("anchor_deviation_mm", np.nan),
                    "spotcheck_category": cat,
                })
        rows.extend(picks)
        print(f"{participant}/{trial_id}: "
              f"{sum(1 for p in picks if p['spotcheck_category']=='wrong_small_anchor_dev')} small, "
              f"{sum(1 for p in picks if p['spotcheck_category']=='wrong_large_anchor_dev')} large, "
              f"{sum(1 for p in picks if p['spotcheck_category']=='correct')} correct, "
              f"{sum(1 for p in picks if p['spotcheck_category']=='ambiguous')} ambiguous")

    out = pd.DataFrame(rows, columns=["participant", "trial_id", "analysis_trial_path", "frame",
                                      "marker_name", "status", "anchor_deviation_mm",
                                      "spotcheck_category"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out)} spot-check frame(s) across "
          f"{out.groupby(['participant', 'trial_id']).ngroups} trial(s) to {out_csv}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-csv", default=str(ANNOTATIONS_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--deviation-threshold-mm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(Path(args.annotations_csv), Path(args.out_csv), args.deviation_threshold_mm, args.seed)


if __name__ == "__main__":
    main()
