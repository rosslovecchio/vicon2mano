#!/usr/bin/env python3
"""Phase 1B: label and frame-quality assessment across the whole dataset.

Classifies every marker-frame in every trial into exactly one of
vicon2mano.core.quality.CATEGORIES, from 6 SEPARATELY-measured signals that
are never combined into one score (see core/quality.py's module docstring
for the precedence rule and why it is a categorical label, not a score):

1. manual agreement with reviewed labels (manual_annotations.csv -- only
   ~500 of the ~58 million marker-frames in this dataset have this)
2. geometric self-consistency (geometric_self_consistency's events.csv)
3. anchor-frame validity (palm triangle present and non-degenerate)
4. trajectory continuity (core.continuity -- reported as its own column,
   not folded into the category: none of the requested category names
   correspond to it)
5. suspicious swap events (same source as #2; event_type/associated_bone
   carried as companion columns for whichever frames have them)
6. ambiguous cases (manual review's automatic anchor-occlusion verdict)

Per-trial full-resolution detail (one row per marker-frame, matching the
existing per_frame_comparison.csv precedent from
scripts/shared/analyze_manual_agreement.py) is written to
results/shared/quality/<participant>_<trial>/marker_frame_classification.csv.
The dataset only has ONE requested top-level file,
raw_vicon_quality_summary.csv -- at (participant, trial, marker) grain with
a count per category, since the full table is tens of millions of rows and
is not useful flattened into one file; the per-trial files carry the
literal frame-level detail if needed.

Also writes: difficult_trials.csv, excluded_or_ambiguous_frames.csv
(the small manually-reviewed ambiguous set, plus anchor-invalid VOLUME per
trial -- not the literal anchor-invalid frame list, for the same size
reason), and anchor_and_reference_bone_diagnostics.csv (anchor validity
across EVERY frame of every trial, extending the validation audit's
sampled-frames-only version).

Usage
-----
python scripts/shared/build_raw_vicon_quality_summary.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv                    # noqa: E402
from vicon2mano.core import dataset as ds                      # noqa: E402
from vicon2mano.core import continuity as cont                 # noqa: E402
from vicon2mano.core import validation as val                  # noqa: E402
from vicon2mano.core import quality as q                       # noqa: E402

GEOM_DIR = REPO_ROOT / "results" / "shared" / "geometric_consistency"
ANNOTATIONS_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "manual_annotations.csv"
OUT_DIR = REPO_ROOT / "results" / "shared" / "quality"
ANCHOR_NAMES = val.DEFAULT_ANCHOR_NAMES

TRIAL_KEYS = ["Trial 1 Hands only", "Trial 1 HOI", "Trial2 Hands only", "Trial2 HOI"]
DIFFICULTY_CATEGORIES = ("observed_suspicious", "anchor_invalid", "label_manually_wrong",
                         "ghost_candidate")


def trial_dir(participant: str, trial_id: str) -> str:
    return f"{participant}_{ds.slug(trial_id)}"


def anchor_validity(markers: np.ndarray, labels: list[str],
                     degenerate_threshold: float) -> np.ndarray:
    """(T,) bool: palm anchor triple present and non-degenerate that frame.
    Same sine-of-angle degeneracy measure as the validation audit's
    anchor_degeneracy, generalised to every frame rather than just the
    sampled ones."""
    base = {l.split(":")[-1]: i for i, l in enumerate(labels)}
    T = markers.shape[0]
    if not all(n in base for n in ANCHOR_NAMES):
        return np.zeros(T, dtype=bool)
    o = markers[:, base[ANCHOR_NAMES[0]]]
    xp = markers[:, base[ANCHOR_NAMES[1]]]
    yp = markers[:, base[ANCHOR_NAMES[2]]]
    x, y = xp - o, yp - o
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sine = np.linalg.norm(np.cross(x, y), axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    return finite & (sine >= degenerate_threshold)


def expand_events_to_bool(events: pd.DataFrame, marker_base_names: list[str], T: int) -> dict:
    """{base_marker_name: (T,) bool} -- True where that marker falls inside
    any geometric-self-consistency event (full [start_frame, end_frame]
    span), via interval accumulation rather than a per-frame Python loop."""
    out = {m: np.zeros(T, dtype=bool) for m in marker_base_names}
    if events is None or events.empty:
        return out
    ev = events.copy()
    ev["base_marker"] = ev["marker"].str.split(":").str[-1]
    for m, g in ev.groupby("base_marker"):
        if m not in out:
            continue
        diff = np.zeros(T + 1, dtype=int)
        for _, r in g.iterrows():
            s, e = int(r["start_frame"]), min(int(r["end_frame"]), T - 1)
            diff[s] += 1
            diff[e + 1] -= 1
        out[m] = np.cumsum(diff)[:T] > 0
    return out


def event_detail_lookup(events: pd.DataFrame) -> dict:
    """{(base_marker_name, frame): (event_type, associated_bone)} for every
    frame actually covered by an event -- used to carry the "suspicious
    swap event" detail columns without re-deriving them."""
    out = {}
    if events is None or events.empty:
        return out
    for _, r in events.iterrows():
        base = r["marker"].split(":")[-1]
        for f in range(int(r["start_frame"]), int(r["end_frame"]) + 1):
            out[(base, f)] = (r["event_type"], r["associated_bone"])
    return out


def run(out_dir: Path, degenerate_threshold: float, write_per_trial_detail: bool) -> None:
    sessions = ds.load_trial_sessions(ds.REF_CSV)
    participants = sorted(sessions, key=ds.participant_sort_key)

    annotations = pd.read_csv(ANNOTATIONS_CSV) if ANNOTATIONS_CSV.exists() else pd.DataFrame(
        columns=["participant", "trial_id", "frame", "marker_name", "status"])

    summary_rows = []
    anchor_diag_rows = []
    excluded_rows = []
    t_start = time.time()
    n_trials_done = 0

    for participant in participants:
        have = sessions[participant]
        for trial_id in TRIAL_KEYS:
            if trial_id not in have:
                continue
            trial_path, *_ = ds.find_trial_csv(participant, trial_id)
            if trial_path is None:
                continue

            markers, labels = load_csv(str(trial_path))
            T, N, _ = markers.shape
            base_names = [l.split(":")[-1] for l in labels]

            is_missing = ~np.isfinite(markers).all(axis=2)              # (T, N)
            anchor_valid_t = anchor_validity(markers, labels, degenerate_threshold)  # (T,)
            continuity_suspicious = cont.flag_discontinuities(markers)   # (T, N)

            events_path = GEOM_DIR / trial_dir(participant, trial_id) / "events.csv"
            events = pd.read_csv(events_path) if events_path.exists() else None
            geometric_suspicious_by_marker = expand_events_to_bool(events, base_names, T)
            event_detail = event_detail_lookup(events)

            trial_annot = annotations[(annotations["participant"] == participant) &
                                      (annotations["trial_id"] == trial_id)].copy()
            trial_annot["base_marker"] = trial_annot["marker_name"].str.split(":").str[-1]
            # Sparse overlay: manual review covers at most a few hundred
            # marker-frames per trial out of up to ~1.4M -- group once per
            # marker instead of touching every frame to look it up.
            manual_by_marker: dict[str, dict[int, str]] = {
                m: dict(zip(g["frame"], g["status"]))
                for m, g in trial_annot.groupby("base_marker")
            }

            detail_rows = [] if write_per_trial_detail else None
            counts_per_marker = {m: {c: 0 for c in q.CATEGORIES} for m in base_names}

            for mi, m in enumerate(base_names):
                missing_m = is_missing[:, mi]
                suspicious_m = geometric_suspicious_by_marker.get(m, np.zeros(T, dtype=bool))
                manual_frames_m = manual_by_marker.get(m, {})
                manual_cat_m = np.full(T, "", dtype=object)
                for f, status in manual_frames_m.items():
                    manual_cat_m[f] = q.resolve_manual_category(status)

                cats = q.classify_vectorized(missing_m, anchor_valid_t, suspicious_m, manual_cat_m)
                cat_counts = pd.Series(cats).value_counts()
                for c in q.CATEGORIES:
                    counts_per_marker[m][c] = int(cat_counts.get(c, 0))

                if detail_rows is not None:
                    # Thin the detail file: keep every frame that isn't the
                    # least-informative default (plausible/missing with no
                    # continuity flag either) -- vectorised selection, no
                    # per-frame Python loop over the full T.
                    keep = continuity_suspicious[:, mi] | ~np.isin(cats, ("observed_plausible", "missing"))
                    for f in np.flatnonzero(keep):
                        f = int(f)
                        ev_type, ev_bone = event_detail.get((m, f), (None, None))
                        detail_rows.append((participant, trial_id, m, f, bool(missing_m[f]),
                                           bool(anchor_valid_t[f]), bool(suspicious_m[f]),
                                           ev_type, ev_bone, bool(continuity_suspicious[f, mi]),
                                           manual_frames_m.get(f), cats[f]))

                if counts_per_marker[m]["ambiguous"] or counts_per_marker[m]["anchor_invalid"]:
                    excluded_rows.append({
                        "participant": participant, "trial_id": trial_id, "marker": m,
                        "n_ambiguous": counts_per_marker[m]["ambiguous"],
                        "n_anchor_invalid": counts_per_marker[m]["anchor_invalid"],
                    })

                row = {"participant": participant, "trial_id": trial_id, "marker": m, "n_frames": T}
                row.update(counts_per_marker[m])
                row["pct_continuity_suspicious"] = 100.0 * continuity_suspicious[:, mi].mean()
                summary_rows.append(row)

            anchor_diag_rows.append({
                "participant": participant, "trial_id": trial_id, "n_frames": T,
                "has_anchor_triple": all(n in base_names for n in ANCHOR_NAMES),
                "pct_anchor_valid": 100.0 * anchor_valid_t.mean(),
                "n_anchor_invalid_frames": int((~anchor_valid_t).sum()),
            })

            if detail_rows is not None:
                d = out_dir / trial_dir(participant, trial_id)
                d.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(detail_rows, columns=[
                    "participant", "trial_id", "marker", "frame", "is_missing", "anchor_valid",
                    "geometric_suspicious", "event_type", "associated_bone",
                    "continuity_suspicious", "manual_status", "category",
                ]).to_csv(d / "marker_frame_classification.csv", index=False)

            n_trials_done += 1
            print(f"[{n_trials_done}] {participant}/{trial_id}: {T} frames x {N} markers "
                  f"({time.time() - t_start:.0f}s elapsed)")

    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "assessment_type", q.ASSESSMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "raw_vicon_quality_summary.csv", index=False)

    anchor_diag = pd.DataFrame(anchor_diag_rows).sort_values("pct_anchor_valid")
    anchor_diag.to_csv(out_dir / "anchor_and_reference_bone_diagnostics.csv", index=False)

    difficulty = (summary.groupby(["participant", "trial_id"])
                 .apply(lambda g: pd.Series({
                     "n_frames_total": g["n_frames"].iloc[0] * len(g),
                     "n_difficult": g[list(DIFFICULTY_CATEGORIES)].to_numpy().sum(),
                 }), include_groups=False)
                 .reset_index())
    difficulty["pct_difficult"] = 100.0 * difficulty["n_difficult"] / difficulty["n_frames_total"]
    difficulty = difficulty.sort_values("pct_difficult", ascending=False)
    difficulty.to_csv(out_dir / "difficult_trials.csv", index=False)

    ambiguous_rows = annotations[annotations["status"] == "ambiguous"][
        ["participant", "trial_id", "frame", "marker_name", "status"]]
    ambiguous_rows.to_csv(out_dir / "excluded_or_ambiguous_frames.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(out_dir / "anchor_invalid_volume_by_marker.csv", index=False)

    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs in {out_dir}:")
    print(f"  raw_vicon_quality_summary.csv: {len(summary)} (participant, trial, marker) rows")
    print(f"  difficult_trials.csv: {len(difficulty)} trials, worst: "
          f"{difficulty.iloc[0]['participant']}/{difficulty.iloc[0]['trial_id']} "
          f"({difficulty.iloc[0]['pct_difficult']:.1f}%)")
    print(f"  anchor_and_reference_bone_diagnostics.csv: {len(anchor_diag)} trials")
    print(f"  excluded_or_ambiguous_frames.csv: {len(ambiguous_rows)} manually-confirmed ambiguous rows")

    total_plausible = int(summary["observed_plausible"].sum())
    total_marker_frames = int(summary["n_frames"].sum())  # sum of T over every (trial, marker) row
    print(f"\n  trusted training subset (observed_plausible, unreviewed+anchor-valid+non-suspicious): "
          f"{total_plausible}/{total_marker_frames} marker-frames "
          f"({100*total_plausible/total_marker_frames:.1f}%)")
    print(f"  manually validated test subset (any manual review): {len(annotations)} marker-frames")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--degenerate-threshold", type=float, default=0.1)
    ap.add_argument("--no-per-trial-detail", action="store_true",
                    help="skip writing marker_frame_classification.csv per trial "
                         "(faster; the aggregate summary is still produced)")
    args = ap.parse_args(argv)
    run(Path(args.out_dir), args.degenerate_threshold, not args.no_per_trial_detail)


if __name__ == "__main__":
    main()
