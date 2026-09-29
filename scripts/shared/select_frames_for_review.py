#!/usr/bin/env python3
"""Select 500 frames across all 66 trials for manual review (Vicon/MATLAB),
plus a +/-5 frame window around each, per the sampling plan:

- 200 random frames, pooled across all trials (longer trials contribute
  proportionally more, since the pool is frame-indices not trial-indices).
- 150 frames with the largest geometric-self-consistency anomalies
  (results/shared/geometric_consistency/*/events.csv, ranked by
  deviation_magnitude_mm, using each event's peak_frame).
- 75 frames with the poorest trajectory availability (most markers
  simultaneously missing that frame, reconstructed from each trial's own
  gaps.csv -- no need to reload the raw marker CSVs).
- 50 frames where geometry and availability "disagree": a suspicious
  geometric event whose peak is NOT adjacent to a Vicon gap
  (`adjacent_missing_vicon == False` in events.csv) -- i.e. geometry flags
  a problem that missing data does not explain, excluding frames already
  taken for the top-150-anomaly category.
- 25 random frames from trials whose reference_source_type is "static" or
  "calibration" (results/shared/trial_overview.csv).

Duplicates across categories are resolved by keeping the first category
that claimed a frame and topping up that category's quota from further
down its own ranked/random pool -- see `select_category`.

Requires results/shared/trial_overview.csv (build_trial_overview.py) and
the full results/shared/geometric_consistency/*/ output
(geometric_self_consistency.py --all) to already exist.

Usage
-----
python scripts/shared/select_frames_for_review.py
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

from vicon2mano.core import dataset as ds                                     # noqa: E402

GEOM_DIR = REPO_ROOT / "results" / "shared" / "geometric_consistency"
OVERVIEW_CSV = REPO_ROOT / "results" / "shared" / "trial_overview.csv"
OUT_DIR = REPO_ROOT / "results" / "shared" / "frame_sample"

WINDOW_RADIUS = 5
QUOTAS = {
    "random": 200,
    "largest_anomaly": 150,
    "poor_availability": 75,
    "geometry_availability_disagreement": 50,
    "static_or_calibration": 25,
}


def trial_dir_name(participant: str, trial_id: str) -> str:
    return f"{participant}_{ds.slug(trial_id)}"


def load_all_events(overview: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in overview.iterrows():
        d = GEOM_DIR / trial_dir_name(row["participant"], row["trial_id"]) / "events.csv"
        if not d.exists():
            continue
        ev = pd.read_csv(d)
        if len(ev):
            frames.append(ev)
    if not frames:
        return pd.DataFrame(columns=["subject_id", "trial_id", "peak_frame",
                                     "deviation_magnitude_mm", "adjacent_missing_vicon"])
    return pd.concat(frames, ignore_index=True)


def load_all_gaps(overview: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for _, row in overview.iterrows():
        d = GEOM_DIR / trial_dir_name(row["participant"], row["trial_id"]) / "gaps.csv"
        if d.exists():
            out[(row["participant"], row["trial_id"])] = pd.read_csv(d)
    return out


def per_frame_missing_count(gaps: pd.DataFrame, n_frames: int) -> np.ndarray:
    """How many markers are simultaneously missing at each frame, built
    from gap start/end runs (interval accumulation) rather than reloading
    the raw marker CSV."""
    diff = np.zeros(n_frames + 1, dtype=int)
    for _, row in gaps.iterrows():
        s, e = int(row["start_frame"]), int(row["end_frame"])
        diff[s] += 1
        if e + 1 <= n_frames:
            diff[e + 1] -= 1
    return np.cumsum(diff)[:n_frames]


def worst_availability_frames(overview: pd.DataFrame, gaps_by_trial: dict,
                               n_frames_by_trial: dict, top_k: int) -> pd.DataFrame:
    rows = []
    for (participant, trial_id), gaps in gaps_by_trial.items():
        n_frames = n_frames_by_trial[(participant, trial_id)]
        if n_frames == 0 or gaps.empty:
            continue
        counts = per_frame_missing_count(gaps, n_frames)
        # Only frames with at least one missing marker are candidates --
        # a trial with zero gaps contributes nothing here, correctly.
        candidate_frames = np.flatnonzero(counts > 0)
        if candidate_frames.size == 0:
            continue
        top_local = candidate_frames[np.argsort(-counts[candidate_frames])[:top_k]]
        for f in top_local:
            rows.append({"participant": participant, "trial_id": trial_id,
                        "frame": int(f), "n_markers_missing": int(counts[f])})
    if not rows:
        return pd.DataFrame(columns=["participant", "trial_id", "frame", "n_markers_missing"])
    return (pd.DataFrame(rows)
            .sort_values("n_markers_missing", ascending=False)
            .reset_index(drop=True))


def select_category(candidates: pd.DataFrame, quota: int, taken: set,
                     category: str, reason_col: str | None = None) -> pd.DataFrame:
    """Walk ``candidates`` (already ranked/shuffled by the caller) in order,
    taking the first ``quota`` frames not already in ``taken`` -- this is
    the "keep it once, fill the remaining quota from that category" rule:
    a frame skipped here because another category already claimed it does
    not shrink this category's final count, since the walk simply continues
    to the next candidate in the same ranked list.
    """
    rows = []
    for _, row in candidates.iterrows():
        if len(rows) >= quota:
            break
        key = (row["participant"], row["trial_id"], int(row["frame"]))
        if key in taken:
            continue
        taken.add(key)
        out = {"participant": row["participant"], "trial_id": row["trial_id"],
               "frame": int(row["frame"]), "category": category}
        if reason_col and reason_col in row:
            out["selection_metric"] = row[reason_col]
        rows.append(out)
    return pd.DataFrame(rows, columns=["participant", "trial_id", "frame", "category", "selection_metric"])


def build_windows(anchors: pd.DataFrame, n_frames_by_trial: dict) -> pd.DataFrame:
    """+/-WINDOW_RADIUS frames around each anchor, clipped to the trial's
    valid frame range, deduplicated so an overlap between two nearby
    anchors' windows is listed once (with both anchors recorded)."""
    rows: dict[tuple, dict] = {}
    for _, row in anchors.iterrows():
        participant, trial_id, anchor_frame = row["participant"], row["trial_id"], int(row["frame"])
        n_frames = n_frames_by_trial.get((participant, trial_id))
        if n_frames is None:
            continue
        lo = max(0, anchor_frame - WINDOW_RADIUS)
        hi = min(n_frames - 1, anchor_frame + WINDOW_RADIUS)
        for f in range(lo, hi + 1):
            key = (participant, trial_id, f)
            if key not in rows:
                rows[key] = {
                    "participant": participant, "trial_id": trial_id, "frame": f,
                    "is_anchor": f == anchor_frame,
                    "anchor_frames": {anchor_frame},
                    "categories": {row["category"]},
                }
            else:
                rows[key]["is_anchor"] = rows[key]["is_anchor"] or (f == anchor_frame)
                rows[key]["anchor_frames"].add(anchor_frame)
                rows[key]["categories"].add(row["category"])

    out = []
    for v in rows.values():
        out.append({
            "participant": v["participant"], "trial_id": v["trial_id"], "frame": v["frame"],
            "is_anchor": v["is_anchor"],
            "anchor_frames": ";".join(str(f) for f in sorted(v["anchor_frames"])),
            "categories": ";".join(sorted(v["categories"])),
        })
    return (pd.DataFrame(out)
            .sort_values(["participant", "trial_id", "frame"])
            .reset_index(drop=True))


def run(seed: int, out_dir: Path) -> None:
    if not OVERVIEW_CSV.exists():
        raise SystemExit(f"{OVERVIEW_CSV} not found -- run scripts/shared/build_trial_overview.py first")
    overview = pd.read_csv(OVERVIEW_CSV)

    trial_summaries = []
    for _, row in overview.iterrows():
        d = GEOM_DIR / trial_dir_name(row["participant"], row["trial_id"]) / "trial_summary.csv"
        if d.exists():
            trial_summaries.append(pd.read_csv(d))
    trial_summary = pd.concat(trial_summaries, ignore_index=True)
    n_frames_by_trial = {(r["subject_id"], r["trial_id"]): int(r["n_frames"])
                        for _, r in trial_summary.iterrows()}

    rng = np.random.default_rng(seed)
    taken: set = set()
    picks = []

    # 1. random, pooled across all trials proportional to trial length.
    pool = [(p, t, f) for (p, t), n in n_frames_by_trial.items() for f in range(n)]
    rng.shuffle(pool)
    random_candidates = pd.DataFrame(pool, columns=["participant", "trial_id", "frame"])
    picks.append(select_category(random_candidates, QUOTAS["random"], taken, "random"))

    # 2. largest geometric anomalies.
    events = load_all_events(overview)
    events = events.rename(columns={"subject_id": "participant", "peak_frame": "frame"})
    ranked_anomalies = events.sort_values("deviation_magnitude_mm", ascending=False)
    picks.append(select_category(ranked_anomalies, QUOTAS["largest_anomaly"], taken,
                                 "largest_anomaly", reason_col="deviation_magnitude_mm"))

    # 3. poorest trajectory availability.
    gaps_by_trial = load_all_gaps(overview)
    worst_avail = worst_availability_frames(overview, gaps_by_trial, n_frames_by_trial,
                                            top_k=QUOTAS["poor_availability"] * 3)
    picks.append(select_category(worst_avail, QUOTAS["poor_availability"], taken,
                                 "poor_availability", reason_col="n_markers_missing"))

    # 4. geometry/availability disagreement: suspicious but not gap-adjacent.
    disagreement = ranked_anomalies[~ranked_anomalies["adjacent_missing_vicon"]]
    picks.append(select_category(disagreement, QUOTAS["geometry_availability_disagreement"],
                                 taken, "geometry_availability_disagreement",
                                 reason_col="deviation_magnitude_mm"))

    # 5. static/calibration reference trials.
    static_trials = overview[overview["reference_source_type"].isin(["static", "calibration"])]
    static_pool = [(p, t, f) for p, t in zip(static_trials["participant"], static_trials["trial_id"])
                   for f in range(n_frames_by_trial.get((p, t), 0))]
    rng.shuffle(static_pool)
    static_candidates = pd.DataFrame(static_pool, columns=["participant", "trial_id", "frame"])
    picks.append(select_category(static_candidates, QUOTAS["static_or_calibration"], taken,
                                 "static_or_calibration"))

    anchors = pd.concat(picks, ignore_index=True)
    for cat, quota in QUOTAS.items():
        got = (anchors["category"] == cat).sum()
        if got < quota:
            print(f"[warn] category {cat!r}: only {got}/{quota} frames available "
                  f"(pool exhausted after deduplication)")

    windows = build_windows(anchors, n_frames_by_trial)

    out_dir.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out_dir / "selected_anchors.csv", index=False)
    windows.to_csv(out_dir / "selected_windows.csv", index=False)

    write_report(out_dir, anchors, windows)
    print(f"Selected {len(anchors)} anchor frame(s) -> {len(windows)} window frame(s) "
          f"(with overlap dedup)")
    print(f"Saved to {out_dir}")


def write_report(out_dir: Path, anchors: pd.DataFrame, windows: pd.DataFrame) -> None:
    lines = []
    lines.append("# Frame sample for manual review\n")
    lines.append(f"- Anchor frames selected: {len(anchors)}")
    lines.append(f"- Window radius: +/-{WINDOW_RADIUS} frames")
    lines.append(f"- Total distinct frames after window expansion and dedup: {len(windows)}\n")
    lines.append("## By category\n")
    lines.append("| category | quota | selected |")
    lines.append("|---|---|---|")
    for cat, quota in QUOTAS.items():
        got = int((anchors["category"] == cat).sum())
        lines.append(f"| {cat} | {quota} | {got} |")
    lines.append("")
    lines.append("## Category definitions\n")
    lines.append("- **random**: uniformly sampled from the pooled (trial, frame) space "
                 "across all 66 trials (longer trials contribute proportionally more).")
    lines.append("- **largest_anomaly**: the highest `deviation_magnitude_mm` geometric "
                 "self-consistency events, at each event's peak frame.")
    lines.append("- **poor_availability**: frames with the most markers simultaneously "
                 "missing (reconstructed from each trial's gap table).")
    lines.append("- **geometry_availability_disagreement**: suspicious geometric events "
                 "whose peak is NOT adjacent to a Vicon gap -- i.e. geometry flags a "
                 "problem that missing data does not explain -- excluding frames already "
                 "claimed by `largest_anomaly`.")
    lines.append("- **static_or_calibration**: random frames drawn only from trials whose "
                 "`reference_source_type` is `static` or `calibration`.\n")
    lines.append("## Files\n")
    lines.append("- `selected_anchors.csv`: one row per anchor frame (participant, trial_id, "
                 "frame, category, selection_metric).")
    lines.append("- `selected_windows.csv`: every frame in every anchor's +/-5 window, "
                 "deduplicated across overlapping windows, with `is_anchor` and the "
                 "originating `anchor_frames`/`categories` for traceability.\n")
    lines.append("## Limitations\n")
    lines.append("- This selection is drawn from `assessment_type = geometric_self_consistency` "
                 "only; it does not use, and is not informed by, the separate manual-label "
                 "agreement assessment (P10/Trial2_handsonly).")
    lines.append("- A frame's category reflects why it was picked, not a verdict -- manual "
                 "review is what determines whether it is actually a problem.")
    lines.append("- `poor_availability` and `geometry_availability_disagreement` are built "
                 "from each trial's own gap/event tables (see "
                 "`results/shared/geometric_consistency/`), which are themselves diagnostic, "
                 "not ground truth.")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    run(args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
