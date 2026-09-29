#!/usr/bin/env python3
"""Run the single-trial agreement analysis across every trial that has a
manually-labelled export, then roll the per-marker reliability numbers up
across trials/participants.

This is the follow-on step called for in
``results/shared/agreement/*/report.md``'s "Overall assessment": a single
trial can only show a marker is reliable *in that trial*; identifying a
marker as consistently reliable needs the same analysis repeated across
trials and subjects, compared side by side. Still measurement only -- no
new relabelling or correction, and the cross-trial rollup is descriptive
(mean/min/max/std of each trial's own numbers), not a fitted model.

Usage
-----
python scripts/shared/aggregate_manual_agreement.py
python scripts/shared/aggregate_manual_agreement.py --out-dir results/shared/agreement/cross_trial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core import dataset as ds             # noqa: E402
from vicon2mano.core import agreement as ag           # noqa: E402
import analyze_manual_agreement as single              # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "results" / "shared" / "agreement" / "cross_trial"


def write_report(out_dir: Path, all_reliability: pd.DataFrame,
                  cross_summary: pd.DataFrame, trials: list[tuple[str, str, Path, Path]]) -> None:
    lines = []
    lines.append("# Cross-trial marker reliability rollup\n")
    n_trials = len(trials)
    n_participants = len({p for p, *_ in trials})
    lines.append(f"- Trials analysed: {n_trials} ({n_participants} participant(s))")
    for p, t, vicon_csv, manual_csv in trials:
        lines.append(f"  - {p} / {t} (`{vicon_csv.name}` vs. `{manual_csv.name}`)")
    lines.append("")

    if n_trials < 2:
        lines.append("**Only one trial has a manually-labelled export right now, so "
                     "every between-trial statistic below (`*_std`, min==max) is "
                     "vacuous by construction -- this rollup exists so the pipeline "
                     "is ready the moment more trials are hand-labelled, not because "
                     "one trial supports a cross-trial conclusion yet. Per CLAUDE.md: "
                     "a few hundred hand-labelled frames spanning poses, across more "
                     "trials, is the highest-value thing to add next.**\n")

    lines.append("## Per-marker reliability, averaged across trials\n")
    lines.append("`*_mean`/`*_min`/`*_max`/`*_std` are computed over each trial's own "
                 "per-marker `overall_usable_pct` etc. (see the single-trial report "
                 "for what each of those means) -- not re-derived from pooled raw "
                 "frames, so a trial with more frames does not silently dominate a "
                 "trial with fewer.\n")
    cols = ["marker", "n_trials", "availability_pct_mean", "availability_pct_std",
            "label_retention_pct_valid_mean", "label_retention_pct_valid_std",
            "overall_usable_pct_mean", "overall_usable_pct_min", "overall_usable_pct_max",
            "total_event_count", "max_gap_duration"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in cross_summary.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(str(v) if c in ("marker", "n_trials", "total_event_count",
                                         "max_gap_duration")
                         else ("nan" if pd.isna(v) else f"{v:.2f}"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-trial detail\n")
    lines.append("Full per-marker table for every trial (long format); see each "
                 "trial's own `report.md` for the fuller diagnostic breakdown "
                 "(events, gaps, per-marker error) this is drawn from.\n")
    detail_cols = ["participant", "trial", "marker", "availability_pct",
                  "label_retention_pct_valid", "overall_usable_pct"]
    lines.append("| " + " | ".join(detail_cols) + " |")
    lines.append("|" + "---|" * len(detail_cols))
    for _, row in all_reliability.sort_values(["marker", "participant", "trial"]).iterrows():
        cells = [str(row[c]) if c in ("participant", "trial", "marker")
                 else f"{row[c]:.2f}" for c in detail_cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Limitations\n")
    lines.append("- Every limitation in each trial's own `report.md` still applies here.")
    lines.append("- This rollup is descriptive (mean/min/max/std of each trial's own "
                 "numbers), not a fitted or weighted model of reliability.")
    lines.append("- A marker with few trials contributes an unreliable mean/std just as "
                 "much as one with many -- `n_trials` is shown so this isn't hidden.")
    lines.append("- This does not define a reliable/unreliable classification threshold; "
                 "that was explicitly deferred to a later step.")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path, thresholds_mm: list[float], jump_mad_multiplier: float,
        jump_floor_mm: float, per_trial_plots: bool, per_trial_reports: bool) -> None:
    trials = ds.find_manually_labelled_trials()
    if not trials:
        raise SystemExit(f"No manually-labelled exports found under {ds.DATA_ROOT}")

    print(f"Found {len(trials)} trial(s) with a manually-labelled export:")
    for p, t, vicon_csv, manual_csv in trials:
        print(f"  {p} / {t}")

    reliability_rows = []
    for participant, trial_stem, vicon_csv, manual_csv in trials:
        trial_out_dir = out_dir.parent / f"{participant}_{trial_stem}"
        print(f"\n=== {participant} / {trial_stem} ===")
        reliability = single.run(
            participant, trial_stem, vicon_csv, manual_csv, trial_out_dir,
            thresholds_mm, jump_mad_multiplier, jump_floor_mm,
            per_trial_plots, per_trial_reports)
        tagged = reliability.copy()
        tagged.insert(0, "trial", trial_stem)
        tagged.insert(0, "participant", participant)
        reliability_rows.append(tagged)

    all_reliability = pd.concat(reliability_rows, ignore_index=True)
    cross_summary = ag.cross_trial_summary(all_reliability)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_reliability.to_csv(out_dir / "all_trials_reliability.csv", index=False)
    cross_summary.to_csv(out_dir / "marker_reliability_cross_trial.csv", index=False)
    write_report(out_dir, all_reliability, cross_summary, trials)

    print(f"\nSaved cross-trial rollup to {out_dir}")
    worst = cross_summary.head(5)[["marker", "overall_usable_pct_mean", "n_trials"]]
    print("Worst 5 markers by mean overall_usable_pct:")
    for _, row in worst.iterrows():
        print(f"  {row['marker']}: {row['overall_usable_pct_mean']:.1f}% "
              f"(n_trials={row['n_trials']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--thresholds-mm", default="1,2,5,10")
    ap.add_argument("--jump-mad-multiplier", type=float, default=6.0)
    ap.add_argument("--jump-floor-mm", type=float, default=2.0)
    ap.add_argument("--no-per-trial-plots", action="store_true",
                    help="skip generating each trial's own figures (faster; the "
                         "cross-trial rollup doesn't need them)")
    ap.add_argument("--no-per-trial-reports", action="store_true",
                    help="skip writing each trial's own report.md")
    args = ap.parse_args(argv)

    thresholds = [float(t) for t in args.thresholds_mm.split(",") if t.strip()]
    run(Path(args.out_dir), thresholds, args.jump_mad_multiplier, args.jump_floor_mm,
        not args.no_per_trial_plots, not args.no_per_trial_reports)


if __name__ == "__main__":
    main()
