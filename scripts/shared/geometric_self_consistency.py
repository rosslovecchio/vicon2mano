#!/usr/bin/env python3
"""Geometric self-consistency report: does a trial's own bone-length
geometry hang together, checked against nothing but itself.

**assessment_type = geometric_self_consistency**

This is NOT the Vicon-vs-manual-label agreement analysis
(``scripts/shared/analyze_manual_agreement.py`` /
``vicon2mano.core.agreement``) and its numbers are never pooled with that
one -- see ``vicon2mano/core/geometric_consistency.py`` for the full
distinction. In short: this can flag a bone length as anatomically
implausible, but it has no independent identity check, so it can detect
implausible trajectories without being able to prove which marker's label
is actually correct. Read the "What this can and cannot tell you" section
of every report this script writes before treating a low score here as
"mislabelled".

Only the P10/Trial2_handsonly manual-label agreement report is referenced
from the global report here, and only as a separate validation point (does
this geometric method roughly agree with the one trial that has a human
check) -- never averaged or combined with the geometric numbers themselves.

Bone topology: ``quality_cascade.infer_bones`` (finger chains + rigid
palm/forearm plate pairs). Reference lengths + deviations:
``vicon2mano.core.bones.consensus_bone_lengths`` / ``bone_error`` -- moved
there from ``scripts/gmm/relabel_trial.py`` specifically so this script
could reuse the exact, already-validated implementation instead of writing
a second one (see that module's docstring for the reuse rationale and
CLAUDE.md's 2026-09-11/12 notes for how it was arrived at).

Usage
-----
python scripts/shared/geometric_self_consistency.py --participant P10 --trial "Trial2 Hands only"
python scripts/shared/geometric_self_consistency.py --all
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv                                   # noqa: E402
from vicon2mano.core import dataset as ds                                     # noqa: E402
from vicon2mano.core import geometric_consistency as gc                       # noqa: E402
from vicon2mano.core.bones import consensus_bone_lengths                      # noqa: E402
from vicon2mano.strategies.cascade.quality_cascade import infer_bones         # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "shared" / "geometric_consistency"

# Trial names exactly as manual_frames.csv's "Session" column spells them --
# same convention scripts/mano/relabel_trial.py uses for "every trial in
# the dataset", reused rather than inventing a second way to enumerate
# trials.
TRIAL_KEYS = ["Trial 1 Hands only", "Trial 1 HOI", "Trial2 Hands only", "Trial2 HOI"]

WHAT_THIS_CANNOT_TELL_YOU = (
    "**What this can and cannot tell you**: this report scores whether each "
    "frame's bone lengths are consistent with this trial's own learned "
    "reference geometry (`consensus_bone_lengths`). A large deviation means "
    "the trajectory is anatomically implausible given this trial's own "
    "geometry -- it does NOT mean a specific marker's *label* is wrong, and "
    "a clean bone length does NOT prove the label is right. A pair of "
    "markers whose exchange happens to preserve every bone length touching "
    "them (most plausible on a rigid plate, where markers are "
    "near-symmetric) would pass through this check completely undetected. "
    "This is `assessment_type = geometric_self_consistency`, not "
    "`manual_label_agreement` -- the two are computed independently and "
    "never pooled; see `scripts/shared/analyze_manual_agreement.py` for the "
    "latter, which is the only one of the two with an actual independent "
    "identity check (a human review), and only exists for one trial today."
)

REFERENCE_SOURCE_NOTE = (
    "Static or calibration trials may intentionally serve as reference data "
    "for estimating subject-specific bone lengths. The reference trial can "
    "differ from the movement trial being assessed."
)


def analyze_trial(markers: np.ndarray, labels: list[str], *,
                   subject_id: str, trial_id: str, analysis_csv_name: str,
                   thresholds_mm: list[float], jump_mad_multiplier: float,
                   min_deviation_mm: float):
    bones = infer_bones(labels)
    bone_ij = [(b[0], b[1]) for b in bones]
    ref_lengths, inlier_frames = consensus_bone_lengths(markers, bone_ij, min_bones=3)
    reference_table = gc.bone_reference_table(markers, bones, ref_lengths, inlier_frames)

    df = gc.build_marker_frame_table(markers, labels, bones, ref_lengths,
                                     subject_id=subject_id, trial_id=trial_id)
    df = gc.add_threshold_columns(df, thresholds_mm)
    df = gc.flag_suspicious_events(df, k=jump_mad_multiplier, min_deviation_mm=min_deviation_mm)
    events = gc.geometric_events(df)
    gaps = gc.missingness_events(df, subject_id=subject_id, trial_id=trial_id)
    marker_summary = gc.per_marker_summary(df, events, gaps, thresholds_mm)
    overall = gc.trial_overall_summary(marker_summary)
    overall["subject_id"] = subject_id
    overall["trial_id"] = trial_id
    overall["n_bones"] = len(bones)
    overall["n_consensus_inlier_frames"] = len(inlier_frames)
    overall["n_frames"] = markers.shape[0]

    # Reference-source metadata: descriptive only, never a gate on whether
    # this trial gets analysed (see REFERENCE_SOURCE_NOTE). In this
    # implementation the reference bone lengths are always learned from the
    # analysis trial's own frames (self-consistency, not a separate
    # dedicated-static file), so reference_trial == analysis_trial here --
    # tracked as its own field regardless, since a future version may wire
    # in a genuinely separate reference trial (e.g. via
    # vicon2mano.core.dataset.find_static_csv) without needing every
    # downstream table to change shape.
    source_type, is_static = gc.classify_reference_source(analysis_csv_name)
    overall["analysis_trial"] = analysis_csv_name
    overall["reference_trial"] = analysis_csv_name
    overall["reference_source_type"] = source_type
    overall["reference_source_is_static"] = is_static
    overall["reference_status"] = gc.reference_status(ref_lengths)
    overall["n_valid_reference_bones"] = int(np.isfinite(ref_lengths).sum())

    return df, events, gaps, marker_summary, overall, reference_table


def make_plots(df: pd.DataFrame, marker_summary: pd.DataFrame, events: pd.DataFrame,
                out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = marker_summary.sort_values("median_deviation_mm")["marker"]
    data = [df.loc[df["marker"] == m, "deviation_mm"].dropna() for m in order]
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(order)), 5))
    try:
        ax.boxplot(data, tick_labels=order, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Bone-length deviation from consensus reference (mm)")
    ax.set_title("Geometric self-consistency: deviation by marker (outliers hidden)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(fig_dir / "deviation_by_marker_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.3 * len(order))))
    if len(events):
        marker_pos = {m: i for i, m in enumerate(order)}
        colors = {"single_frame_discontinuity": "#DD8452", "sustained_geometric_deviation": "#C44E52"}
        for _, ev in events.iterrows():
            y = marker_pos.get(ev["marker"])
            if y is None:
                continue
            ax.plot([ev["start_frame"], ev["end_frame"]], [y, y], lw=3,
                    color=colors.get(ev["event_type"], "#888888"),
                    solid_capstyle="butt")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
    ax.set_xlabel("Frame")
    ax.set_title("Suspicious geometric-deviation events by marker and location")
    fig.tight_layout()
    fig.savefig(fig_dir / "event_locations.png", dpi=150)
    plt.close(fig)


def write_trial_report(out_dir: Path, overall: dict, marker_summary: pd.DataFrame,
                        events: pd.DataFrame, gaps: pd.DataFrame,
                        reference_table: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Geometric self-consistency: {overall['subject_id']} / {overall['trial_id']}\n")
    lines.append(f"**assessment_type = {gc.ASSESSMENT_TYPE}**\n")
    lines.append(WHAT_THIS_CANNOT_TELL_YOU + "\n")
    lines.append(f"- Frames: {overall['n_frames']}, markers: {overall['n_markers']}, "
                 f"bones: {overall['n_bones']}")
    lines.append(f"- Consensus reference learned from {overall['n_consensus_inlier_frames']} "
                 f"mutually-consistent frames")
    lines.append(f"- Mean availability: {overall['mean_availability_pct']:.1f}%, "
                 f"worst marker: {overall['min_availability_pct']:.1f}%")
    lines.append(f"- Mean of per-marker median deviation: {overall['mean_median_deviation_mm']:.2f}mm, "
                 f"worst per-marker p95: {overall['max_p95_deviation_mm']:.2f}mm")
    lines.append(f"- {overall['total_suspicious_events']} suspicious event(s), "
                 f"{overall['total_affected_observations']} affected observations")
    lines.append(f"- {overall['total_vicon_gap_count']} Vicon gap(s), "
                 f"max gap duration {overall['max_gap_duration']} frames\n")

    lines.append("## Reference source\n")
    lines.append(REFERENCE_SOURCE_NOTE + "\n")
    lines.append(f"- analysis_trial: `{overall['analysis_trial']}`")
    lines.append(f"- reference_trial: `{overall['reference_trial']}`")
    lines.append(f"- reference_source_type: {overall['reference_source_type']}")
    lines.append(f"- reference_source_is_static: {overall['reference_source_is_static']}")
    lines.append(f"- reference_status: {overall['reference_status']} "
                 f"({overall['n_valid_reference_bones']}/{overall['n_bones']} bones have a "
                 f"valid reference length)")
    lines.append(f"- participant: {overall['subject_id']}\n")

    lines.append("### Reference bone lengths\n")
    lines.append("Number of valid (consensus-inlier) frames used per bone, and the "
                 "resulting reference length -- see `reference_bones.csv` for the full "
                 "table.\n")
    ref_cols = ["associated_bone", "reference_length_mm", "n_valid_frames_used"]
    lines.append("| " + " | ".join(ref_cols) + " |")
    lines.append("|" + "---|" * len(ref_cols))
    for _, row in reference_table.sort_values("associated_bone").iterrows():
        length = "nan" if pd.isna(row["reference_length_mm"]) else f"{row['reference_length_mm']:.2f}"
        lines.append(f"| {row['associated_bone']} | {length} | {row['n_valid_frames_used']} |")
    lines.append("")

    lines.append("## Per-marker summary\n")
    cols = ["marker", "availability_pct", "median_deviation_mm", "p95_deviation_mm",
            "max_deviation_mm", "suspicious_event_count", "affected_observations",
            "vicon_gap_count", "max_gap_duration"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in marker_summary.iterrows():
        cells = [str(row[c]) if c in ("marker", "suspicious_event_count", "affected_observations",
                                      "vicon_gap_count", "max_gap_duration")
                 else f"{row[c]:.2f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Suspicious events\n")
    if len(events):
        ev_cols = ["marker", "start_frame", "end_frame", "duration", "event_type",
                  "associated_bone", "deviation_magnitude_mm", "adjacent_missing_vicon"]
        lines.append("| " + " | ".join(ev_cols) + " |")
        lines.append("|" + "---|" * len(ev_cols))
        for _, row in events.sort_values("deviation_magnitude_mm", ascending=False).head(50).iterrows():
            cells = [f"{row[c]:.2f}" if c == "deviation_magnitude_mm" else str(row[c]) for c in ev_cols]
            lines.append("| " + " | ".join(cells) + " |")
        if len(events) > 50:
            lines.append(f"\n... and {len(events) - 50} more (see events.csv)")
    else:
        lines.append("No suspicious events at the current thresholds.")
    lines.append("")

    lines.append("## Figures\n")
    lines.append("- [deviation_by_marker_boxplot.png](figures/deviation_by_marker_boxplot.png)")
    lines.append("- [event_locations.png](figures/event_locations.png)\n")

    lines.append("## Limitations\n")
    lines.append("- Single-trial diagnostic; does not generalize to other trials/subjects.")
    lines.append("- Geometric plausibility, not label correctness -- see the note above.")
    lines.append("- A marker with few incident bones (e.g. a fingertip) has less redundancy "
                 "to catch its own errors than a heavily-connected one (e.g. a palm marker).")
    lines.append("- Not pooled with, and not a substitute for, manual-label agreement.")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_one(participant: str, trial: str, out_dir: Path, thresholds_mm: list[float],
            jump_mad_multiplier: float, min_deviation_mm: float, make_plots_flag: bool):
    trial_path, *_ = ds.find_trial_csv(participant, trial)
    if trial_path is None:
        raise SystemExit(f"No CSV mapped to {participant}/{trial}")
    print(f"Loading {trial_path}")
    markers, labels = load_csv(str(trial_path))
    print(f"  {markers.shape[0]} frames, {markers.shape[1]} markers")

    df, events, gaps, marker_summary, overall, reference_table = analyze_trial(
        markers, labels, subject_id=participant, trial_id=trial,
        analysis_csv_name=trial_path.name, thresholds_mm=thresholds_mm,
        jump_mad_multiplier=jump_mad_multiplier, min_deviation_mm=min_deviation_mm)
    print(f"  reference_source_type={overall['reference_source_type']} "
          f"reference_status={overall['reference_status']} "
          f"({overall['n_valid_reference_bones']}/{overall['n_bones']} bones)")

    print(f"  availability {overall['mean_availability_pct']:.1f}% mean, "
          f"{overall['total_suspicious_events']} suspicious event(s), "
          f"{overall['total_affected_observations']} affected observations")

    out_dir.mkdir(parents=True, exist_ok=True)
    marker_summary.insert(0, "assessment_type", gc.ASSESSMENT_TYPE)
    marker_summary.insert(1, "subject_id", participant)
    marker_summary.insert(2, "trial_id", trial)
    marker_summary.to_csv(out_dir / "marker_summary.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    gaps.to_csv(out_dir / "gaps.csv", index=False)
    reference_table.to_csv(out_dir / "reference_bones.csv", index=False)
    pd.DataFrame([overall]).to_csv(out_dir / "trial_summary.csv", index=False)

    if make_plots_flag:
        make_plots(df, marker_summary, events, out_dir)
    write_trial_report(out_dir, overall, marker_summary, events, gaps, reference_table)
    print(f"  Saved to {out_dir}")

    return marker_summary, overall


def write_global_report(out_dir: Path, all_marker_summary: pd.DataFrame,
                         trial_summaries: pd.DataFrame, failures: list[tuple[str, str, str]]) -> None:
    lines = []
    lines.append("# Geometric self-consistency: dataset-level summary\n")
    lines.append(f"**assessment_type = {gc.ASSESSMENT_TYPE}**\n")
    lines.append(WHAT_THIS_CANNOT_TELL_YOU + "\n")
    lines.append(f"- Trials analysed: {len(trial_summaries)}")
    if failures:
        lines.append(f"- Trials skipped ({len(failures)}):")
        for p, t, reason in failures:
            lines.append(f"  - {p} / {t}: {reason}")
    lines.append("")

    lines.append("## Reference source, by trial\n")
    lines.append(REFERENCE_SOURCE_NOTE + "\n")
    ref_cols = ["subject_id", "trial_id", "analysis_trial", "reference_trial",
               "reference_source_type", "reference_status", "n_valid_reference_bones",
               "n_bones"]
    lines.append("| " + " | ".join(ref_cols) + " |")
    lines.append("|" + "---|" * len(ref_cols))
    for _, row in trial_summaries.sort_values(["subject_id", "trial_id"]).iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in ref_cols) + " |")
    lines.append("")

    lines.append("## Worst markers dataset-wide (by mean median deviation)\n")
    agg = (all_marker_summary.groupby("marker")
           .agg(n_trials=("trial_id", "nunique"),
                mean_availability_pct=("availability_pct", "mean"),
                mean_median_deviation_mm=("median_deviation_mm", "mean"),
                max_p95_deviation_mm=("p95_deviation_mm", "max"),
                total_suspicious_events=("suspicious_event_count", "sum"),
                max_gap_duration=("max_gap_duration", "max"))
           .sort_values("mean_median_deviation_mm", ascending=False)
           .reset_index())
    cols = list(agg.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in agg.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(str(v) if c in ("marker", "n_trials", "total_suspicious_events",
                                         "max_gap_duration") else f"{v:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Separate validation reference (NOT pooled with the above)\n")
    lines.append("The only trial with an independent, human-reviewed label check is "
                 "P10/Trial2_handsonly -- see its manual-label agreement report:\n")
    lines.append("- [../agreement/P10_Trial2_handsonly/report.md]"
                 "(../agreement/P10_Trial2_handsonly/report.md)")
    lines.append("- [../agreement/cross_trial/report.md](../agreement/cross_trial/report.md)\n")
    lines.append("These numbers are shown here only as a separate point of reference -- e.g. "
                 "to sanity-check whether P10/Trial2_handsonly's geometric score in this report "
                 "roughly agrees with its manual-label reliability score. They are never "
                 "averaged, merged, or otherwise pooled with the geometric-self-consistency "
                 "numbers in this report; the two measure different things (see the note "
                 "above) and mixing them would misrepresent both.\n")

    lines.append("## Limitations\n")
    lines.append("- Geometric plausibility, not label correctness -- repeated from the top "
                 "of this report because it is the single most important caveat here.")
    lines.append("- Dataset-wide aggregates here are simple means/sums across trials, not a "
                 "fitted or weighted model.")
    lines.append("- Trials that failed to analyse (see above, if any) are excluded, not "
                 "counted as passing.")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(out_dir: Path, thresholds_mm: list[float], jump_mad_multiplier: float,
            min_deviation_mm: float, make_plots_flag: bool) -> None:
    sessions = ds.load_trial_sessions(ds.REF_CSV)
    participants = sorted(sessions, key=ds.participant_sort_key)

    all_marker_summary_rows = []
    trial_summary_rows = []
    failures: list[tuple[str, str, str]] = []

    for participant in participants:
        have = sessions[participant]
        for trial in TRIAL_KEYS:
            if trial not in have:
                continue
            trial_out_dir = out_dir / f"{participant}_{ds.slug(trial)}"
            print(f"\n=== {participant} / {trial} ===")
            try:
                marker_summary, overall = run_one(
                    participant, trial, trial_out_dir, thresholds_mm,
                    jump_mad_multiplier, min_deviation_mm, make_plots_flag)
            except Exception as exc:                              # noqa: BLE001
                traceback.print_exc()
                failures.append((participant, trial, f"{type(exc).__name__}: {exc}"))
                continue
            all_marker_summary_rows.append(marker_summary)
            trial_summary_rows.append(overall)

    if not trial_summary_rows:
        raise SystemExit("No trials analysed successfully.")

    all_marker_summary = pd.concat(all_marker_summary_rows, ignore_index=True)
    trial_summaries = pd.DataFrame(trial_summary_rows)

    global_out_dir = out_dir / "dataset"
    global_out_dir.mkdir(parents=True, exist_ok=True)
    all_marker_summary.to_csv(global_out_dir / "dataset_marker_summary.csv", index=False)
    trial_summaries.to_csv(global_out_dir / "dataset_trial_summary.csv", index=False)
    write_global_report(global_out_dir, all_marker_summary, trial_summaries, failures)

    print(f"\nAnalysed {len(trial_summaries)} trial(s), {len(failures)} failure(s)")
    print(f"Saved dataset-level rollup to {global_out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default=None)
    ap.add_argument("--trial", default=None, help="manual_frames.csv Session name")
    ap.add_argument("--all", action="store_true",
                    help="run every (participant, trial) in manual_frames.csv, then write "
                         "the dataset-level rollup")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--thresholds-mm", default="1,2,5,10")
    ap.add_argument("--jump-mad-multiplier", type=float, default=6.0)
    ap.add_argument("--min-deviation-mm", type=float, default=4.0,
                    help="floor on the suspicious-event threshold -- bone deviation is "
                         "zero-inflated for a geometrically clean marker, which collapses "
                         "MAD to 0; without this floor every nonzero deviation would be "
                         "flagged (same failure mode as agreement.py's --jump-floor-mm)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    thresholds = [float(t) for t in args.thresholds_mm.split(",") if t.strip()]
    out_dir = Path(args.out_dir)

    if args.all:
        run_all(out_dir, thresholds, args.jump_mad_multiplier, args.min_deviation_mm,
                not args.no_plots)
    else:
        if not (args.participant and args.trial):
            raise SystemExit("Pass --participant and --trial, or --all")
        trial_out_dir = out_dir / f"{args.participant}_{ds.slug(args.trial)}"
        run_one(args.participant, args.trial, trial_out_dir, thresholds,
                args.jump_mad_multiplier, args.min_deviation_mm, not args.no_plots)


if __name__ == "__main__":
    main()
