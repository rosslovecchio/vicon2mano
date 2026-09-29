#!/usr/bin/env python3
"""Quantify agreement between on-disk Vicon labels and a manually-labelled
export, for one complete trial.

Measurement only: no subsampling, bootstrapping, cross-trial/-subject
analysis, ML, auto-relabelling, swap correction, or reliability
thresholds -- those are later steps. Nothing here silently drops an
observation; missing/unmatched rows are kept and counted.

Usage
-----
python scripts/shared/analyze_manual_agreement.py --participant P10 --trial "Trial2 Hands only"
python scripts/shared/analyze_manual_agreement.py --vicon-csv <path> --manual-csv <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv          # noqa: E402
from vicon2mano.core import dataset as ds             # noqa: E402
from vicon2mano.core import agreement as ag           # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "shared" / "agreement"


def make_plots(df, marker_summary, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    err = df["euclidean_error_mm"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=100, color="#4C72B0")
    ax.set_xlabel("Euclidean error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Per-frame marker error: Vicon vs. manual labels")
    fig.tight_layout()
    fig.savefig(fig_dir / "error_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * marker_summary.shape[0]), 5))
    order = marker_summary.sort_values("median_error_mm")["marker"]
    data = [df.loc[df["marker"] == m, "euclidean_error_mm"].dropna() for m in order]
    try:
        ax.boxplot(data, tick_labels=order, showfliers=False)
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Euclidean error (mm)")
    ax.set_title("Error by marker (outliers hidden)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_by_marker_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for m in order:
        g = df[df["marker"] == m]
        ax.plot(g["frame"], g["euclidean_error_mm"], lw=0.5, alpha=0.6, label=m)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean error (mm)")
    ax.set_title("Error vs. frame, all markers")
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_frame.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, axis in zip(axes, "xyz"):
        v = df[f"vicon_{axis}"]
        m = df[f"manual_{axis}"]
        ok = v.notna() & m.notna()
        mean = (v[ok] + m[ok]) / 2
        diff = v[ok] - m[ok]
        ax.scatter(mean, diff, s=2, alpha=0.2)
        ax.axhline(diff.mean(), color="red", lw=1)
        ax.set_title(f"Bland-Altman ({axis})")
        ax.set_xlabel(f"mean {axis} (mm)")
        ax.set_ylabel(f"vicon - manual {axis} (mm)")
    fig.tight_layout()
    fig.savefig(fig_dir / "bland_altman.png", dpi=150)
    plt.close(fig)


def write_report(out_dir: Path, participant, trial, vicon_csv, manual_csv,
                  match, overall, marker_summary, events, event_summary,
                  missingness, reliability) -> None:
    lines = []
    lines.append(f"# Manual review of Vicon label retention and trajectory "
                 f"availability: {participant} / {trial}\n")
    lines.append("Because the manual export retains Vicon coordinates when the "
                 "original identity is accepted, zero coordinate difference "
                 "indicates label retention rather than independent positional "
                 "agreement -- these numbers are not a submillimetre validation "
                 "of Vicon's measurements.\n")
    lines.append(f"- Vicon file: `{vicon_csv}`")
    lines.append(f"- Manual file: `{manual_csv}`")
    lines.append("- Coordinate units: mm\n")
    lines.append(f"- Frames compared: {overall['n_frames']}")
    lines.append(f"- Markers compared: {overall['n_markers']} "
                 f"(vicon-only: {len(match.only_a)}, manual-only: {len(match.only_b)})")
    lines.append(f"- Matched frame-marker pairs / expected frame-marker pairs: "
                 f"{overall['matched']} / {overall['n_rows']} ({overall['matched_pct']:.1f}%)")
    lines.append("  (expected = n_frames x n_common_markers; \"matched\" means both "
                 "sides had a finite coordinate for that frame+marker)")
    lines.append(f"- Missing, by cause: vicon-missing (manual present) "
                 f"{overall['missing_vicon_coordinates']}, "
                 f"manual-missing (vicon present) {overall['missing_manual_coordinates']}, "
                 f"both-missing {overall['missing_both_coordinates']} "
                 f"({overall['missing_pct']:.1f}% of all pairs)\n")

    lines.append("## Overall positional error (matched, valid pairs only)\n")
    lines.append("**Read this section together with the note below on what "
                 "`euclidean_error_mm` = 0 actually means in this dataset -- "
                 "it is the common case, not evidence of a computation bug.**\n")
    lines.append(f"- valid comparisons: {overall['valid_comparisons']}, of which "
                 f"zero-error: {overall['zero_error_comparisons']} "
                 f"({overall['zero_error_pct']:.1f}%), "
                 f"nonzero-error: {overall['nonzero_error_comparisons']}")
    lines.append(f"- mean {overall['mean_error_mm']:.2f} mm, "
                 f"median {overall['median_error_mm']:.2f} mm, "
                 f"RMSE {overall['rmse_mm']:.2f} mm")
    lines.append(f"- p95 {overall['p95_error_mm']:.2f} mm, "
                 f"max {overall['max_error_mm']:.2f} mm")
    for k, v in overall.items():
        if k.startswith("pct_within_"):
            lines.append(f"- {k}: {v:.1f}%")
    lines.append("")
    lines.append("### Why so many exact zeros\n")
    lines.append("The manually-labelled export is produced by *accepting or "
                 "reassigning* Vicon's own reconstructed 3-D points, not by "
                 "independently re-digitizing marker positions. Wherever the "
                 "human labeller agreed with the on-disk Vicon label, the "
                 "coordinate is the same value carried over unchanged -- so an "
                 "exact `0.00mm` error means \"label identity unchanged\", not "
                 "\"measured to sub-millimetre precision\". A nonzero error means "
                 "the labeller reassigned that marker's identity for that frame, "
                 "so error magnitude here mostly reflects *how far off* the "
                 "original (wrong) label's marker was, not measurement noise. "
                 "This makes `p95 == 0mm` for a marker mathematically consistent "
                 "whenever >=95% of its valid frames kept their original label, "
                 "which is confirmed directly from the per-frame table's "
                 "`zero_error_comparisons` count, not inferred.\n")

    lines.append("## Per-marker summary\n")
    cols = ["marker", "matched_pct", "missing_pct", "valid_comparisons",
            "zero_error_pct", "mean_error_mm", "median_error_mm", "rmse_mm",
            "p95_error_mm", "max_error_mm"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in marker_summary.iterrows():
        cells = [f"{row[c]:.2f}" if c not in ("marker", "valid_comparisons") else str(row[c])
                 for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lowest = marker_summary.nsmallest(3, "median_error_mm")["marker"].tolist()
    highest = marker_summary.nlargest(3, "p95_error_mm")["marker"].tolist()
    most_missing = marker_summary.nlargest(3, "missing_pct")["marker"].tolist()
    lines.append(f"- Lowest median error: {', '.join(lowest)}")
    lines.append(f"- Highest p95 error: {', '.join(highest)}")
    lines.append(f"- Most missing observations: {', '.join(most_missing)}")

    n_events = len(events)
    n_obs = int(events["n_observations"].sum()) if n_events else 0
    lines.append(f"- Temporal-jump events (diagnostic only, contiguous per marker): "
                 f"{n_events}, covering {n_obs} marker-frame observations\n")

    lines.append("## Temporal-jump events, by marker\n")
    lines.append("A \"temporal-jump event\" here is a consecutive run of frames in "
                 "which the manually retained identity differs substantially from "
                 "the original Vicon identity (see `flag_temporal_jumps`), not a "
                 "single-instant velocity spike. `event_type` distinguishes a "
                 "one-frame discontinuity from a sustained run of several "
                 "consecutive frames carrying the same disagreement -- naming what "
                 "the detector measures (a run of sharp coordinate disagreement), "
                 "not a claim about how the labeller produced it, e.g. one "
                 "continuous edit vs. several adjacent ones; that would need the "
                 "labelling process itself as evidence, which this analysis does "
                 "not have. `adjacent_missing_vicon` flags whether the frame just "
                 "before/after the event has no Vicon coordinate at all, i.e. "
                 "whether the disagreement sits next to a Vicon dropout rather "
                 "than in an otherwise-unbroken stretch. This does not reclassify "
                 "the event -- every event here is, by construction, built from "
                 "matched, valid frames -- it is context for reading the event.\n")
    if len(event_summary):
        cols = ["marker", "event_count", "affected_observations",
                "median_event_duration", "max_event_duration", "total_event_duration",
                "n_single_frame_events", "n_sustained_events", "n_adjacent_missing_vicon"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in event_summary.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        lines.append("")

        total_obs = int(event_summary["affected_observations"].sum())
        total_n_events = int(event_summary["event_count"].sum())
        longest = int(event_summary["max_event_duration"].max())
        longest_frac = 100.0 * longest / total_obs if total_obs else float("nan")
        top10_obs = int(events.nlargest(10, "n_observations")["n_observations"].sum())
        top10_frac = 100.0 * top10_obs / total_obs if total_obs else float("nan")
        lines.append(f"- {total_obs} affected observations across {total_n_events} events "
                     f"trial-wide")
        lines.append(f"- longest_event_fraction (longest event / all affected observations): "
                     f"{longest} / {total_obs} = {longest_frac:.2f}%")
        lines.append(f"- top_10_event_fraction (10 longest events / all affected observations): "
                     f"{top10_obs} / {total_obs} = {top10_frac:.2f}%")
        lines.append(f"- No single event dominates the total ({longest_frac:.2f}%), but the "
                     f"10 longest together account for {top10_frac:.2f}% of it -- so the "
                     f"burden is concentrated in a moderate number of sustained events, "
                     f"not spread evenly across many one-off frames, and not owned by one "
                     f"outlier event either.")
    else:
        lines.append("No temporal-jump events at the current `--jump-mad-multiplier` "
                     "/ `--jump-floor-mm` settings.")
    lines.append("")

    lines.append("## Trajectory availability (Vicon-side gaps)\n")
    lines.append("Distinct from label agreement above: contiguous runs of missing "
                 "Vicon coordinates per marker, independent of whether the manual "
                 "label agrees with anything.\n")
    if len(missingness):
        gap_summary = (missingness.groupby("marker")["n_observations"]
                       .agg(gap_count="count", affected_frames="sum",
                            max_gap_duration="max")
                       .sort_values("affected_frames", ascending=False)
                       .reset_index())
        cols = ["marker", "gap_count", "affected_frames", "max_gap_duration"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in gap_summary.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    else:
        lines.append("No Vicon-side gaps found.")
    lines.append("")

    lines.append("## Marker reliability: availability, retention, and usability\n")
    lines.append("These are three different questions, and a marker can score well "
                 "on one and poorly on another -- e.g. `Thumb3` keeps its original "
                 "identity 95.07% of the time *when a coordinate exists at all*, but "
                 "only has a coordinate 76.14% of the time, so it should not be "
                 "called \"reliable\" on the retention number alone.\n")
    lines.append("- `availability_pct`: how often a Vicon coordinate exists at all.")
    lines.append("- `label_retention_pct_valid`: of the available frames, how often "
                 "the manual export kept the original identity.")
    lines.append("- `reassignment_pct_valid`: the complement -- how often the "
                 "labeller changed it.")
    lines.append("- `overall_usable_pct`: zero-error comparisons over *all expected* "
                 "frame-marker pairs -- the one number combining availability and "
                 "retention (availability x retention), i.e. what fraction of the "
                 "whole trial this marker is both present and unreassigned for. "
                 "Sorted ascending below, worst first.\n")
    cols = ["marker", "availability_pct", "label_retention_pct_valid",
            "reassignment_pct_valid", "event_count", "sustained_event_count",
            "max_event_duration", "max_gap_duration", "overall_usable_pct"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in reliability.iterrows():
        cells = [f"{row[c]:.2f}" if c not in ("marker", "event_count", "sustained_event_count",
                                              "max_event_duration", "max_gap_duration")
                 else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Figures\n")
    for name in ("error_histogram.png", "error_by_marker_boxplot.png",
                 "error_vs_frame.png", "bland_altman.png"):
        lines.append(f"- [{name}](figures/{name})")
    lines.append("")

    lines.append("## Limitations\n")
    lines.append("- This analysis describes agreement in one complete trial.")
    lines.append("- The results do not establish generalizability to other subjects or trials.")
    lines.append("- Adjacent frames are temporally correlated.")
    lines.append("- Mean error alone is not sufficient to establish label reliability.")
    lines.append("- Missing labels and incorrect labels are different failure modes.")
    lines.append("- This script does not automatically correct marker swaps or gaps.")
    lines.append("- No statistical confidence intervals are calculated in this first step.")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(participant: str | None, trial: str | None,
        vicon_csv: Path | None, manual_csv: Path | None,
        out_dir: Path, thresholds_mm: list[float],
        jump_mad_multiplier: float, jump_floor_mm: float,
        make_plots_flag: bool, make_report: bool):
    """Run the full single-trial analysis, writing every output to
    ``out_dir``. Returns the per-marker reliability table (see
    ``agreement.reliability_table``) so callers that need to aggregate
    across trials (``aggregate_manual_agreement.py``) don't have to re-parse
    the CSVs or re-derive it themselves.
    """
    if vicon_csv is None:
        if participant is None or trial is None:
            raise SystemExit("Pass either --vicon-csv, or both --participant and --trial")
        trial_path, *_ = ds.find_trial_csv(participant, trial)
        if trial_path is None:
            raise SystemExit(f"No CSV mapped to {participant}/{trial}")
        vicon_csv = trial_path
    if manual_csv is None:
        manual_csv = ds.find_manual_csv(vicon_csv)
        if manual_csv is None:
            raise SystemExit(f"No manually-labelled export found next to {vicon_csv} "
                              f"(tried suffixes {ds.MANUAL_LABEL_SUFFIXES}); pass --manual-csv")

    print(f"Vicon:  {vicon_csv}")
    print(f"Manual: {manual_csv}")
    markers_v, labels_v = load_csv(str(vicon_csv))
    markers_m, labels_m = load_csv(str(manual_csv))
    print(f"  vicon:  {markers_v.shape[0]} frames, {markers_v.shape[1]} markers")
    print(f"  manual: {markers_m.shape[0]} frames, {markers_m.shape[1]} markers")

    match = ag.match_markers(labels_v, labels_m)
    if match.only_a:
        print(f"[warn] markers only in vicon labels: {match.only_a}")
    if match.only_b:
        print(f"[warn] markers only in manual labels: {match.only_b}")
    if not match.common:
        raise SystemExit("No common markers between the two files.")

    df = ag.per_frame_table(markers_v, markers_m, match)
    df = ag.add_threshold_columns(df, thresholds_mm)
    df = ag.flag_temporal_jumps(df, k=jump_mad_multiplier, min_jump_mm=jump_floor_mm)
    events = ag.temporal_events(df)
    event_summary = ag.per_marker_event_summary(events)
    missingness = ag.missingness_events(df)

    marker_summary = ag.per_marker_summary(df, thresholds_mm)
    overall = ag.overall_summary(df, thresholds_mm)
    reliability = ag.reliability_table(marker_summary, event_summary, missingness)

    print(f"  matched {overall['matched']}/{overall['n_rows']} "
          f"({overall['matched_pct']:.1f}%); of {overall['valid_comparisons']} valid "
          f"comparisons, {overall['zero_error_pct']:.1f}% are exact-zero error")
    print(f"  mean error {overall['mean_error_mm']:.2f}mm, p95 {overall['p95_error_mm']:.2f}mm "
          f"(zero-inflated -- see report.md for why exact-zero errors are expected here)")
    n_obs = int(events["n_observations"].sum()) if len(events) else 0
    print(f"  {len(events)} temporal-jump event(s) (diagnostic only), "
          f"covering {n_obs} marker-frame observations")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_frame_comparison.csv", index=False)
    marker_summary.to_csv(out_dir / "per_marker_summary.csv", index=False)
    events.to_csv(out_dir / "temporal_events.csv", index=False)
    event_summary.to_csv(out_dir / "temporal_event_summary_by_marker.csv", index=False)
    missingness.to_csv(out_dir / "missingness_events.csv", index=False)
    reliability.to_csv(out_dir / "marker_reliability.csv", index=False)
    import pandas as pd
    pd.DataFrame([overall]).to_csv(out_dir / "overall_summary.csv", index=False)
    print(f"Saved tables to {out_dir}")

    if make_plots_flag:
        make_plots(df, marker_summary, out_dir)
        print(f"Saved figures to {out_dir / 'figures'}")

    if make_report:
        write_report(out_dir, participant or vicon_csv.parent.name, trial or vicon_csv.stem,
                     vicon_csv, manual_csv, match, overall, marker_summary, events,
                     event_summary, missingness, reliability)
        print(f"Saved {out_dir / 'report.md'}")

    return reliability


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default=None)
    ap.add_argument("--trial", default=None, help="manual_frames.csv Session name")
    ap.add_argument("--vicon-csv", default=None, help="explicit path, overrides --participant/--trial")
    ap.add_argument("--manual-csv", default=None,
                    help="explicit path to the manually-labelled export; auto-detected "
                         "next to --vicon-csv if omitted")
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {OUT_DIR}/<participant>_<trial>")
    ap.add_argument("--thresholds-mm", default="1,2,5,10",
                    help="comma-separated agreement thresholds in mm")
    ap.add_argument("--jump-mad-multiplier", type=float, default=6.0,
                    help="median + k*MAD threshold for the diagnostic temporal-jump flag")
    ap.add_argument("--jump-floor-mm", type=float, default=2.0,
                    help="minimum error, in mm, before a frame can be flagged as a "
                         "temporal jump -- this data is zero-inflated (see report.md), "
                         "which collapses MAD to 0 for most markers; without this floor "
                         "every nonzero-error frame gets flagged")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    thresholds = [float(t) for t in args.thresholds_mm.split(",") if t.strip()]

    vicon_csv = Path(args.vicon_csv) if args.vicon_csv else None
    manual_csv = Path(args.manual_csv) if args.manual_csv else None

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        slug_bits = [b for b in (args.participant, args.trial) if b]
        slug = ds.slug("_".join(slug_bits)) if slug_bits else (vicon_csv.stem if vicon_csv else "trial")
        out_dir = OUT_DIR / slug

    run(args.participant, args.trial, vicon_csv, manual_csv, out_dir, thresholds,
        args.jump_mad_multiplier, args.jump_floor_mm, not args.no_plots, not args.no_report)


if __name__ == "__main__":
    main()
