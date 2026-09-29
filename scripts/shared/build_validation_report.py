#!/usr/bin/env python3
"""Build the final 4 outputs of the manual-validation plan from the
500-frame sample's completed ``manual_annotations.csv``:

- ``results/shared/validation/trial_summary.csv``  (deliverable #3)
- ``results/shared/validation/marker_summary.csv``  (deliverable #4)
- ``results/shared/validation/validation_report.html``  (deliverable #5)

(``sampling_manifest.csv``, deliverable #1, is built by
``build_sampling_manifest.py``; ``manual_annotations.csv``, deliverable #2,
is produced by hand via ``label_selected_frames.m`` -- this script only
consumes both, and requires manual_annotations.csv to already exist.)

assessment_type = manual_identity_validation (see
vicon2mano/core/validation.py) -- an independent identity check, never
pooled with the P10 manual-label *retention* assessment
(vicon2mano/core/agreement.py) or with geometric_self_consistency's own
numbers. Both of those are cited in their own report sections for
comparison only.

Usage
-----
python scripts/shared/build_validation_report.py
python scripts/shared/build_validation_report.py --annotations-csv <path>
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

from vicon2mano.core import dataset as ds        # noqa: E402
from vicon2mano.core import validation as val    # noqa: E402

GEOM_DIR = REPO_ROOT / "results" / "shared" / "geometric_consistency"
FRAME_SAMPLE_DIR = REPO_ROOT / "results" / "shared" / "frame_sample"
MANIFEST_CSV = FRAME_SAMPLE_DIR / "sampling_manifest.csv"
DEFAULT_ANNOTATIONS_CANDIDATES = [
    FRAME_SAMPLE_DIR / "manual_annotations.csv",
    Path(r"C:\Users\RL000009\MATLAB\Projects\VICON_Labelling\gesture_labelling\manual_annotations.csv"),
]
OUT_DIR = REPO_ROOT / "results" / "shared" / "validation"

REQUIRED_ANNOTATION_COLUMNS = ("trial_id", "participant", "frame", "marker_name",
                               "vicon_label", "manual_label", "status", "confidence", "reviewer")


def find_annotations_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"{explicit} not found")
        return explicit
    for cand in DEFAULT_ANNOTATIONS_CANDIDATES:
        if cand.exists():
            return cand
    raise SystemExit(
        "manual_annotations.csv not found in any default location:\n  "
        + "\n  ".join(str(c) for c in DEFAULT_ANNOTATIONS_CANDIDATES)
        + "\nComplete (at least some of) the manual review with "
          "label_selected_frames.m first, or pass --annotations-csv.")


def trial_dir_name(participant: str, trial_id: str) -> str:
    return f"{participant}_{ds.slug(trial_id)}"


def load_events_by_trial(pairs) -> dict[tuple, pd.DataFrame]:
    out = {}
    for participant, trial_id in pairs:
        d = GEOM_DIR / trial_dir_name(participant, trial_id) / "events.csv"
        if d.exists():
            out[(participant, trial_id)] = pd.read_csv(d)
    return out


def load_manifest() -> pd.DataFrame | None:
    if MANIFEST_CSV.exists():
        return pd.read_csv(MANIFEST_CSV)
    return None


def validate_annotation_columns(annotations: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_ANNOTATION_COLUMNS if c not in annotations.columns]
    if missing:
        raise SystemExit(f"manual_annotations.csv is missing required column(s): {missing}")
    bad_status = set(annotations["status"].unique()) - set(val.STATUSES)
    if bad_status:
        raise SystemExit(f"manual_annotations.csv has unrecognised status value(s): {bad_status} "
                          f"(expected one of {val.STATUSES})")


def build_html_report(out_dir: Path, annotations: pd.DataFrame, manifest: pd.DataFrame | None,
                       overall: dict, marker_summary: pd.DataFrame, trial_summary: pd.DataFrame,
                       contingency: pd.DataFrame, per_frame: pd.DataFrame,
                       p10_section: str) -> None:
    def df_to_html(df: pd.DataFrame, float_cols=()) -> str:
        d = df.copy()
        for c in float_cols:
            if c in d.columns:
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        return d.to_html(index=False, border=0, classes="tbl")

    n_frames_sampled = len(manifest) if manifest is not None else annotations[
        ["participant", "trial_id", "frame"]].drop_duplicates().shape[0]

    category_counts_html = ""
    static_section = ""
    if manifest is not None:
        cc = manifest["category"].value_counts().rename_axis("category").reset_index(name="count")
        category_counts_html = df_to_html(cc)

        static_trials = manifest[manifest["category"] == "static_or_calibration"]
        if len(static_trials):
            static_keys = set(zip(static_trials["participant"], static_trials["trial_id"]))
            static_annot = annotations[annotations.apply(
                lambda r: (r["participant"], r["trial_id"]) in static_keys, axis=1)]
            if len(static_annot):
                static_overall = val.label_accuracy_summary(static_annot)
                static_section = f"""
                <h2>Static / calibration trials (kept separate)</h2>
                <p>Frames sampled from trials whose reference_source_type is static or
                calibration are reported here on their own, not folded into the headline
                accuracy above -- their labelling process and purpose differ from movement
                trials (see the geometric self-consistency reports' reference-source notes).</p>
                <ul>
                  <li>Judgements: {static_overall['n_judgements']}</li>
                  <li>Accuracy (determinate judgements only): {static_overall['accuracy']:.1%}
                      (n={static_overall['n_determinate']})</li>
                  <li>Frames fully confirmed correct: {static_overall['pct_frames_all_correct']:.1f}%</li>
                </ul>
                """

    examples_html = "<p>No wrong_label/ghost examples in this sample yet.</p>"
    flagged = annotations[annotations["status"].isin(["wrong_label", "ghost"])]
    if len(flagged):
        cols = ["participant", "trial_id", "frame", "marker_name", "vicon_label",
                "manual_label", "status", "confidence"]
        examples_html = df_to_html(flagged[cols].head(20))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Manual identity validation report</title>
<style>
body {{ font-family: sans-serif; max-width: 1000px; margin: 2em auto; line-height: 1.5; }}
table.tbl {{ border-collapse: collapse; margin: 1em 0; }}
table.tbl th, table.tbl td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
table.tbl th {{ background: #eee; }}
table.tbl td:first-child, table.tbl th:first-child {{ text-align: left; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
.note {{ background: #fff8e6; border-left: 4px solid #e6b800; padding: 0.5em 1em; }}
</style></head><body>

<h1>Independent manual identity validation report</h1>
<p><code>assessment_type = manual_identity_validation</code></p>

<div class="note">
This report measures whether a human, judging from scratch, agrees with the
on-disk Vicon marker label at {n_frames_sampled} sampled frames across (up to)
66 trials. It does <b>not</b> establish overall dataset correctness (this is a
sample, not a census) and it is a separate assessment from both the P10
manual-label <i>retention</i> analysis and the geometric self-consistency
screen -- see the "P10" and "geometry vs. manual" sections below for how
those relate, without being combined into one number.
</div>

<h2>1. How frames were selected</h2>
<p>500 anchor frames across 5 categories (random, largest geometric anomaly,
poorest trajectory availability, geometry/availability disagreement,
static-or-calibration trials), each expanded to a +/-5 frame window for
context. See <code>results/shared/frame_sample/report.md</code> for the full
method and <code>sampling_manifest.csv</code> for the exact list.</p>
{category_counts_html}

<h2>2. How manual labels were defined</h2>
<p>Five statuses only: <code>correct</code>, <code>wrong_label</code>,
<code>missing</code> (no coordinate that frame, auto-detected -- not a human
judgement), <code>ghost</code> (a coordinate exists but is not a real
marker), <code>ambiguous</code> (reviewer could not tell). No coordinates
were changed or filled at this stage. Accuracy below is computed over
"determinate" judgements only (<code>correct</code> + <code>wrong_label</code>
+ <code>ghost</code>) -- <code>missing</code> and <code>ambiguous</code> are
excluded from the denominator rather than silently counted either way.</p>

<h2>3. Label accuracy</h2>
<ul>
  <li>Total judgements: {overall['n_judgements']}</li>
  <li>correct: {overall['n_correct']}, wrong_label: {overall['n_wrong_label']},
      missing: {overall['n_missing']}, ghost: {overall['n_ghost']},
      ambiguous: {overall['n_ambiguous']}</li>
  <li>Overall accuracy (determinate only): <b>{overall['accuracy']:.1%}</b>
      (n={overall['n_determinate']})</li>
  <li>Frames sampled: {overall['n_frames']}; fully confirmed correct:
      <b>{overall['pct_frames_all_correct']:.1f}%</b>
      ({overall['n_frames_all_correct']}/{overall['n_frames']})</li>
</ul>

<h2>4. Marker-wise error rates</h2>
{df_to_html(marker_summary, float_cols=['accuracy'])}

<h2>5. Trial-wise error rates</h2>
{df_to_html(trial_summary, float_cols=['accuracy'])}

<h2>6. Geometry vs. manual comparison</h2>
<p>Per sampled frame (excluding frames with any <code>ambiguous</code>
judgement -- see <code>vicon2mano.core.validation.geometry_vs_manual_table</code>):
does a geometric self-consistency anomaly at that frame predict a manual
identity error there? {len(per_frame)} of {n_frames_sampled} sampled frames
are included in this comparison after that exclusion.</p>
{df_to_html(contingency)}
<p>Read this table, do not average it into either assessment's own numbers --
see rule-of-thumb guidance in <code>results/shared/frame_sample/report.md</code>'s
companion planning notes for how to act on high/low false-alarm and
miss rates here.</p>

<h2>7. Representative examples</h2>
<p>Up to 20 flagged (wrong_label / ghost) marker judgements from this sample:</p>
{examples_html}

{static_section}

<h2>8. P10 cross-check (reported separately, not combined)</h2>
{p10_section}

</body></html>
"""
    (out_dir / "validation_report.html").write_text(html, encoding="utf-8")


def build_p10_section(annotations: pd.DataFrame) -> str:
    p10 = annotations[annotations["participant"] == "P10"]
    if p10.empty:
        return ("<p>No P10 frames happened to be sampled in this run of the 500-frame "
                "selection, so there is nothing to cross-check yet.</p>")
    p10_overall = val.label_accuracy_summary(p10)
    return f"""
    <p>P10/Trial2_handsonly already has a separate assessment reported as
    <b>manual label-retention analysis</b> (does the manually-labelled export
    keep or change the on-disk Vicon label -- see
    <code>results/shared/agreement/P10_Trial2_handsonly/report.md</code>).
    The numbers below are this run's <b>independent manual identity
    validation</b> restricted to whichever P10 frames were sampled here --
    a different question, on possibly a different trial within P10, asked by
    a human with no reference to the retention analysis. These two are
    reported side by side to sanity-check the new workflow, and are
    <b>not averaged or combined</b>.</p>
    <ul>
      <li>P10 judgements in this sample: {p10_overall['n_judgements']}</li>
      <li>P10 accuracy (determinate only): {p10_overall['accuracy']:.1%}
          (n={p10_overall['n_determinate']})</li>
      <li>P10 frames fully confirmed correct: {p10_overall['pct_frames_all_correct']:.1f}%</li>
    </ul>
    """


def run(annotations_csv: Path, out_dir: Path) -> None:
    annotations = pd.read_csv(annotations_csv)
    validate_annotation_columns(annotations)
    print(f"Loaded {len(annotations)} annotation row(s) from {annotations_csv}")

    manifest = load_manifest()

    overall = val.label_accuracy_summary(annotations)
    marker_summary = val.per_marker_accuracy(annotations)
    trial_summary = val.per_trial_accuracy(annotations)

    pairs = list(zip(annotations["participant"], annotations["trial_id"]))
    events_by_trial = load_events_by_trial(set(pairs))
    per_frame = val.geometry_vs_manual_table(annotations, events_by_trial)
    contingency = val.geometry_vs_manual_contingency(per_frame)

    print(f"  overall accuracy (determinate): {overall['accuracy']:.1%} "
          f"(n={overall['n_determinate']}); "
          f"{overall['pct_frames_all_correct']:.1f}% of frames fully confirmed correct")

    out_dir.mkdir(parents=True, exist_ok=True)
    trial_summary.insert(0, "assessment_type", val.ASSESSMENT_TYPE)
    marker_summary.insert(0, "assessment_type", val.ASSESSMENT_TYPE)
    trial_summary.to_csv(out_dir / "trial_summary.csv", index=False)
    marker_summary.to_csv(out_dir / "marker_summary.csv", index=False)
    per_frame.to_csv(out_dir / "geometry_vs_manual_per_frame.csv", index=False)
    contingency.to_csv(out_dir / "geometry_vs_manual_contingency.csv", index=False)

    p10_section = build_p10_section(annotations)
    build_html_report(out_dir, annotations, manifest, overall, marker_summary, trial_summary,
                       contingency, per_frame, p10_section)

    print(f"Saved trial_summary.csv, marker_summary.csv, validation_report.html to {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-csv", default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    annotations_csv = find_annotations_csv(Path(args.annotations_csv) if args.annotations_csv else None)
    run(annotations_csv, Path(args.out_dir))


if __name__ == "__main__":
    main()
