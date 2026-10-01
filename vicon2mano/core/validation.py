"""Independent manual identity validation: compare hand-reviewed marker
identities (from the 500-frame sample, labelled via
``label_selected_frames.m``) against the on-disk Vicon labels, and against
the geometric self-consistency screen.

**assessment_type = "manual_identity_validation"**

This is a *third*, separate assessment from the other two in this repo:

- ``vicon2mano.core.agreement`` (assessment: manual-label *retention*) asks
  "did the P10 manual-labelled export keep or change each Vicon label",
  using a second *file* as the comparison, for one trial.
- ``vicon2mano.core.geometric_consistency`` (assessment_type =
  geometric_self_consistency) asks "is this frame's geometry plausible",
  with no human judgement involved, for all 66 trials.
- This module asks "is the Vicon label actually the correct physical
  marker", judged from scratch by a human for a 500-frame sample spanning
  all 66 trials -- the only one of the three with an independent identity
  ground truth. It is never pooled with the P10 retention numbers (see
  ``geometry_vs_manual_table``'s docstring and the validation report's own
  "P10" section, which exists only to sanity-check that this workflow
  produces something consistent with what the retention analysis already
  found on that one trial -- not to average the two together).

Status vocabulary (exactly what ``label_selected_frames.m`` writes):
``correct``, ``wrong_label``, ``missing``, ``ghost``, ``ambiguous``. See
that script's docstring for definitions -- this module treats them as
given, it does not redefine them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASSESSMENT_TYPE = "manual_identity_validation"

STATUSES = ("correct", "wrong_label", "missing", "ghost", "ambiguous")

# A judgement is "determinate" if the human could actually decide whether
# the label was right or wrong. "missing" has no label to judge (no
# coordinate exists) and "ambiguous" is an explicit "couldn't tell" -- both
# are excluded from accuracy denominators so they neither help nor hurt the
# score, instead of silently counting as correct (missing) or wrong
# (ambiguous), either of which would misrepresent what was actually judged.
DETERMINATE_STATUSES = ("correct", "wrong_label", "ghost")


def _accuracy_stats(df: pd.DataFrame) -> dict:
    n_total = len(df)
    counts = {s: int((df["status"] == s).sum()) for s in STATUSES}
    determinate = df[df["status"].isin(DETERMINATE_STATUSES)]
    n_determinate = len(determinate)
    n_correct = counts["correct"]
    return {
        "n_judgements": n_total,
        **{f"n_{s}": counts[s] for s in STATUSES},
        "n_determinate": n_determinate,
        "accuracy": n_correct / n_determinate if n_determinate else np.nan,
    }


def label_accuracy_summary(annotations: pd.DataFrame) -> dict:
    """Overall accuracy across the whole manual sample, plus the fraction
    of sampled *frames* (not individual marker judgements) that are fully
    confirmed correct.

    A frame counts as "all correct" only if every one of its judged markers
    is ``correct`` or ``missing`` (a missing marker has no identity to get
    wrong) -- a frame with even one ``ambiguous`` marker is NOT counted as
    confirmed correct, since "couldn't tell" is not the same as "verified
    fine".
    """
    out = _accuracy_stats(annotations)
    frame_keys = annotations[["participant", "trial_id", "frame"]].drop_duplicates()
    n_frames = len(frame_keys)
    bad_statuses = {"wrong_label", "ghost", "ambiguous"}
    bad_frames = (annotations[annotations["status"].isin(bad_statuses)]
                  [["participant", "trial_id", "frame"]].drop_duplicates())
    n_all_correct = n_frames - len(bad_frames)
    out["n_frames"] = n_frames
    out["n_frames_all_correct"] = n_all_correct
    out["pct_frames_all_correct"] = 100.0 * n_all_correct / n_frames if n_frames else np.nan
    return out


def per_marker_accuracy(annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per marker (``marker_name``), same statistics as
    :func:`label_accuracy_summary` broken out per marker."""
    rows = []
    for marker, g in annotations.groupby("marker_name"):
        row = {"marker_name": marker}
        row.update(_accuracy_stats(g))
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("accuracy")
            .reset_index(drop=True))


def per_trial_accuracy(annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, trial_id), same statistics as
    :func:`label_accuracy_summary` broken out per trial."""
    rows = []
    for (participant, trial_id), g in annotations.groupby(["participant", "trial_id"]):
        row = {"participant": participant, "trial_id": trial_id}
        row.update(_accuracy_stats(g))
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("accuracy")
            .reset_index(drop=True))


def frame_geometric_status(events: pd.DataFrame, frame: int) -> tuple[bool, float]:
    """Whether ``frame`` falls inside any geometric-self-consistency event
    in ``events`` (that trial's ``events.csv``), and the largest
    ``deviation_magnitude_mm`` among events that cover it.

    Uses the event's full ``[start_frame, end_frame]`` span, not just its
    ``peak_frame`` -- a frame can be part of a flagged run without being
    the exact peak.
    """
    if events is None or events.empty:
        return False, 0.0
    covering = events[(events["start_frame"] <= frame) & (events["end_frame"] >= frame)]
    if covering.empty:
        return False, 0.0
    return True, float(covering["deviation_magnitude_mm"].max())


def geometry_vs_manual_table(annotations: pd.DataFrame,
                              events_by_trial: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """Per-sampled-frame comparison table (one row per (participant,
    trial_id, frame)) with ``geometric_anomaly``, ``geometric_score``,
    ``trajectory_availability``, and ``manual_error_present`` -- the input
    to the 2x2 contingency table in the validation report.

    ``manual_error_present`` is only ``True``/``False`` for frames with no
    ``ambiguous`` marker; a frame with any ambiguous judgement is dropped
    from this table entirely (not coerced to either side), since an
    inconclusive human judgement cannot honestly support either half of
    "does geometry predict manual error" -- see
    :func:`geometry_vs_manual_contingency` for where that exclusion is
    reported.

    ``trajectory_availability`` is the fraction of that frame's judged
    markers with status != "missing" -- computed from what the reviewer
    actually observed, not re-derived from the raw Vicon gap tables, so it
    reflects exactly the same frame the manual judgement is about.
    """
    rows = []
    for (participant, trial_id, frame), g in annotations.groupby(
            ["participant", "trial_id", "frame"]):
        if (g["status"] == "ambiguous").any():
            continue
        events = events_by_trial.get((participant, trial_id))
        is_anomalous, score = frame_geometric_status(events, frame)
        n_total = len(g)
        n_missing = int((g["status"] == "missing").sum())
        availability = (n_total - n_missing) / n_total if n_total else np.nan
        manual_error = bool(g["status"].isin(["wrong_label", "ghost"]).any())
        rows.append({
            "participant": participant, "trial_id": trial_id, "frame": frame,
            "geometric_anomaly": is_anomalous, "geometric_score": score,
            "trajectory_availability": availability, "manual_error_present": manual_error,
        })
    return pd.DataFrame(rows, columns=["participant", "trial_id", "frame", "geometric_anomaly",
                                       "geometric_score", "trajectory_availability",
                                       "manual_error_present"])


def geometry_vs_manual_contingency(per_frame: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 (Normal/Anomalous x No/Yes) table from
    :func:`geometry_vs_manual_table`'s output."""
    rows = []
    for geom_label, is_anom in (("Normal", False), ("Anomalous", True)):
        for err_label, has_err in (("No", False), ("Yes", True)):
            count = int(((per_frame["geometric_anomaly"] == is_anom) &
                        (per_frame["manual_error_present"] == has_err)).sum())
            rows.append({"Geometry": geom_label, "Manual error": err_label, "Count": count})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Building manual_annotations.csv from a hand-corrected trial export, as an
# alternative to typing per-marker verdicts in label_selected_frames.m:
# relabel directly in Nexus, export a corrected CSV (same
# "<stem>_manuallylabelled.csv" convention as vicon2mano.core.dataset's
# find_manual_csv), and diff it against the original. This still needs a
# record of which frames were actually looked at (an unedited marker is
# otherwise indistinguishable from an unreviewed one) -- that is what
# visited_frames.csv (logged by the simplified label_selected_frames.m) is
# for; this function only ever classifies frames that appear there.
# ---------------------------------------------------------------------------


def classify_marker_frame(orig_valid: bool, corrected_valid: bool, same_coords: bool) -> str | None:
    """One (marker, frame) judgement inferred from a before/after coordinate
    diff, given a human deliberately reviewed that frame in Nexus.

    - both missing -> ``"missing"`` (nothing was there to judge, same as the
      auto-detected case in ``label_selected_frames.m``).
    - present before, missing after -> ``"ghost"``: the reviewer deleted the
      label rather than reassigning it, which only makes sense if the point
      was not a real, correctly-identified marker. Per instruction, a label
      that disappears during review was NOT correct -- this is a determinate
      error, not a data gap, even though the surface shape (present -> NaN)
      looks like one.
    - present before and after, coordinates unchanged -> ``"correct"``.
    - present before and after, coordinates changed -> ``"wrong_label"``: the
      reviewer moved the value, i.e. reassigned it to what they judged the
      real identity to be.
    - missing before, present after -> ``None``: the reviewer added a
      coordinate where the original had none, i.e. gap-filling -- out of
      scope for this identity check (see the plan's "do not fill gaps yet"),
      so the caller should skip this case (with a warning), not classify it.
    """
    if not orig_valid and not corrected_valid:
        return "missing"
    if orig_valid and not corrected_valid:
        return "ghost"
    if not orig_valid and corrected_valid:
        return None
    return "correct" if same_coords else "wrong_label"


def build_annotations_from_correction(
    markers_orig: np.ndarray,
    labels_orig: list[str],
    markers_corrected: np.ndarray,
    labels_corrected: list[str],
    visited_frames: list[int],
    *,
    participant: str,
    trial_id: str,
    reviewer: str,
    match_markers_fn,
    atol_mm: float = 1.0,
) -> pd.DataFrame:
    """``manual_annotations.csv`` rows for one trial, built by diffing the
    original Vicon-labelled recording against a hand-corrected export, for
    exactly the frames in ``visited_frames`` (everything else is
    unreviewed and must not be scored either way).

    ``match_markers_fn`` is ``vicon2mano.core.agreement.match_markers`` --
    passed in rather than imported directly so this module does not import
    ``core.agreement`` at module load time (the two assessments stay
    independent modules; this function only borrows its base-name marker
    matching, which is generic string plumbing, not an agreement-specific
    concept).

    ``atol_mm`` defaults to 1.0mm, not 0 -- re-exporting a trial's
    trajectories through Nexus's ``GetTrajectory`` and writing them back out
    introduces ~0.005mm of round-trip noise even on markers nobody touched
    (measured directly on a real corrected export), so an exact-equality or
    near-zero tolerance would misclassify most unedited markers as
    ``wrong_label``. 1mm sits far above that noise floor and far below any
    real mislabelling (adjacent markers are tens of millimetres apart).
    """
    match = match_markers_fn(labels_orig, labels_corrected)
    rows = []
    skipped_gap_fills = []
    for frame in visited_frames:
        for name in match.common:
            a = markers_orig[frame, match.idx_a[name]]
            b = markers_corrected[frame, match.idx_b[name]]
            ok_a = np.isfinite(a).all()
            ok_b = np.isfinite(b).all()
            same = bool(ok_a and ok_b and np.allclose(a, b, atol=atol_mm))
            status = classify_marker_frame(ok_a, ok_b, same)
            if status is None:
                skipped_gap_fills.append((frame, name))
                continue
            rows.append({
                "trial_id": trial_id, "participant": participant, "frame": frame,
                "marker_name": name, "vicon_label": name,
                "manual_label": name if status in ("correct", "missing") else "",
                "status": status, "confidence": "high", "reviewer": reviewer,
            })
    if skipped_gap_fills:
        print(f"[warn] {trial_id}: skipped {len(skipped_gap_fills)} marker-frame(s) where a "
              f"coordinate was added that the original didn't have (gap-filling is out of "
              f"scope here) -- e.g. {skipped_gap_fills[:5]}")
    return pd.DataFrame(rows, columns=["trial_id", "participant", "frame", "marker_name",
                                       "vicon_label", "manual_label", "status", "confidence",
                                       "reviewer"])
