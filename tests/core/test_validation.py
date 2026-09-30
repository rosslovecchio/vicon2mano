"""Tests for vicon2mano.core.validation (assessment_type =
manual_identity_validation) -- the independent 500-frame manual sample,
kept deliberately separate from both core.agreement (P10 label retention)
and core.geometric_consistency (assessment_type = geometric_self_consistency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vicon2mano.core import validation as val


def _annotations(rows):
    """rows: list of (participant, trial_id, frame, marker_name, status)."""
    return pd.DataFrame(rows, columns=["participant", "trial_id", "frame", "marker_name", "status"])


def test_label_accuracy_summary_basic_counts():
    df = _annotations([
        ("P1", "T1", 0, "A", "correct"),
        ("P1", "T1", 0, "B", "correct"),
        ("P1", "T1", 1, "A", "wrong_label"),
        ("P1", "T1", 1, "B", "missing"),
        ("P1", "T1", 2, "A", "ambiguous"),
        ("P1", "T1", 2, "B", "ghost"),
    ])
    out = val.label_accuracy_summary(df)
    assert out["n_judgements"] == 6
    assert out["n_correct"] == 2
    assert out["n_wrong_label"] == 1
    assert out["n_missing"] == 1
    assert out["n_ambiguous"] == 1
    assert out["n_ghost"] == 1
    # determinate = correct + wrong_label + ghost = 2 + 1 + 1 = 4
    assert out["n_determinate"] == 4
    assert out["accuracy"] == pytest.approx(2 / 4)
    assert out["n_frames"] == 3
    # frame 0: all correct -> counts. frame 1: has wrong_label -> doesn't.
    # frame 2: has ambiguous+ghost -> doesn't.
    assert out["n_frames_all_correct"] == 1
    assert out["pct_frames_all_correct"] == pytest.approx(100 / 3)


def test_label_accuracy_summary_missing_only_frame_counts_as_all_correct():
    df = _annotations([
        ("P1", "T1", 0, "A", "correct"),
        ("P1", "T1", 0, "B", "missing"),
    ])
    out = val.label_accuracy_summary(df)
    assert out["n_frames_all_correct"] == 1
    assert out["pct_frames_all_correct"] == pytest.approx(100.0)


def test_accuracy_nan_when_no_determinate_judgements():
    df = _annotations([
        ("P1", "T1", 0, "A", "missing"),
        ("P1", "T1", 0, "B", "ambiguous"),
    ])
    out = val.label_accuracy_summary(df)
    assert out["n_determinate"] == 0
    assert np.isnan(out["accuracy"])


def test_per_marker_accuracy_sorted_worst_first():
    df = _annotations([
        ("P1", "T1", 0, "Good", "correct"),
        ("P1", "T1", 1, "Good", "correct"),
        ("P1", "T1", 0, "Bad", "wrong_label"),
        ("P1", "T1", 1, "Bad", "correct"),
    ])
    out = val.per_marker_accuracy(df)
    assert out.iloc[0]["marker_name"] == "Bad"
    assert out.iloc[0]["accuracy"] == pytest.approx(0.5)
    assert out.iloc[1]["marker_name"] == "Good"
    assert out.iloc[1]["accuracy"] == pytest.approx(1.0)


def test_per_trial_accuracy_groups_by_participant_and_trial():
    df = _annotations([
        ("P1", "T1", 0, "A", "correct"),
        ("P1", "T2", 0, "A", "wrong_label"),
        ("P2", "T1", 0, "A", "correct"),
    ])
    out = val.per_trial_accuracy(df)
    assert len(out) == 3
    row_p1t2 = out[(out.participant == "P1") & (out.trial_id == "T2")].iloc[0]
    assert row_p1t2["accuracy"] == pytest.approx(0.0)


def test_frame_geometric_status_inside_and_outside_event_span():
    events = pd.DataFrame({
        "start_frame": [10, 50], "end_frame": [15, 50],
        "deviation_magnitude_mm": [30.0, 90.0],
    })
    assert val.frame_geometric_status(events, 12) == (True, 30.0)
    assert val.frame_geometric_status(events, 50) == (True, 90.0)
    assert val.frame_geometric_status(events, 20) == (False, 0.0)


def test_frame_geometric_status_empty_events():
    assert val.frame_geometric_status(pd.DataFrame(columns=["start_frame", "end_frame",
                                                             "deviation_magnitude_mm"]), 5) == (False, 0.0)
    assert val.frame_geometric_status(None, 5) == (False, 0.0)


def test_frame_geometric_status_uses_max_over_overlapping_events():
    events = pd.DataFrame({
        "start_frame": [0, 0], "end_frame": [20, 20],
        "deviation_magnitude_mm": [10.0, 99.0],
    })
    assert val.frame_geometric_status(events, 5) == (True, 99.0)


def test_geometry_vs_manual_table_excludes_ambiguous_frames():
    df = _annotations([
        ("P1", "T1", 0, "A", "correct"),
        ("P1", "T1", 1, "A", "ambiguous"),
        ("P1", "T1", 1, "B", "correct"),
    ])
    out = val.geometry_vs_manual_table(df, {})
    assert len(out) == 1
    assert out.iloc[0]["frame"] == 0


def test_geometry_vs_manual_table_availability_and_error_flags():
    df = _annotations([
        ("P1", "T1", 0, "A", "correct"),
        ("P1", "T1", 0, "B", "missing"),
        ("P1", "T1", 1, "A", "wrong_label"),
        ("P1", "T1", 1, "B", "correct"),
    ])
    events = {("P1", "T1"): pd.DataFrame({
        "start_frame": [1], "end_frame": [1], "deviation_magnitude_mm": [50.0],
    })}
    out = val.geometry_vs_manual_table(df, events).set_index("frame")
    assert out.loc[0, "trajectory_availability"] == pytest.approx(0.5)
    assert not out.loc[0, "manual_error_present"]
    assert not out.loc[0, "geometric_anomaly"]
    assert out.loc[1, "trajectory_availability"] == pytest.approx(1.0)
    assert out.loc[1, "manual_error_present"]
    assert out.loc[1, "geometric_anomaly"]
    assert out.loc[1, "geometric_score"] == pytest.approx(50.0)


def test_geometry_vs_manual_contingency_2x2_shape_and_counts():
    per_frame = pd.DataFrame({
        "geometric_anomaly": [False, False, True, True, True],
        "manual_error_present": [False, True, False, True, True],
    })
    table = val.geometry_vs_manual_contingency(per_frame)
    assert len(table) == 4
    lookup = {(r["Geometry"], r["Manual error"]): r["Count"] for _, r in table.iterrows()}
    assert lookup[("Normal", "No")] == 1
    assert lookup[("Normal", "Yes")] == 1
    assert lookup[("Anomalous", "No")] == 1
    assert lookup[("Anomalous", "Yes")] == 2


# ---------------------------------------------------------------------------
# Coordinate-diff classification (the "relabel in Nexus and diff" workflow)
# ---------------------------------------------------------------------------


def test_classify_marker_frame_both_missing_is_missing():
    assert val.classify_marker_frame(False, False, False) == "missing"


def test_classify_marker_frame_label_removed_is_ghost_not_missing():
    # Present before, deleted during review -- per instruction, this means
    # the label was NOT correct (a real error), not a benign data gap.
    assert val.classify_marker_frame(True, False, False) == "ghost"


def test_classify_marker_frame_unchanged_is_correct():
    assert val.classify_marker_frame(True, True, True) == "correct"


def test_classify_marker_frame_moved_is_wrong_label():
    assert val.classify_marker_frame(True, True, False) == "wrong_label"


def test_classify_marker_frame_added_is_none_gap_fill_out_of_scope():
    assert val.classify_marker_frame(False, True, False) is None


def test_build_annotations_from_correction_full_workflow():
    from vicon2mano.core.agreement import match_markers

    labels = ["A", "B", "C", "D"]
    T = 3
    markers_orig = np.zeros((T, 4, 3))
    markers_orig[:, 0] = [1.0, 0.0, 0.0]   # A: unchanged everywhere -> correct
    markers_orig[:, 1] = [2.0, 0.0, 0.0]   # B: will be moved at frame 1 -> wrong_label
    markers_orig[:, 2] = [3.0, 0.0, 0.0]   # C: will be deleted at frame 1 -> ghost
    markers_orig[:, 3] = np.nan            # D: missing throughout, untouched -> missing

    markers_corrected = markers_orig.copy()
    markers_corrected[1, 1] = [99.0, 0.0, 0.0]   # B moved
    markers_corrected[1, 2] = np.nan             # C removed

    visited_frames = [1]  # only frame 1 was actually reviewed
    out = val.build_annotations_from_correction(
        markers_orig, labels, markers_corrected, labels, visited_frames,
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers)

    by_marker = out.set_index("marker_name")
    assert by_marker.loc["A", "status"] == "correct"
    assert by_marker.loc["B", "status"] == "wrong_label"
    assert by_marker.loc["C", "status"] == "ghost"
    assert by_marker.loc["D", "status"] == "missing"
    # frame 0 and 2 were never visited -> nothing recorded for them at all.
    assert set(out["frame"]) == {1}
    assert len(out) == 4


def test_build_annotations_from_correction_skips_gap_fill_with_warning(capsys):
    from vicon2mano.core.agreement import match_markers

    labels = ["A"]
    markers_orig = np.full((2, 1, 3), np.nan)
    markers_corrected = markers_orig.copy()
    markers_corrected[0, 0] = [5.0, 0.0, 0.0]  # added where original had nothing

    out = val.build_annotations_from_correction(
        markers_orig, labels, markers_corrected, labels, [0],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers)

    assert len(out) == 0
    assert "gap-filling" in capsys.readouterr().out
