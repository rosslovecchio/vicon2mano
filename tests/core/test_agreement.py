"""Tests for vicon2mano.core.agreement -- Vicon-vs-manual label agreement.

Measurement-only module: no relabelling or correction, so tests focus on
matching, error arithmetic, and that missing/unmatched observations are
recorded rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vicon2mano.core import agreement as ag


def test_match_markers_by_base_name():
    labels_a = ["P1:Palm1", "P1:Thumb1", "P1:OnlyA"]
    labels_b = ["P1:Palm1", "P1:Thumb1", "P1:OnlyB"]
    m = ag.match_markers(labels_a, labels_b)
    assert m.common == ["Palm1", "Thumb1"]
    assert m.only_a == ["OnlyA"]
    assert m.only_b == ["OnlyB"]
    assert m.idx_a == {"Palm1": 0, "Thumb1": 1, "OnlyA": 2}
    assert m.idx_b == {"Palm1": 0, "Thumb1": 1, "OnlyB": 2}


def test_per_frame_table_matched_error():
    # 2 frames, 1 common marker, a known 3-4-0 displacement -> error 5.
    markers_a = np.array([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])
    markers_b = np.array([[[3.0, 4.0, 0.0]], [[10.0, 0.0, 0.0]]])
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    assert len(df) == 2
    assert (df["match_status"] == "matched").all()
    np.testing.assert_allclose(df["euclidean_error_mm"].to_numpy(), [5.0, 0.0])
    np.testing.assert_allclose(df["dx_mm"].to_numpy(), [-3.0, 0.0])


def test_per_frame_table_missing_coordinates_kept_not_dropped():
    markers_a = np.array([[[np.nan, np.nan, np.nan]], [[1.0, 2.0, 3.0]]])
    markers_b = np.array([[[1.0, 2.0, 3.0]], [[np.nan, np.nan, np.nan]]])
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    assert len(df) == 2  # both rows kept despite missing coordinates
    assert df.loc[0, "match_status"] == "missing_vicon_coordinates"
    assert df.loc[1, "match_status"] == "missing_manual_coordinates"
    assert df["euclidean_error_mm"].isna().all()


def test_per_frame_table_unequal_frame_counts_truncates_and_warns(capsys):
    markers_a = np.zeros((5, 1, 3))
    markers_b = np.zeros((3, 1, 3))
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    assert len(df) == 3
    assert "differ" in capsys.readouterr().out


def test_add_threshold_columns():
    markers_a = np.array([[[0.0, 0.0, 0.0]]])
    markers_b = np.array([[[0.6, 0.0, 0.0]]])
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.add_threshold_columns(df, [0.5, 1.0])
    assert not df.loc[0, "within_0.5mm"]
    assert df.loc[0, "within_1mm"]


def test_flag_temporal_jumps_marks_outlier_not_baseline():
    errors = np.array([1.0, 1.1, 0.9, 1.0, 50.0, 1.0])
    markers_a = np.zeros((len(errors), 1, 3))
    markers_b = markers_a.copy()
    markers_b[:, 0, 0] = errors  # euclidean error == this column's value
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.flag_temporal_jumps(df, k=3.0, min_jump_mm=2.0)
    assert df.loc[4, "error_jump_flag"]
    assert not df.loc[0, "error_jump_flag"]


def test_flag_temporal_jumps_zero_inflated_data_does_not_flag_every_nonzero():
    # Mostly-zero error with a few small nonzero values and one genuine
    # outlier -- median/MAD collapse to 0 here, so without a floor every
    # nonzero-error frame (including the harmless 0.3mm ones) would be
    # flagged. This reproduces the bug found on real zero-inflated data
    # (a manually-labelled export that mostly reuses the Vicon trajectory).
    errors = np.array([0, 0, 0, 0, 0, 0.3, 0, 0.2, 0, 80.0, 0, 0])
    markers_a = np.zeros((len(errors), 1, 3))
    markers_b = markers_a.copy()
    markers_b[:, 0, 0] = errors
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.flag_temporal_jumps(df, k=6.0, min_jump_mm=2.0)
    assert df["error_jump_flag"].sum() == 1
    assert df.loc[df["euclidean_error_mm"] == 80.0, "error_jump_flag"].all()


def test_temporal_events_groups_contiguous_frames_per_marker():
    df = pd.DataFrame({
        "marker": ["M1"] * 6,
        "frame": [0, 1, 2, 10, 11, 20],
        "euclidean_error_mm": [50.0, 60.0, 55.0, 40.0, 45.0, 30.0],
        "error_jump_flag": [True, True, True, True, True, True],
        "match_status": ["matched"] * 6,
    })
    events = ag.temporal_events(df)
    assert len(events) == 3
    assert events.iloc[0][["start_frame", "end_frame", "n_observations"]].tolist() == [0, 2, 3]
    assert events.iloc[1][["start_frame", "end_frame", "n_observations"]].tolist() == [10, 11, 2]
    assert events.iloc[2][["start_frame", "end_frame", "n_observations"]].tolist() == [20, 20, 1]
    assert events.iloc[0]["event_type"] == "sustained_label_disagreement"
    assert events.iloc[2]["event_type"] == "single_frame_discontinuity"


def test_temporal_events_flags_adjacency_to_missing_vicon():
    df = pd.DataFrame({
        "marker": ["M1"] * 4,
        "frame": [0, 1, 2, 3],
        "euclidean_error_mm": [np.nan, 50.0, 60.0, np.nan],
        "error_jump_flag": [False, True, True, False],
        "match_status": ["missing_vicon_coordinates", "matched", "matched",
                          "missing_vicon_coordinates"],
    })
    events = ag.temporal_events(df)
    assert len(events) == 1
    assert events.iloc[0]["adjacent_missing_vicon"]


def test_temporal_events_not_adjacent_to_missing_vicon_when_surrounded_by_matches():
    df = pd.DataFrame({
        "marker": ["M1"] * 5,
        "frame": [0, 1, 2, 3, 4],
        "euclidean_error_mm": [0.0, 50.0, 60.0, 0.0, 0.0],
        "error_jump_flag": [False, True, True, False, False],
        "match_status": ["matched"] * 5,
    })
    events = ag.temporal_events(df)
    assert len(events) == 1
    assert not events.iloc[0]["adjacent_missing_vicon"]


def test_missingness_events_groups_contiguous_gaps():
    df = pd.DataFrame({
        "marker": ["M1"] * 6,
        "frame": [0, 1, 2, 3, 4, 5],
        "match_status": ["matched", "missing_vicon_coordinates",
                          "missing_vicon_coordinates", "matched",
                          "missing_both_coordinates", "matched"],
    })
    gaps = ag.missingness_events(df)
    assert len(gaps) == 2
    assert gaps.iloc[0][["start_frame", "end_frame", "n_observations"]].tolist() == [1, 2, 2]
    assert gaps.iloc[1][["start_frame", "end_frame", "n_observations"]].tolist() == [4, 4, 1]


def test_per_marker_event_summary_aggregates_durations_and_types():
    events = pd.DataFrame({
        "marker": ["M1", "M1", "M2"],
        "start_frame": [0, 10, 0],
        "end_frame": [2, 10, 4],
        "n_observations": [3, 1, 5],
        "event_type": ["sustained_label_disagreement", "single_frame_discontinuity",
                       "sustained_label_disagreement"],
        "adjacent_missing_vicon": [False, True, False],
    })
    summary = ag.per_marker_event_summary(events)
    m1 = summary.set_index("marker").loc["M1"]
    assert m1["event_count"] == 2
    assert m1["affected_observations"] == 4
    assert m1["max_event_duration"] == 3
    assert m1["n_single_frame_events"] == 1
    assert m1["n_sustained_events"] == 1
    assert m1["n_adjacent_missing_vicon"] == 1
    # sorted by affected_observations descending -> M2 (5) before M1 (4)
    assert summary.iloc[0]["marker"] == "M2"


def test_per_marker_summary_uses_nan_when_no_matches():
    markers_a = np.array([[[np.nan, np.nan, np.nan]]])
    markers_b = np.array([[[1.0, 2.0, 3.0]]])
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.add_threshold_columns(df, [5.0])
    summary = ag.per_marker_summary(df, [5.0])
    row = summary.iloc[0]
    assert row["matched"] == 0
    assert np.isnan(row["mean_error_mm"])
    assert row["missing_vicon_coordinates"] == 1


def test_overall_summary_matches_marker_level_counts():
    markers_a = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    markers_b = np.array([[[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]])
    match = ag.match_markers(["M1", "M2"], ["M1", "M2"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.add_threshold_columns(df, [2.0])
    overall = ag.overall_summary(df, [2.0])
    assert overall["n_rows"] == 2
    assert overall["matched"] == 1
    assert overall["n_markers"] == 2
    assert overall["pct_within_2mm"] == pytest.approx(50.0)


def test_reliability_table_separates_availability_from_retention():
    # 4 frames: M1 available in all 4, agrees (zero error) in 3, missing in 0.
    # M2 available in only 2 of 4 frames, agrees in both available ones --
    # perfect retention-given-availability, but only 50% overall usable.
    markers_a = np.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[5.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
    ])
    markers_b = np.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
    ])
    match = ag.match_markers(["M1", "M2"], ["M1", "M2"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.add_threshold_columns(df, [1.0])
    df = ag.flag_temporal_jumps(df)
    events = ag.temporal_events(df)
    event_summary = ag.per_marker_event_summary(events)
    missingness = ag.missingness_events(df)
    marker_summary = ag.per_marker_summary(df, [1.0])

    table = ag.reliability_table(marker_summary, event_summary, missingness)
    m1 = table.set_index("marker").loc["M1"]
    m2 = table.set_index("marker").loc["M2"]

    assert m1["availability_pct"] == pytest.approx(100.0)
    assert m1["label_retention_pct_valid"] == pytest.approx(75.0)
    assert m1["overall_usable_pct"] == pytest.approx(75.0)

    assert m2["availability_pct"] == pytest.approx(50.0)
    assert m2["label_retention_pct_valid"] == pytest.approx(100.0)
    # perfect retention-given-availability, but only usable half the time
    assert m2["overall_usable_pct"] == pytest.approx(50.0)


def test_cross_trial_summary_single_trial_has_nan_std_and_equal_min_max():
    all_reliability = pd.DataFrame({
        "marker": ["M1"],
        "availability_pct": [90.0],
        "label_retention_pct_valid": [95.0],
        "overall_usable_pct": [85.5],
        "event_count": [3],
        "sustained_event_count": [1],
        "max_gap_duration": [10],
    })
    summary = ag.cross_trial_summary(all_reliability)
    row = summary.iloc[0]
    assert row["n_trials"] == 1
    assert np.isnan(row["overall_usable_pct_std"])
    assert row["overall_usable_pct_min"] == row["overall_usable_pct_max"] == pytest.approx(85.5)


def test_cross_trial_summary_multiple_trials_aggregates_correctly():
    all_reliability = pd.DataFrame({
        "marker": ["M1", "M1", "M2", "M2"],
        "availability_pct": [80.0, 90.0, 100.0, 100.0],
        "label_retention_pct_valid": [90.0, 95.0, 100.0, 90.0],
        "overall_usable_pct": [72.0, 85.5, 100.0, 90.0],
        "event_count": [3, 5, 0, 1],
        "sustained_event_count": [1, 2, 0, 0],
        "max_gap_duration": [10, 20, 0, 5],
    })
    summary = ag.cross_trial_summary(all_reliability).set_index("marker")
    m1 = summary.loc["M1"]
    assert m1["n_trials"] == 2
    assert m1["overall_usable_pct_mean"] == pytest.approx((72.0 + 85.5) / 2)
    assert m1["overall_usable_pct_min"] == pytest.approx(72.0)
    assert m1["overall_usable_pct_max"] == pytest.approx(85.5)
    assert m1["total_event_count"] == 8
    assert m1["max_gap_duration"] == 20
    # sorted ascending by overall_usable_pct_mean -> M1 (worse) before M2
    assert summary.reset_index().iloc[0]["marker"] == "M1"


def test_reliability_table_zero_events_and_gaps_are_zero_not_nan():
    markers_a = np.zeros((3, 1, 3))
    markers_b = np.zeros((3, 1, 3))
    match = ag.match_markers(["M1"], ["M1"])
    df = ag.per_frame_table(markers_a, markers_b, match)
    df = ag.add_threshold_columns(df, [1.0])
    df = ag.flag_temporal_jumps(df)
    events = ag.temporal_events(df)
    event_summary = ag.per_marker_event_summary(events)
    missingness = ag.missingness_events(df)
    marker_summary = ag.per_marker_summary(df, [1.0])

    table = ag.reliability_table(marker_summary, event_summary, missingness)
    row = table.iloc[0]
    assert row["event_count"] == 0
    assert row["sustained_event_count"] == 0
    assert row["max_event_duration"] == 0
    assert row["max_gap_duration"] == 0
    assert row["overall_usable_pct"] == pytest.approx(100.0)
