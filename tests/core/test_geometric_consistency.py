"""Tests for vicon2mano.core.geometric_consistency (assessment_type =
"geometric_self_consistency").

Deliberately separate from tests/core/test_agreement.py, matching the
module split: this covers bone-length self-consistency only, never the
Vicon-vs-manual-label comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vicon2mano.core import bones as bn
from vicon2mano.core import geometric_consistency as gc


def _rigid_triangle(T, noise=0.0, rng=None):
    """3 points, 3-4-5 triangle, rigidly translated/rotated per frame."""
    rng = rng or np.random.default_rng(0)
    pts0 = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [30.0, 40.0, 0.0]])
    out = np.zeros((T, 3, 3))
    for t in range(T):
        th = 0.05 * t
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[t] = pts0 @ R.T + np.array([t * 0.2, 0.0, 0.0])
        if noise:
            out[t] += rng.normal(scale=noise, size=out[t].shape)
    return out


LABELS = ["A", "B", "C"]
BONES = [(0, 1, "A-B"), (1, 2, "B-C"), (0, 2, "A-C")]


def test_constant_bone_lengths_zero_deviation_and_no_flags():
    markers = _rigid_triangle(60)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    df = gc.build_marker_frame_table(markers, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    assert np.isfinite(df["deviation_mm"]).all()
    np.testing.assert_allclose(df["deviation_mm"], 0.0, atol=1e-6)
    df = gc.flag_suspicious_events(df)
    assert not df["suspicious_flag"].any()


def test_zero_mad_data_does_not_flag_every_nonzero_deviation():
    # Mostly-zero deviation with a few small nonzero readings and one real
    # anomaly -- median/MAD collapse to 0 here (matches real bone-length
    # data: a marker is consistent almost always), so without the floor
    # every nonzero deviation, including harmless 0.5mm jitter, would be
    # flagged. Mirrors agreement.flag_temporal_jumps's own zero-MAD guard.
    dev = np.array([0, 0, 0, 0, 0.5, 0, 0.3, 0, 40.0, 0, 0])
    df = pd.DataFrame({
        "subject_id": "S1", "trial_id": "T1", "frame": np.arange(len(dev)),
        "marker": "A", "deviation_mm": dev, "associated_bone": "A-B",
        "vicon_available": True,
    })
    flagged = gc.flag_suspicious_events(df, k=6.0, min_deviation_mm=2.0)
    assert flagged["suspicious_flag"].sum() == 1
    assert flagged.loc[flagged["deviation_mm"] == 40.0, "suspicious_flag"].all()


def test_missing_markers_produce_nan_deviation_not_crash():
    markers = _rigid_triangle(20)
    markers[5:10, 1] = np.nan   # marker B missing for 5 frames
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    df = gc.build_marker_frame_table(markers, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    b_rows = df[df["marker"] == "B"].set_index("frame")
    assert not b_rows.loc[5:9, "vicon_available"].any()
    assert b_rows.loc[5:9, "deviation_mm"].isna().all()
    # Marker A's incident bone A-B is also unusable while B is missing, but
    # A-C survives (C unaffected), so A should NOT go NaN.
    a_rows = df[df["marker"] == "A"].set_index("frame")
    assert np.isfinite(a_rows.loc[5:9, "deviation_mm"]).all()

    df = gc.add_threshold_columns(df, [1.0])
    summary = gc.per_marker_summary(df, gc.geometric_events(gc.flag_suspicious_events(df)),
                                    pd.DataFrame(columns=["marker", "duration"]), [1.0])
    b_summary = summary.set_index("marker").loc["B"]
    assert b_summary["valid_vicon_observations"] == 15
    assert b_summary["availability_pct"] == pytest.approx(75.0)


def test_marker_with_no_incident_bones_is_nan_not_error():
    markers = np.zeros((10, 4, 3))
    labels = ["A", "B", "C", "Lonely"]
    bones = [(0, 1, "A-B")]  # marker "Lonely" (index 3) has no bones at all
    ref, _ = bn.consensus_bone_lengths(markers, [(0, 1)], min_bones=1)
    df = gc.build_marker_frame_table(markers, labels, bones, ref,
                                     subject_id="S1", trial_id="T1")
    lonely = df[df["marker"] == "Lonely"]
    assert lonely["deviation_mm"].isna().all()
    assert lonely["associated_bone"].isna().all()


def test_sudden_marker_swap_detected_as_single_frame_event():
    markers = _rigid_triangle(60)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    swapped = markers.copy()
    swapped[30, [0, 1]] = swapped[30, [1, 0]]
    df = gc.build_marker_frame_table(swapped, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    df = gc.flag_suspicious_events(df, min_deviation_mm=2.0)
    events = gc.geometric_events(df)
    assert len(events) >= 1
    assert (events["start_frame"] == 30).any()
    single = events[events["start_frame"] == 30].iloc[0]
    assert single["event_type"] == "single_frame_discontinuity"
    assert single["duration"] == 1
    assert single["associated_bone"] is not None
    assert single["peak_frame"] == 30


def test_sustained_deviation_detected_as_multi_frame_event():
    markers = _rigid_triangle(60)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    corrupted = markers.copy()
    # Push marker C away from the triangle for a sustained 10-frame block --
    # a persistent mislabelling/drift regime, not a single-instant blip.
    corrupted[20:30, 2] += np.array([50.0, 0.0, 0.0])
    df = gc.build_marker_frame_table(corrupted, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    df = gc.flag_suspicious_events(df, min_deviation_mm=2.0)
    events = gc.geometric_events(df)
    sustained = events[events["event_type"] == "sustained_geometric_deviation"]
    assert len(sustained) >= 1
    assert sustained["duration"].max() >= 8


def test_gaps_grouped_contiguously():
    markers = _rigid_triangle(20)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    markers[3:6, 0] = np.nan
    markers[15, 0] = np.nan
    df = gc.build_marker_frame_table(markers, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    gaps = gc.missingness_events(df, subject_id="S1", trial_id="T1")
    a_gaps = gaps[gaps["marker"] == "A"].sort_values("start_frame")
    assert len(a_gaps) == 2
    assert a_gaps.iloc[0][["start_frame", "end_frame", "duration"]].tolist() == [3, 5, 3]
    assert a_gaps.iloc[1][["start_frame", "end_frame", "duration"]].tolist() == [15, 15, 1]


def test_event_adjacent_to_missing_vicon_flagged():
    markers = _rigid_triangle(15)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    corrupted = markers.copy()
    corrupted[6, [0, 1]] = corrupted[6, [1, 0]]
    corrupted[5, 0] = np.nan  # gap right before the swap event
    df = gc.build_marker_frame_table(corrupted, LABELS, BONES, ref,
                                     subject_id="S1", trial_id="T1")
    df = gc.flag_suspicious_events(df, min_deviation_mm=2.0)
    events = gc.geometric_events(df)
    ev = events[events["start_frame"] == 6]
    assert len(ev) >= 1
    assert ev.iloc[0]["adjacent_missing_vicon"]


def test_reproducibility_same_inputs_same_outputs():
    markers = _rigid_triangle(80, noise=0.05, rng=np.random.default_rng(7))
    bone_ij = [(b[0], b[1]) for b in BONES]
    ref1, inl1 = bn.consensus_bone_lengths(markers, bone_ij, seed=3, min_bones=3)
    ref2, inl2 = bn.consensus_bone_lengths(markers, bone_ij, seed=3, min_bones=3)
    np.testing.assert_array_equal(ref1, ref2)
    np.testing.assert_array_equal(inl1, inl2)

    df1 = gc.build_marker_frame_table(markers, LABELS, BONES, ref1, subject_id="S1", trial_id="T1")
    df2 = gc.build_marker_frame_table(markers, LABELS, BONES, ref2, subject_id="S1", trial_id="T1")
    pd.testing.assert_frame_equal(df1, df2)

    s1 = gc.per_marker_summary(gc.add_threshold_columns(df1, [1.0]),
                               pd.DataFrame(columns=["marker", "duration"]),
                               pd.DataFrame(columns=["marker", "duration"]), [1.0])
    s2 = gc.per_marker_summary(gc.add_threshold_columns(df2, [1.0]),
                               pd.DataFrame(columns=["marker", "duration"]),
                               pd.DataFrame(columns=["marker", "duration"]), [1.0])
    pd.testing.assert_frame_equal(s1, s2)


@pytest.mark.parametrize("filename,expected_type,expected_is_static", [
    ("Static_trial01.csv", "static", True),
    ("Static01.csv", "static", True),
    ("P12_calibration_take.csv", "calibration", True),
    ("Trial2_handsonly.csv", "movement", False),
    ("Trial1_HOI.csv", "movement", False),
    ("weird_filename_9000.csv", "unknown", False),
])
def test_classify_reference_source(filename, expected_type, expected_is_static):
    kind, is_static = gc.classify_reference_source(filename)
    assert kind == expected_type
    assert is_static == expected_is_static


def test_reference_status_valid_when_all_bones_finite():
    assert gc.reference_status(np.array([30.0, 40.0, 50.0])) == "valid"


def test_reference_status_insufficient_when_some_bones_nan():
    assert gc.reference_status(np.array([30.0, np.nan, 50.0])) == "insufficient_bones"


def test_reference_status_no_valid_reference_when_all_nan():
    assert gc.reference_status(np.array([np.nan, np.nan])) == "no_valid_reference"


def test_reference_status_unknown_when_no_bones():
    assert gc.reference_status(np.zeros(0)) == "unknown"


def test_bone_reference_table_reports_length_and_valid_frame_count():
    markers = _rigid_triangle(60)
    bone_ij = [(b[0], b[1]) for b in BONES]
    ref, inliers = bn.consensus_bone_lengths(markers, bone_ij, min_bones=3)
    table = gc.bone_reference_table(markers, BONES, ref, inliers)
    assert list(table["associated_bone"]) == ["A-B", "B-C", "A-C"]
    np.testing.assert_allclose(table["reference_length_mm"], [30.0, 40.0, 50.0], atol=1e-6)
    assert (table["n_valid_frames_used"] == len(inliers)).all()


def test_bone_reference_table_handles_permanently_missing_marker():
    rng = np.random.default_rng(6)
    triangle = _rigid_triangle(60, rng=rng)
    markers = np.full((60, 4, 3), np.nan)
    markers[:, :3] = triangle
    bones = [(0, 1, "A-B"), (2, 3, "C-Lonely")]
    ref, inliers = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in bones], min_bones=1)
    table = gc.bone_reference_table(markers, bones, ref, inliers)
    row_ab = table[table["associated_bone"] == "A-B"].iloc[0]
    row_missing = table[table["associated_bone"] == "C-Lonely"].iloc[0]
    assert np.isfinite(row_ab["reference_length_mm"])
    assert row_ab["n_valid_frames_used"] > 0
    assert np.isnan(row_missing["reference_length_mm"])
    assert row_missing["n_valid_frames_used"] == 0


def test_trial_overall_summary_includes_assessment_type():
    markers = _rigid_triangle(30)
    ref, _ = bn.consensus_bone_lengths(markers, [(b[0], b[1]) for b in BONES], min_bones=3)
    df = gc.add_threshold_columns(
        gc.build_marker_frame_table(markers, LABELS, BONES, ref, subject_id="S1", trial_id="T1"),
        [1.0])
    events = gc.geometric_events(gc.flag_suspicious_events(df))
    gaps = pd.DataFrame(columns=["marker", "duration"])
    summary = gc.per_marker_summary(df, events, gaps, [1.0])
    overall = gc.trial_overall_summary(summary)
    assert overall["assessment_type"] == "geometric_self_consistency"
    assert overall["n_markers"] == 3
    assert overall["total_affected_missing_frames"] == 0
