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


# ---------------------------------------------------------------------------
# Palm-local-frame invariance (the coordinate-frame-offset fix)
# ---------------------------------------------------------------------------


def _palm_hand(T, rng):
    """Palm1/2/3 (rigid triangle) plus one finger marker rigidly attached
    to the palm -- a minimal "hand" for testing frame invariance."""
    labels = ["Palm1", "Palm2", "Palm3", "Finger1"]
    local = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 50.0, 0.0], [20.0, 20.0, 30.0]])
    out = np.zeros((T, 4, 3))
    for t in range(T):
        th = 0.1 * t
        c, s = np.cos(th), np.sin(th)
        Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([t * 2.0, t * 1.5, 0.0])
        out[t] = local @ Rw.T + shift
        if rng is not None:
            out[t] += rng.normal(scale=0.01, size=out[t].shape)
    return out, labels


def test_to_palm_local_frame_returns_none_without_anchors():
    markers = np.zeros((3, 2, 3))
    assert val.to_palm_local_frame(markers, ["A", "B"]) is None


def test_to_palm_local_frame_cancels_global_rigid_motion():
    rng = np.random.default_rng(0)
    hand, labels = _palm_hand(20, rng)

    # A second "file" of the same hand, globally translated + rotated --
    # simulating Nexus's different coordinate reference, with nobody having
    # touched any label.
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    Rglobal = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shifted = hand @ Rglobal.T + np.array([500.0, -300.0, 100.0])

    local_a = val.to_palm_local_frame(hand, labels)
    local_b = val.to_palm_local_frame(shifted, labels)
    np.testing.assert_allclose(local_a, local_b, atol=0.1)


def test_build_annotations_from_correction_ignores_global_offset_with_local_frame():
    from vicon2mano.core.agreement import match_markers

    hand, labels = _palm_hand(5, rng=None)
    # Globally shift+rotate the "corrected" file -- same real-world data,
    # different coordinate system, nothing actually relabelled.
    th = 0.3
    c, s = np.cos(th), np.sin(th)
    Rglobal = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corrected = hand @ Rglobal.T + np.array([200.0, 150.0, -50.0])

    out = val.build_annotations_from_correction(
        hand, labels, corrected, labels, [2],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers, use_palm_local_frame=True)
    assert (out.set_index("marker_name")["status"] == "correct").all()


def test_build_annotations_from_correction_raw_frame_false_positives_on_global_offset():
    # Without the fix, the same globally-shifted data would be misread as
    # every marker being wrong -- documents the bug this feature fixes.
    from vicon2mano.core.agreement import match_markers

    hand, labels = _palm_hand(5, rng=None)
    th = 0.3
    c, s = np.cos(th), np.sin(th)
    Rglobal = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corrected = hand @ Rglobal.T + np.array([200.0, 150.0, -50.0])

    out = val.build_annotations_from_correction(
        hand, labels, corrected, labels, [2],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers, use_palm_local_frame=False)
    assert (out.set_index("marker_name")["status"] == "wrong_label").all()


def test_build_annotations_from_correction_still_detects_real_swap_in_local_frame():
    from vicon2mano.core.agreement import match_markers

    hand, labels = _palm_hand(5, rng=None)
    corrected = hand.copy()
    # Genuinely swap Finger1 with a far-away position at frame 2 (simulating
    # a real mislabel), on top of an unrelated global coordinate shift.
    corrected[2, 3] = corrected[2, 3] + np.array([80.0, 0.0, 0.0])
    th = 0.3
    c, s = np.cos(th), np.sin(th)
    Rglobal = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corrected = corrected @ Rglobal.T + np.array([200.0, 150.0, -50.0])

    out = val.build_annotations_from_correction(
        hand, labels, corrected, labels, [2],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers, use_palm_local_frame=True)
    by_marker = out.set_index("marker_name")
    assert by_marker.loc["Finger1", "status"] == "wrong_label"
    assert by_marker.loc["Palm1", "status"] == "correct"


def test_build_annotations_from_correction_warns_and_falls_back_without_anchors(capsys):
    from vicon2mano.core.agreement import match_markers

    labels = ["A", "B"]
    markers_orig = np.zeros((2, 2, 3))
    markers_corrected = markers_orig.copy()

    val.build_annotations_from_correction(
        markers_orig, labels, markers_corrected, labels, [0],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers, use_palm_local_frame=True)
    assert "falling back to raw world coordinates" in capsys.readouterr().out


def test_build_annotations_from_correction_anchor_occlusion_is_ambiguous_not_missing():
    # Regression test for a real finding: the "poor_availability" sampling
    # category specifically targets frames with low marker availability,
    # which correlates with the PALM ANCHORS themselves being occluded --
    # that undefines the local frame for every other marker too, but a
    # marker that is genuinely present in both raw files must not be
    # reported as "missing" or "ghost" just because of that, since neither
    # claim is true about the marker itself. It must come out "ambiguous".
    from vicon2mano.core.agreement import match_markers

    hand, labels = _palm_hand(3, rng=None)
    corrected = hand.copy()
    # Occlude Palm1 in the corrected file at frame 1 only -- Finger1 itself
    # stays fully present (raw-valid) in both files at that frame.
    corrected[1, 0] = np.nan

    out = val.build_annotations_from_correction(
        hand, labels, corrected, labels, [1],
        participant="P1", trial_id="T1", reviewer="tester",
        match_markers_fn=match_markers, use_palm_local_frame=True)
    by_marker = out.set_index("marker_name")
    assert by_marker.loc["Finger1", "status"] == "ambiguous"
    # Palm1 itself is raw-missing in corrected -> genuinely "ghost", not
    # "ambiguous" (its own absence is the real, determinate finding here).
    assert by_marker.loc["Palm1", "status"] == "ghost"


def test_palm_triangle_bone_deltas_zero_for_rigid_unchanged_triangle():
    hand, labels = _palm_hand(3, rng=None)
    deltas = val.palm_triangle_bone_deltas(hand, labels, hand, labels, frame=1)
    assert set(deltas) == {"Palm1-Palm2", "Palm2-Palm3", "Palm1-Palm3"}
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in deltas.values())


def test_palm_triangle_bone_deltas_detects_distorted_triangle():
    hand, labels = _palm_hand(3, rng=None)
    corrected = hand.copy()
    corrected[1, 2] += np.array([20.0, 0.0, 0.0])  # move Palm3 at frame 1 only
    deltas = val.palm_triangle_bone_deltas(hand, labels, corrected, labels, frame=1)
    assert deltas["Palm2-Palm3"] > 1.0
    assert deltas["Palm1-Palm3"] > 1.0
    assert deltas["Palm1-Palm2"] == pytest.approx(0.0, abs=1e-9)


def test_palm_triangle_bone_deltas_empty_without_anchor_triple():
    markers = np.zeros((2, 2, 3))
    assert val.palm_triangle_bone_deltas(markers, ["A", "B"], markers, ["A", "B"], frame=0) == {}


def test_palm_triangle_bone_deltas_partial_when_one_anchor_missing_that_frame():
    # Palm1 missing invalidates only the two bones touching it
    # (Palm1-Palm2, Palm1-Palm3); Palm2-Palm3 doesn't involve Palm1 at all,
    # so it stays computable.
    hand, labels = _palm_hand(3, rng=None)
    corrected = hand.copy()
    corrected[1, 0] = np.nan  # Palm1 missing at frame 1
    deltas = val.palm_triangle_bone_deltas(hand, labels, corrected, labels, frame=1)
    assert set(deltas) == {"Palm2-Palm3"}
