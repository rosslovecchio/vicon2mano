"""Tests for the pure logic in scripts/shared/select_frames_for_review.py
(the 500-frame manual-review sample): dedup/top-up, per-frame availability
reconstruction from gap runs, and window expansion.

Imported by path since scripts/ isn't a package (same convention
scripts/shared/aggregate_manual_agreement.py uses for
analyze_manual_agreement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_SHARED = Path(__file__).resolve().parents[2] / "scripts" / "shared"
if str(SCRIPTS_SHARED) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SHARED))

import select_frames_for_review as sfr  # noqa: E402


def test_per_frame_missing_count_single_gap():
    gaps = pd.DataFrame({"start_frame": [2], "end_frame": [4]})
    counts = sfr.per_frame_missing_count(gaps, 8)
    np.testing.assert_array_equal(counts, [0, 0, 1, 1, 1, 0, 0, 0])


def test_per_frame_missing_count_overlapping_gaps_stack():
    gaps = pd.DataFrame({"start_frame": [1, 3], "end_frame": [5, 4]})
    counts = sfr.per_frame_missing_count(gaps, 8)
    np.testing.assert_array_equal(counts, [0, 1, 1, 2, 2, 1, 0, 0])


def test_per_frame_missing_count_gap_touching_last_frame():
    gaps = pd.DataFrame({"start_frame": [6], "end_frame": [7]})
    counts = sfr.per_frame_missing_count(gaps, 8)
    np.testing.assert_array_equal(counts, [0, 0, 0, 0, 0, 0, 1, 1])


def test_select_category_deduplicates_and_tops_up_from_same_pool():
    candidates = pd.DataFrame({
        "participant": ["P1"] * 5, "trial_id": ["T1"] * 5,
        "frame": [10, 20, 30, 40, 50],
    })
    taken = {("P1", "T1", 20)}
    picked = sfr.select_category(candidates, 3, taken, "cat")
    assert list(picked["frame"]) == [10, 30, 40]
    assert (picked["category"] == "cat").all()
    # taken is mutated in place with the newly-picked frames too.
    assert ("P1", "T1", 10) in taken


def test_select_category_stops_at_quota_even_with_more_candidates():
    candidates = pd.DataFrame({
        "participant": ["P1"] * 5, "trial_id": ["T1"] * 5,
        "frame": [1, 2, 3, 4, 5],
    })
    picked = sfr.select_category(candidates, 2, set(), "cat")
    assert len(picked) == 2


def test_select_category_returns_fewer_than_quota_when_pool_exhausted():
    candidates = pd.DataFrame({
        "participant": ["P1"] * 2, "trial_id": ["T1"] * 2, "frame": [1, 2],
    })
    picked = sfr.select_category(candidates, 5, set(), "cat")
    assert len(picked) == 2


def test_select_category_carries_selection_metric_when_present():
    candidates = pd.DataFrame({
        "participant": ["P1", "P1"], "trial_id": ["T1", "T1"],
        "frame": [1, 2], "deviation_magnitude_mm": [99.0, 50.0],
    })
    picked = sfr.select_category(candidates, 2, set(), "cat",
                                 reason_col="deviation_magnitude_mm")
    assert list(picked["selection_metric"]) == [99.0, 50.0]


def test_worst_availability_frames_ranks_across_trials():
    overview = pd.DataFrame({"participant": ["P1", "P2"], "trial_id": ["T1", "T1"]})
    gaps_by_trial = {
        ("P1", "T1"): pd.DataFrame({"start_frame": [5], "end_frame": [5]}),
        ("P2", "T1"): pd.DataFrame({"start_frame": [3, 3], "end_frame": [6, 6]}),
    }
    n_frames_by_trial = {("P1", "T1"): 10, ("P2", "T1"): 10}
    worst = sfr.worst_availability_frames(overview, gaps_by_trial, n_frames_by_trial, top_k=5)
    assert worst.iloc[0][["participant", "frame", "n_markers_missing"]].tolist() == ["P2", 3, 2]


def test_build_windows_expands_and_clips_to_trial_bounds():
    anchors = pd.DataFrame({
        "participant": ["P1"], "trial_id": ["T1"], "frame": [2], "category": ["random"],
    })
    n_frames_by_trial = {("P1", "T1"): 6}
    windows = sfr.build_windows(anchors, n_frames_by_trial)
    # radius 5 around frame 2 in a 6-frame trial (valid 0..5) clips to [0,5].
    assert sorted(windows["frame"]) == list(range(0, 6))
    assert windows.loc[windows["frame"] == 2, "is_anchor"].iloc[0]
    assert not windows.loc[windows["frame"] == 0, "is_anchor"].iloc[0]


def test_build_windows_merges_overlapping_anchors_without_duplicate_rows():
    anchors = pd.DataFrame({
        "participant": ["P1", "P1"], "trial_id": ["T1", "T1"],
        "frame": [10, 12], "category": ["random", "largest_anomaly"],
    })
    n_frames_by_trial = {("P1", "T1"): 100}
    windows = sfr.build_windows(anchors, n_frames_by_trial)
    # frames 5..17 covered by the union of [5,15] and [7,17], no duplicates.
    assert len(windows) == len(set(windows["frame"]))
    assert sorted(windows["frame"]) == list(range(5, 18))
    overlap_row = windows[windows["frame"] == 12].iloc[0]
    assert "random" in overlap_row["categories"] and "largest_anomaly" in overlap_row["categories"]
