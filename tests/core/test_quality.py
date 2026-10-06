"""Tests for vicon2mano.core.quality (assessment_type = raw_vicon_quality):
the precedence rule that turns 4 separately-measured signals into one
category per marker-frame, without averaging or weighting any of them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vicon2mano.core import quality as q


def test_manual_review_always_wins_even_if_geometrically_suspicious():
    # A human said "correct" -- an automated suspicious flag must not
    # override that; the human signal has real ground truth, the
    # geometric one does not (see every module's own caveat on this).
    cat = q.classify_marker_frame(
        is_missing=False, manual_status="correct",
        anchor_valid=False, geometric_suspicious=True)
    assert cat == "label_manually_verified"


@pytest.mark.parametrize("status,expected", [
    ("correct", "label_manually_verified"),
    ("wrong_label", "label_manually_wrong"),
    ("ghost", "ghost_candidate"),
    ("ambiguous", "ambiguous"),
    ("missing", "missing"),
])
def test_manual_status_mapping(status, expected):
    assert q.classify_marker_frame(False, status, True, False) == expected


def test_raw_missing_beats_anchor_invalid_and_suspicious_when_unreviewed():
    cat = q.classify_marker_frame(
        is_missing=True, manual_status=None, anchor_valid=False, geometric_suspicious=True)
    assert cat == "missing"


def test_anchor_invalid_beats_suspicious_when_unreviewed():
    cat = q.classify_marker_frame(
        is_missing=False, manual_status=None, anchor_valid=False, geometric_suspicious=True)
    assert cat == "anchor_invalid"


def test_suspicious_when_anchor_valid_and_unreviewed():
    cat = q.classify_marker_frame(
        is_missing=False, manual_status=None, anchor_valid=True, geometric_suspicious=True)
    assert cat == "observed_suspicious"


def test_plausible_is_the_default_fallback():
    cat = q.classify_marker_frame(
        is_missing=False, manual_status=None, anchor_valid=True, geometric_suspicious=False)
    assert cat == "observed_plausible"


def test_resolve_manual_category_none_is_empty_string():
    assert q.resolve_manual_category(None) == ""


def test_classify_vectorized_matches_scalar_rule_elementwise():
    rng = np.random.default_rng(0)
    n = 500
    is_missing = rng.random(n) < 0.2
    anchor_valid = rng.random(n) < 0.7
    geometric_suspicious = rng.random(n) < 0.3
    statuses = [None, "correct", "wrong_label", "ghost", "ambiguous", "missing"]
    manual_status = [statuses[i] for i in rng.integers(0, len(statuses), n)]
    manual_category = np.array([q.resolve_manual_category(s) for s in manual_status], dtype=object)

    vec = q.classify_vectorized(is_missing, anchor_valid, geometric_suspicious, manual_category)
    for i in range(n):
        expected = q.classify_marker_frame(
            bool(is_missing[i]), manual_status[i], bool(anchor_valid[i]), bool(geometric_suspicious[i]))
        assert vec[i] == expected


def test_summarize_counts_shape_and_totals():
    classification = pd.DataFrame({
        "participant": ["P1"] * 4 + ["P1"] * 2,
        "trial_id": ["T1"] * 4 + ["T1"] * 2,
        "marker": ["A"] * 4 + ["B"] * 2,
        "category": ["observed_plausible", "observed_plausible", "missing", "anchor_invalid",
                     "observed_suspicious", "label_manually_wrong"],
    })
    out = q.summarize_counts(classification)
    assert len(out) == 2
    row_a = out[out.marker == "A"].iloc[0]
    assert row_a["n_frames"] == 4
    assert row_a["observed_plausible"] == 2
    assert row_a["missing"] == 1
    assert row_a["anchor_invalid"] == 1
    assert row_a["observed_suspicious"] == 0
    row_b = out[out.marker == "B"].iloc[0]
    assert row_b["n_frames"] == 2
    assert row_b["observed_suspicious"] == 1
    assert row_b["label_manually_wrong"] == 1
