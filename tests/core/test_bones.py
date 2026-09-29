"""Tests for vicon2mano.core.bones -- moved unchanged from
scripts/gmm/relabel_trial.py (see that module's docstring). These tests
cover the module in its new home; strategyGMM's own existing behaviour is
covered by tests/gmm/test_labeler.py and is untouched by the move.
"""

from __future__ import annotations

import numpy as np
import pytest

from vicon2mano.core import bones as bn


def _rigid_triangle(T, rng, noise=0.0):
    """3 points at fixed mutual distances (30, 40, 50 -- a 3-4-5 triangle),
    rigidly translated/rotated per frame."""
    pts0 = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [30.0, 40.0, 0.0]])
    out = np.zeros((T, 3, 3))
    for t in range(T):
        th = 0.05 * t
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([t * 0.3, 0.0, 0.0])
        out[t] = pts0 @ R.T + shift
        if noise:
            out[t] += rng.normal(scale=noise, size=out[t].shape)
    return out


def test_bone_error_zero_for_perfectly_consistent_triangle():
    rng = np.random.default_rng(0)
    markers = _rigid_triangle(50, rng)
    bones = [(0, 1), (1, 2), (0, 2)]
    ref, inliers = bn.consensus_bone_lengths(markers, bones, tol_mm=1.0, min_bones=3)
    np.testing.assert_allclose(ref, [30.0, 40.0, 50.0], atol=1e-6)
    assert len(inliers) == 50
    err = bn.bone_error(markers, bones, ref)
    np.testing.assert_allclose(err, 0.0, atol=1e-6)


def test_consensus_recovers_reference_despite_majority_mislabelled():
    # Mirrors the real Ring1 case from CLAUDE.md: a bone that reads wrong in
    # most frames because of a persistent swap should NOT win the modal
    # reference, since wrong configurations don't agree with each other.
    rng = np.random.default_rng(1)
    T = 100
    good = _rigid_triangle(T, rng)   # bone (0,1) == 30mm always
    bad = good.copy()
    # In 70/100 frames, corrupt point 1 to a different, WRONG fixed distance
    # from point 0 (simulating a consistent-looking but incorrect swap) --
    # each "wrong" frame's own distance still varies frame to frame (real
    # noise), so wrong frames do not agree with EACH OTHER either.
    wrong_frames = rng.choice(T, size=70, replace=False)
    for t in wrong_frames:
        direction = bad[t, 1] - bad[t, 0]
        direction /= np.linalg.norm(direction)
        bad[t, 1] = bad[t, 0] + direction * (30.0 + rng.normal(scale=5.0))

    bones = [(0, 1), (1, 2), (0, 2)]
    ref, inliers = bn.consensus_bone_lengths(bad, bones, tol_mm=1.0, min_bones=3)
    assert ref[0] == pytest.approx(30.0, abs=0.5)
    # The 30 correctly-labelled frames should dominate the inlier set.
    assert len(inliers) >= 25


def test_bone_error_missing_markers_produces_nan_not_crash():
    markers = np.full((10, 2, 3), np.nan)
    bones = [(0, 1)]
    ref, inliers = bn.consensus_bone_lengths(markers, bones, min_bones=1)
    assert np.isnan(ref).all()
    assert inliers.size == 0
    err = bn.bone_error(markers, bones, ref)
    assert np.isnan(err).all()


def test_consensus_bone_lengths_empty_bone_list():
    markers = np.zeros((5, 2, 3))
    ref, inliers = bn.consensus_bone_lengths(markers, [])
    assert ref.shape == (0,)


def test_bone_error_constant_length_gives_zero_error():
    # Sudden marker swap: bone length constant except for a single-frame
    # discontinuity -- this is the shape geometric_self_consistency.py
    # needs to detect as a "suspicious event".
    rng = np.random.default_rng(2)
    markers = _rigid_triangle(60, rng)
    swapped = markers.copy()
    swapped[30, [0, 1]] = swapped[30, [1, 0]]  # swap points 0 and 1 at frame 30
    bones = [(0, 1), (1, 2), (0, 2)]
    ref, _ = bn.consensus_bone_lengths(markers, bones, min_bones=3)
    err = bn.bone_error(swapped, bones, ref)
    # Bone (0,1) length is symmetric under swapping its own two endpoints,
    # so it stays ~0 error; the OTHER two bones (which touch only one of
    # the swapped points) should spike at exactly frame 30.
    assert err[30, 1] > 5.0 or err[30, 2] > 5.0
    normal_frames = [t for t in range(60) if t != 30]
    assert np.nanmax(err[normal_frames][:, 1:]) < 1.0


def test_modal_bone_lengths_matches_consensus_on_clean_data():
    rng = np.random.default_rng(3)
    markers = _rigid_triangle(80, rng, noise=0.01)
    bones = [(0, 1), (1, 2), (0, 2)]
    modal = bn.modal_bone_lengths(markers, bones)
    consensus, _ = bn.consensus_bone_lengths(markers, bones, min_bones=3)
    np.testing.assert_allclose(modal, consensus, atol=0.5)


def test_consensus_bone_lengths_survives_a_permanently_missing_marker():
    # Regression test for a real bug found running geometric_self_consistency
    # across the whole dataset: P1 has 3 markers (Forearm3/4, Palm3) that are
    # 0% available for the ENTIRE trial. That makes every bone touching them
    # NaN in every frame, so the old "hypothesis must have literally every
    # bone finite" requirement never found a single qualifying frame -- and
    # the function fell back to all-NaN for EVERY bone, including the ones
    # among the 19 other markers that were 87-100% available and perfectly
    # fine. A permanently-missing marker must not poison every other bone's
    # reference length.
    rng = np.random.default_rng(5)
    T = 100
    # 4 points: a good rigid triangle (0,1,2) plus point 3, which never has
    # a valid coordinate at all (simulating the permanently-missing marker).
    triangle = _rigid_triangle(T, rng, noise=0.02)
    markers = np.full((T, 4, 3), np.nan)
    markers[:, :3] = triangle
    bones = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]  # last two touch the missing marker
    ref, inliers = bn.consensus_bone_lengths(markers, bones, tol_mm=1.0, min_bones=3)
    # The 3 good bones must still get a real reference length...
    assert np.isfinite(ref[:3]).all()
    np.testing.assert_allclose(ref[:3], [30.0, 40.0, 50.0], atol=0.5)
    # ...and the 2 bones touching the always-missing marker are correctly NaN.
    assert np.isnan(ref[3:]).all()
    assert len(inliers) > 50

    err = bn.bone_error(markers, bones, ref)
    assert np.isfinite(err[:, :3]).all()
    assert np.isnan(err[:, 3:]).all()


def test_consensus_bone_lengths_reproducible_with_fixed_seed():
    rng = np.random.default_rng(4)
    markers = _rigid_triangle(200, rng, noise=0.05)
    bones = [(0, 1), (1, 2), (0, 2)]
    ref1, inliers1 = bn.consensus_bone_lengths(markers, bones, seed=42, min_bones=3)
    ref2, inliers2 = bn.consensus_bone_lengths(markers, bones, seed=42, min_bones=3)
    np.testing.assert_array_equal(ref1, ref2)
    np.testing.assert_array_equal(inliers1, inliers2)
