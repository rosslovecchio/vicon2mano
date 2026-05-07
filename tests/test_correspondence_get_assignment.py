"""Tests for correspondence.get_assignment and _majority_vote.

Covers:
  - get_assignment with no labeler falls back to Hungarian
  - get_assignment uses labeler when provided
  - get_assignment falls back to Hungarian when labeler raises, emits warning
  - get_assignment uses label-seed when labeler is None and labels are given
  - get_assignment returns per-frame array when per_frame=True
  - _majority_vote correctness (mode, all-minus1, ties, index bounds)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vicon2mano.correspondence import (
    MANO_JOINT_NAMES,
    _majority_vote,
    get_assignment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_markers_and_joints(T: int = 5, N: int = 21, seed: int = 0):
    rng = np.random.default_rng(seed)
    markers = rng.uniform(-0.15, 0.15, (T, N, 3)).astype(np.float32)
    joints = rng.uniform(-0.15, 0.15, (T, 21, 3)).astype(np.float32)
    return markers, joints


class _FakeLabeler:
    """Minimal DeepLabeler stand-in for testing."""

    def __init__(self, fixed_assign: np.ndarray | None = None, raises: bool = False):
        self._fixed = fixed_assign
        self._raises = raises

    def label_sequence(self, markers_seq: np.ndarray) -> np.ndarray:
        if self._raises:
            raise RuntimeError("Simulated labeler failure")
        T = markers_seq.shape[0]
        if self._fixed is not None:
            return np.tile(self._fixed[None, :], (T, 1))
        return np.zeros((T, 21), dtype=int)


# ---------------------------------------------------------------------------
# get_assignment — no labeler
# ---------------------------------------------------------------------------

class TestGetAssignmentNoLabeler:
    def test_returns_1d_array(self):
        markers, joints = _make_markers_and_joints()
        assign = get_assignment(markers, joints)
        assert assign.ndim == 1
        assert assign.shape == (21,)

    def test_dtype_is_int(self):
        markers, joints = _make_markers_and_joints()
        assign = get_assignment(markers, joints)
        assert assign.dtype == int or np.issubdtype(assign.dtype, np.integer)

    def test_values_minus1_or_valid_index(self):
        markers, joints = _make_markers_and_joints()
        N = markers.shape[1]
        assign = get_assignment(markers, joints)
        assert np.all((assign == -1) | ((assign >= 0) & (assign < N)))

    def test_no_duplicate_marker_indices(self):
        markers, joints = _make_markers_and_joints()
        assign = get_assignment(markers, joints)
        valid = assign[assign >= 0]
        assert len(valid) == len(np.unique(valid))

    def test_label_seed_path_used_when_labels_given(self):
        """With recognisable Vicon labels and no labeler, the seed path fires."""
        labels = [
            "RWRB", "RTHB1", "RTHB2", "RTHB3", "RTHB4",
            "RIDX1", "RIDX2", "RIDX3", "RIDX4",
            "RMID1", "RMID2", "RMID3", "RMID4",
            "RRNG1", "RRNG2", "RRNG3", "RRNG4",
            "RLIT1", "RLIT2", "RLIT3", "RLIT4",
            "RPALM",
        ]
        T, N = 3, len(labels)
        rng = np.random.default_rng(0)
        markers = rng.uniform(-0.15, 0.15, (T, N, 3)).astype(np.float32)
        joints = rng.uniform(-0.15, 0.15, (T, 21, 3)).astype(np.float32)

        assign = get_assignment(markers, joints, marker_labels=labels)
        # wrist should have been seeded (RWRB is in the label hints)
        wrist_idx = MANO_JOINT_NAMES.index("wrist")
        assert assign[wrist_idx] >= 0


# ---------------------------------------------------------------------------
# get_assignment — with labeler
# ---------------------------------------------------------------------------

class TestGetAssignmentWithLabeler:
    def test_uses_labeler_result(self):
        markers, joints = _make_markers_and_joints(T=6, N=21)
        fixed = np.arange(21, dtype=int)  # identity mapping
        labeler = _FakeLabeler(fixed_assign=fixed)

        assign = get_assignment(markers, joints, labeler=labeler)
        # majority vote of a constant array must equal that constant
        assert np.array_equal(assign, fixed)

    def test_per_frame_returns_2d(self):
        T = 7
        markers, joints = _make_markers_and_joints(T=T, N=21)
        labeler = _FakeLabeler(fixed_assign=np.zeros(21, dtype=int))

        assign = get_assignment(markers, joints, labeler=labeler, per_frame=True)
        assert assign.shape == (T, 21)

    def test_fallback_on_exception_emits_warning(self):
        markers, joints = _make_markers_and_joints()
        labeler = _FakeLabeler(raises=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assign = get_assignment(markers, joints, labeler=labeler)

        assert len(caught) == 1
        assert "DeepLabeler failed" in str(caught[0].message)
        # Must still return a valid assignment (Hungarian fallback)
        assert assign.shape == (21,)

    def test_fallback_result_is_valid(self):
        markers, joints = _make_markers_and_joints()
        labeler = _FakeLabeler(raises=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assign = get_assignment(markers, joints, labeler=labeler)

        N = markers.shape[1]
        assert np.all((assign == -1) | ((assign >= 0) & (assign < N)))


# ---------------------------------------------------------------------------
# _majority_vote
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_constant_sequence(self):
        """All frames give the same assignment → that assignment is returned."""
        fixed = np.arange(21, dtype=int)
        assign_seq = np.tile(fixed, (10, 1))  # (10, 21)
        result = _majority_vote(assign_seq, n_markers=21)
        assert np.array_equal(result, fixed)

    def test_all_minus1(self):
        """When every frame has -1 for a joint, that joint stays -1."""
        assign_seq = np.full((5, 21), -1, dtype=int)
        result = _majority_vote(assign_seq, n_markers=21)
        assert np.all(result == -1)

    def test_mixed_with_some_minus1(self):
        """Majority vote ignores -1 entries; valid indices determine the winner."""
        assign_seq = np.full((10, 21), -1, dtype=int)
        # For joint 0: 7 votes for marker 3, 3 votes for marker 5
        assign_seq[:7, 0] = 3
        assign_seq[7:, 0] = 5
        result = _majority_vote(assign_seq, n_markers=21)
        assert result[0] == 3
        # Other joints remain -1
        assert np.all(result[1:] == -1)

    def test_tie_broken_by_argmax(self):
        """On a tie, numpy argmax picks the lowest index — deterministic."""
        assign_seq = np.zeros((4, 21), dtype=int)
        assign_seq[:2, 0] = 0   # 2 votes for marker 0
        assign_seq[2:, 0] = 1   # 2 votes for marker 1
        result = _majority_vote(assign_seq, n_markers=21)
        # bincount([0,0,1,1]).argmax() == 0
        assert result[0] == 0

    def test_out_of_bounds_indices_ignored(self):
        """Indices >= n_markers must be silently dropped."""
        n_markers = 10
        assign_seq = np.full((5, 21), -1, dtype=int)
        assign_seq[:, 0] = 99   # all frames give an out-of-bounds index
        result = _majority_vote(assign_seq, n_markers=n_markers)
        assert result[0] == -1

    def test_output_shape(self):
        assign_seq = np.zeros((8, 21), dtype=int)
        result = _majority_vote(assign_seq, n_markers=21)
        assert result.shape == (21,)

    def test_single_frame(self):
        """T=1 edge case: result should equal the single frame's assignment."""
        fixed = np.arange(21, dtype=int)
        result = _majority_vote(fixed[None, :], n_markers=21)
        assert np.array_equal(result, fixed)
