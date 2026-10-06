"""Tests for vicon2mano.core.continuity -- trajectory continuity, kept
deliberately separate from geometric self-consistency and anchor validity
(see that module's docstring)."""

from __future__ import annotations

import numpy as np
import pytest

from vicon2mano.core import continuity as cont


def test_frame_zero_never_flagged():
    markers = np.zeros((10, 1, 3))
    flags = cont.flag_discontinuities(markers)
    assert not flags[0, 0]


def test_smooth_motion_not_flagged():
    T = 50
    markers = np.zeros((T, 1, 3))
    markers[:, 0, 0] = np.arange(T) * 1.0  # 1mm/frame, smooth
    flags = cont.flag_discontinuities(markers)
    assert not flags.any()


def test_single_large_jump_flagged():
    T = 50
    markers = np.zeros((T, 1, 3))
    markers[:, 0, 0] = np.arange(T) * 1.0
    markers[30:, 0, 0] += 200.0  # 200mm teleport arriving at frame 30
    flags = cont.flag_discontinuities(markers)
    assert flags[30, 0]
    assert flags.sum() == 1


def test_zero_mad_data_does_not_flag_every_nonzero_step():
    # Mostly stationary with a few small moves and one real jump -- without
    # the min_jump_mm floor, MAD collapses to 0 and every nonzero step gets
    # flagged (same failure mode fixed elsewhere in this repo).
    T = 12
    markers = np.zeros((T, 1, 3))
    steps = np.array([0, 0, 0, 0, 0.5, 0, 0.3, 0, 80.0, 0, 0])
    markers[1:, 0, 0] = np.cumsum(steps)
    flags = cont.flag_discontinuities(markers, k=6.0, min_jump_mm=25.0)
    assert flags.sum() == 1
    assert flags[9, 0]  # arrival frame of the 80mm step (index 8 in `steps` -> frame 9)


def test_gap_contributes_no_displacement_and_is_not_flagged():
    T = 10
    markers = np.zeros((T, 1, 3))
    markers[:, 0, 0] = np.arange(T) * 1.0
    markers[5, 0, :] = np.nan  # a single occluded frame
    flags = cont.flag_discontinuities(markers)
    assert not flags.any()


def test_independent_per_marker():
    T = 20
    markers = np.zeros((T, 2, 3))
    markers[:, 0, 0] = np.arange(T) * 1.0
    markers[:, 1, 0] = np.arange(T) * 1.0
    markers[10:, 1, 0] += 300.0  # only marker 1 jumps
    flags = cont.flag_discontinuities(markers)
    assert not flags[:, 0].any()
    assert flags[10, 1]
