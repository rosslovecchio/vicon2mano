"""Tests for scripts/soma/flag_frames.py: label signals per fitted marker,
the robust fit-residual excess, and the 0/1/2 grading."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

SCRIPTS_SOMA = Path(__file__).resolve().parents[2] / "scripts" / "soma"
if str(SCRIPTS_SOMA) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SOMA))

import flag_frames as ff  # noqa: E402

ARGS = Namespace(fit_sigma=6.0, max_joint=150.0, max_step=5.0)


def test_label_signals_follow_the_column_the_vote_assigned():
    # 3 frames, 3 raw columns, labels 0 and 1 wanted
    voted = np.array([[0, 1, -1], [1, 0, -1], [0, -1, -1]])
    per_frame = np.array([[0, 1, -1], [1, 2, -1], [0, -1, -1]])      # frame 1: label 0's column disagrees
    conf = np.array([[0.9, 0.8, 0.0], [0.7, 0.3, 0.0], [0.95, 0.0, 0.0]])
    c, dis, present = ff.marker_label_signals(per_frame, voted, conf, [0, 1], np.arange(3))
    np.testing.assert_allclose(c, [[0.9, 0.8], [0.3, 0.7], [0.95, 0.0]])
    np.testing.assert_array_equal(dis, [[False, False], [True, False], [False, False]])
    np.testing.assert_array_equal(present, [[True, True], [True, True], [True, False]])


def test_robust_excess_ignores_the_outlier_in_its_own_scale():
    res = np.full((200, 2), 5.0) + np.random.default_rng(0).normal(0, 0.5, (200, 2))
    res[10, 0] = 40.0
    ex = ff.robust_excess(res)
    assert ex[10, 0] > 20 and np.abs(ex[:, 1]).max() < 6


def test_grade_levels():
    T, K = 4, 3
    zeros = np.zeros((T, K), bool)
    low = zeros.copy(); low[1, 0] = True                      # frame 1: label doubt only
    excess = np.zeros((T, K)); excess[2, 1] = 9.0             # frame 2: fit signal only
    max_joint = np.array([90.0, 90.0, 90.0, 160.0])           # frame 3: angle signal ...
    step = np.array([0.0, 0.0, 0.0, 9.0])
    flag, ls, fs, a_ = ff.grade(low, zeros, excess, max_joint, step,
                                np.full(T, 18), 14, ARGS)
    np.testing.assert_array_equal(flag, [0, 1, 1, 1])
    # two signals -> unreliable
    low2 = low.copy(); low2[2, 0] = True
    flag, *_ = ff.grade(low2, zeros, excess, max_joint, step, np.full(T, 18), 14, ARGS)
    assert flag[2] == 2


def test_too_few_markers_is_unreliable_even_with_no_signal():
    z = np.zeros((2, 3), bool)
    flag, *_ = ff.grade(z, z, np.zeros((2, 3)), np.zeros(2), np.zeros(2),
                        np.array([18, 10]), 14, ARGS)
    np.testing.assert_array_equal(flag, [0, 2])
