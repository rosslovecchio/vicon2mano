"""scripts/soma/static_pose_check.py: the anatomical-limit penalty used to constrain a static fit."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "soma"))
import static_pose_check as spc  # noqa: E402


def _setup(w=100.0):
    B = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))          # pose axis == basis axis
    lo, hi = np.zeros((15, 3), np.float32), np.zeros((15, 3), np.float32)
    hi[:, 0], lo[:, 0] = 1.0, -0.5                                   # flexion range; abduction and twist fixed at 0
    return spc.limit_penalty(B, lo, hi, w, "cpu")


def test_inside_the_limits_costs_nothing():
    pen = _setup()
    pose = torch.zeros(45)
    pose[0], pose[3] = 0.5, -0.2                                     # flexion only, in range
    assert pen(pose).item() == 0.0


def test_twist_and_overshoot_are_penalised_quadratically():
    pen = _setup(w=10.0)
    p = torch.zeros(45)
    p[2] = 0.1                                                       # index1 twist 0.1 rad (limit 0)
    assert abs(pen(p).item() - 10.0 * 0.01) < 1e-6
    q = torch.zeros(45)
    q[0] = 1.3                                                       # index1 flexion 0.3 over the top
    assert abs(pen(q).item() - 10.0 * 0.09) < 1e-6


def test_pinky1_and_thumb1_weigh_more():
    pen = _setup(w=10.0)
    ordinary, boosted = torch.zeros(45), torch.zeros(45)
    ordinary[3 * 3 + 2] = 0.1                                        # middle1 twist
    boosted[6 * 3 + 2] = 0.1                                         # pinky1 twist
    assert abs(pen(boosted).item() / pen(ordinary).item() - 5.0) < 1e-5


def test_penalty_has_a_gradient_toward_the_limits():
    pen = _setup()
    p = torch.zeros(45, requires_grad=True)
    pen(p + torch.eye(45)[2] * 0.2).backward()
    assert p.grad[2].item() > 0
