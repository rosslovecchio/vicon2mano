"""Tests for vicon2mano.core.camera_calib -- RealSense camera -> Vicon extrinsics (synthetic, known pose)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from vicon2mano.core import camera_calib as cc

INTR = {"fx": 433.5, "fy": 433.0, "ppx": 426.5, "ppy": 240.0, "width": 848, "height": 480,
        "model": "inverse_brown_conrady", "coeffs": [-0.051, 0.063, -0.0004, 0.001, -0.02]}
TRUE = np.r_[Rot.from_euler("xyz", [10, -20, 5], degrees=True).as_rotvec(), 0.05, -0.02, 0.5]  # vicon -> camera
RNG = np.random.default_rng(0)
HAND = RNG.uniform(-0.08, 0.08, (21, 3))                     # 21 "joints" around the Vicon origin, metres


def _cam(X):
    return X @ Rot.from_rotvec(TRUE[:3]).as_matrix().T + TRUE[3:]


def test_project_inverts_deproject_with_distortion():
    uv = cc.project(TRUE, HAND, INTR)
    back = cc.deproject(uv, _cam(HAND)[:, 2], INTR)
    assert np.abs(back - _cam(HAND)).max() < 1e-6


def test_kabsch_recovers_rigid_transform():
    R, t = cc.kabsch(HAND, _cam(HAND))
    assert np.allclose(HAND @ R.T + t, _cam(HAND), atol=1e-9)


def test_pose_from_landmarks_recovers_the_frame_pose():
    lm = cc.project(TRUE, HAND, INTR)
    depth = np.zeros((480, 848))
    for (x, y), z in zip(np.round(lm).astype(int), _cam(HAND)[:, 2]):
        depth[y - 2:y + 3, x - 2:x + 3] = z
    p, px = cc.pose_from_landmarks(lm, depth, HAND, INTR)
    assert px < 0.5 and np.abs(cc.project(p, HAND, INTR) - lm).max() < 0.5


def test_refine_on_dots_ignores_clutter_and_recovers_pose():
    markers = HAND[:12]
    start = TRUE + np.r_[np.radians([1.5, -1.0, 1.0]), 0.01, -0.01, 0.01]
    frames = []
    for k in range(20):                                       # the hand moves; the camera does not
        V = markers + RNG.uniform(-0.1, 0.1, 3)
        dots = np.r_[cc.project(TRUE, V, INTR), RNG.uniform([0, 0], [848, 480], (60, 2))]
        frames.append((V, dots))
    p, res = cc.refine_on_dots(start, frames, INTR, gates=(40, 20, 10, 6, 4, 3))
    assert np.median(res) < 0.5
    assert np.abs(cc.project(p, markers, INTR) - cc.project(TRUE, markers, INTR)).max() < 1.0


def test_is_moving_separates_shifted_views_from_a_still_one():
    scene = RNG.uniform(0, 255, (120, 212))
    still = np.array([scene + RNG.normal(0, 2, scene.shape) for _ in range(30)])
    moved = np.array([np.roll(scene, (k % 7) * 3, axis=1) for k in range(30)])
    assert not cc.is_moving(still) and cc.is_moving(moved)
