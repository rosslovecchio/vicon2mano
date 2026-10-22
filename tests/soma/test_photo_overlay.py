"""photo_overlay: automatic dot labelling and the rigid-marker camera fit, on a synthetic hand seen from above."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS_SOMA = Path(__file__).resolve().parents[2] / "scripts" / "soma"
if str(SCRIPTS_SOMA) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SOMA))

import photo_overlay as po  # noqa: E402

# a flat-ish right hand, mm: x along the fingers, y across, z up from the table (dorsal side up)
HAND3 = {"Palm1": (60, 22, 3), "Palm2": (70, 4, 8), "Palm3": (45, -8, 9), "Thumb1": (62, -38, 5),
         "Index1": (10, -18, 7), "Middle1": (8, -4, 8), "Ring1": (14, 8, 7), "Pinky1": (22, 22, 5)}
F = 1635.0
SIZE = (1330, 2364)


def _pixels(pose_rotvec, cam_t):
    project = po.make_camera(F, SIZE)
    X = np.array(list(HAND3.values()), float)
    c = X.mean(0)
    return project(np.r_[pose_rotvec, cam_t], X - c), X - c, project


def test_auto_label_finds_the_right_dots_among_false_blobs():
    px, _, _ = _pixels([np.pi, 0, 0.4], [0, 0, 300.0])            # camera above the hand, looking down, rotated 0.4 rad
    rng = np.random.default_rng(0)
    false = rng.uniform([100, 100], [1200, 2200], (14, 2))          # tape, shadows, forearm balls
    dots = np.vstack([false[:7], px, false[7:]])
    order = rng.permutation(len(dots))                              # the dots have no order
    P3 = {k: np.array(v, float) for k, v in HAND3.items()}
    lab = po.auto_label(P3, dots[order], F / 300.0)
    got = {k: dots[order][j] for k, j in lab.items()}
    assert set(got) == set(HAND3)
    for k, true in zip(HAND3, px):
        assert np.linalg.norm(got[k] - true) < 1e-6, k


def test_camera_fit_recovers_a_known_pose():
    px, X, project = _pixels([np.pi, 0, 0.4], [0, 0, 300.0])
    p, err = po.solve_pose(project, X, px)
    assert err.max() < 0.5 and abs(p[5] - 300.0) < 3.0


def test_focal_from_exif_defaults_to_24mm_equivalent():
    im = Image.new("RGB", (1330, 2364))                              # no EXIF
    assert abs(po.focal_px(im) - 24 / 43.27 * 1.25 * 2364) < 1e-6
