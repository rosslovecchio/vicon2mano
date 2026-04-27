import numpy as np
import pytest

from vicon2mano.correspondence import hungarian_assignment, label_seed, MANO_JOINT_NAMES


def _make_joints(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((21, 3)).astype(np.float32)


def test_hungarian_assignment_perfect():
    """Identity assignment when markers == joints (shuffled)."""
    joints = _make_joints()
    perm = np.random.default_rng(42).permutation(21)
    markers = joints[perm]

    assign = hungarian_assignment(markers, joints)
    # Each joint should map to its permuted marker
    for j in range(21):
        assert assign[j] == np.where(perm == j)[0][0]


def test_hungarian_assignment_extra_markers():
    """22 markers (one extra) — all 21 joints still get assigned."""
    joints = _make_joints()
    extra = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    markers = np.concatenate([joints, extra], axis=0)

    assign = hungarian_assignment(markers, joints)
    assert (assign >= 0).sum() >= 18  # most joints matched


def test_label_seed_known_names():
    labels = [
        "RWRB", "RTHB1", "RTHB2", "RTHB3", "RTHB4",
        "RIDX1", "RIDX2", "RIDX3", "RIDX4",
        "RMID1", "RMID2", "RMID3", "RMID4",
        "RRNG1", "RRNG2", "RRNG3", "RRNG4",
        "RLIT1", "RLIT2", "RLIT3", "RLIT4",
        "RPALM",
    ]
    seed = label_seed(labels)
    assert seed is not None
    assert "wrist" in seed


def test_label_seed_insufficient_returns_none():
    seed = label_seed(["A", "B", "C"])
    assert seed is None
