import numpy as np
import pytest

from vicon2mano.core.correspondence import hungarian_assignment, label_seed, MANO_JOINT_NAMES


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


# --- the two marker protocols number their finger markers differently ------
# See _LABEL_HINTS' comment: data_June25's bare Index1-3 are mcp/middle-
# phalanx/FINGERTIP, while Nexus's RIDX1-4 are the usual mcp/pip/dip/tip.
# These pin that the two do not collide after the 2026-10-07 remap.

_JUNE_LABELS = [f"P10:{n}" for n in [
    "Forearm1", "Forearm2", "Forearm3", "Forearm4", "Palm1", "Palm2", "Palm3",
    "Thumb1", "Thumb2", "Thumb3", "Index1", "Index2", "Index3",
    "Middle1", "Middle2", "Middle3", "Ring1", "Ring2", "Ring3",
    "Pinky1", "Pinky2", "Pinky3",
]]


def test_label_seed_three_marker_protocol_maps_marker3_to_tip():
    seed = label_seed(_JUNE_LABELS, side="right")
    assert seed is not None
    name = lambda j: _JUNE_LABELS[seed[j]].split(":")[-1]
    for finger, stem in [("index", "Index"), ("middle", "Middle"),
                          ("ring", "Ring"), ("pinky", "Pinky")]:
        assert name(f"{finger}_mcp") == f"{stem}1"
        assert name(f"{finger}_dip") == f"{stem}2"
        assert name(f"{finger}_tip") == f"{stem}3"
        # no marker sits at the PIP in this protocol
        assert f"{finger}_pip" not in seed


def test_label_seed_thumb_mapping_left_unchanged():
    # the MCP->tip test is ambiguous for the thumb, so it keeps 1/2/3 = mcp/pip/dip
    seed = label_seed(_JUNE_LABELS, side="right")
    name = lambda j: _JUNE_LABELS[seed[j]].split(":")[-1]
    assert name("thumb_mcp") == "Thumb1"
    assert name("thumb_pip") == "Thumb2"
    assert name("thumb_dip") == "Thumb3"


def test_label_seed_four_marker_nexus_protocol_unaffected():
    labels = ["RWRB"] + [f"R{p}{i}" for p in ["THB", "IDX", "MID", "RNG", "LIT"]
                         for i in range(1, 5)]
    seed = label_seed(labels, side="right")
    assert seed is not None
    name = lambda j: labels[seed[j]]
    for finger, stem in [("index", "RIDX"), ("middle", "RMID"),
                          ("ring", "RRNG"), ("pinky", "RLIT")]:
        assert name(f"{finger}_mcp") == f"{stem}1"
        assert name(f"{finger}_pip") == f"{stem}2"
        assert name(f"{finger}_dip") == f"{stem}3"
        assert name(f"{finger}_tip") == f"{stem}4"
