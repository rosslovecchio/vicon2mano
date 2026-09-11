"""relabel_frame's generalized candidate/slot masks (label-agnostic support)."""

import numpy as np

from vicon2mano.mano_relabel import relabel_frame


def test_flagged_mode_unchanged_when_slot_mask_omitted():
    # Two markers swapped; only these two are flagged, mirroring the
    # existing cascade-driven call sites.
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    pred = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    flagged = np.array([True, True, False])

    mapping, dists = relabel_frame(frame, pred, flagged, max_dist_m=0.03)

    assert mapping == {0: 1, 1: 0}
    assert set(dists) == {0, 1}
    assert all(d < 1e-6 for d in dists.values())


def test_label_agnostic_mode_can_move_an_unflagged_marker():
    # Same swap, but now every marker is eligible on both sides (the
    # label-agnostic case) -- the third, never-flagged marker's position is
    # unaffected because nothing predicts it should move.
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    pred = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    present = np.array([True, True, True])

    mapping, dists = relabel_frame(frame, pred, present, max_dist_m=0.03)

    assert mapping == {0: 1, 1: 0}
    assert 2 not in mapping


def test_slot_mask_independent_of_candidate_mask():
    # Only marker 0's *label* is eligible to receive a new position, but any
    # present marker may supply it.
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    pred = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    candidate_mask = np.array([True, True])
    slot_mask = np.array([True, False])

    mapping, _ = relabel_frame(frame, pred, candidate_mask, slot_mask, max_dist_m=0.03)

    assert mapping == {0: 1}
    assert 1 not in mapping


def test_max_dist_gate_still_applies_in_label_agnostic_mode():
    # Best swap still leaves a 0.03m residual on each end -- gated out at
    # 0.02m, accepted at 0.05m.
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    pred = np.array([[0.13, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    present = np.array([True, True])

    mapping, _ = relabel_frame(frame, pred, present, max_dist_m=0.02)
    assert mapping == {}

    mapping_ok, dists = relabel_frame(frame, pred, present, max_dist_m=0.05)
    assert mapping_ok == {0: 1, 1: 0}
    assert all(abs(d - 0.03) < 1e-9 for d in dists.values())


def test_missing_marker_never_invented_or_used_as_a_source():
    # Marker 1 is occluded (NaN) this frame. Even though the model predicts
    # a plausible position for slot 1, and slot 0 could use *some* position,
    # a missing observation must never be manufactured, and a missing
    # marker's absence must never let some other marker be pulled into its
    # slot from thin air either.
    frame = np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    present = np.array([True, False])   # marker 1 is not actually present

    mapping, _ = relabel_frame(frame, pred, present, max_dist_m=0.03)

    assert 1 not in mapping             # slot 1 is never filled
    assert 1 not in mapping.values()    # marker 1's (nonexistent) data never used as a source


def test_final_mapping_cannot_duplicate_an_observation_into_two_slots():
    # Three present markers, one predicted slot with no close match of its
    # own (slot 2's prediction sits far from everything). The assignment
    # must stay one-to-one: marker 0's position cannot simultaneously fill
    # slot 0 and be duplicated into slot 2.
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    pred = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    present = np.array([True, True, True])

    mapping, _ = relabel_frame(frame, pred, present, max_dist_m=0.03)

    sources = list(mapping.values())
    assert len(sources) == len(set(sources)), "same observation used as source twice"
    # every present marker still ends up represented in exactly one slot:
    # either it kept its own label (absent from mapping) or supplied exactly
    # one other slot's data.
    slots_filled_by = {**{s: s for s in range(3) if s not in mapping}, **mapping}
    assert sorted(slots_filled_by.values()) == [0, 1, 2]
