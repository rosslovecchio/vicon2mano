"""Tests for vicon2mano.gmm_labeler (strategyGMM).

Built up step by step alongside the module: this file currently covers
step 1 (rigid local frame) and step 2 (per-marker GMM fitting). Hypothesis
generation (top-N assignment) and temporal selection (Kalman + Viterbi)
get their own test sections as those pieces land.
"""

from __future__ import annotations

import itertools

import numpy as np

from vicon2mano import gmm_labeler as gl


# ---------------------------------------------------------------------------
# Step 1: rigid local frame
# ---------------------------------------------------------------------------


def test_rigid_frame_roundtrip():
    rng = np.random.default_rng(0)
    T, N = 5, 4
    markers = rng.normal(size=(T, N, 3)) * 50 + 100
    R, o = gl.rigid_frames(markers, origin_idx=0, x_idx=1, y_idx=2)
    local = gl.to_local(markers, R, o)
    world = gl.to_world(local, R, o)
    np.testing.assert_allclose(world, markers, atol=1e-8)


def test_rigid_frame_axes_orthonormal():
    rng = np.random.default_rng(1)
    markers = rng.normal(size=(3, 3, 3)) * 10
    R, _ = gl.rigid_frames(markers, 0, 1, 2)
    for t in range(R.shape[0]):
        np.testing.assert_allclose(R[t] @ R[t].T, np.eye(3), atol=1e-8)


def _plate_recording(rng, T=200, bad=(50, 60)):
    """3 rigid plate markers + 1 distractor, with the plate's labels
    swapped (0<->1) for frames in ``bad``. Whole body translates/rotates."""
    plate = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    distractor = np.array([80.0, 80.0, 10.0])
    out = np.zeros((T, 4, 3))
    for t in range(T):
        th = 0.02 * t
        c, s = np.cos(th), np.sin(th)
        Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([t * 1.5, 0.0, 0.0])
        pts = np.vstack([plate, distractor]) @ Rw.T + shift
        pts += rng.normal(scale=0.05, size=pts.shape)
        out[t] = pts
    for t in range(*bad):
        out[t, 0], out[t, 1] = out[t, 1].copy(), out[t, 0].copy()
    return out


def test_learn_anchor_triangle_recovers_the_plate_geometry():
    rng = np.random.default_rng(20)
    markers = _plate_recording(rng)
    ref = gl.learn_anchor_triangle(markers, (0, 1, 2))
    # d01 = 30, d02 = 40, d12 = 50 (3-4-5 triangle)
    np.testing.assert_allclose(ref, [30.0, 40.0, 50.0], atol=0.6)


def test_learn_anchor_triangle_uses_the_mode_not_the_median():
    # 60% of frames have a corrupted (stretched) plate, scattered over a
    # wide range. The median would land inside the corrupted bulk; only the
    # mode finds the one *repeated* value, which is the real geometry.
    rng = np.random.default_rng(21)
    T = 500
    markers = np.zeros((T, 3, 3))
    markers[:, 1, 0] = 30.0
    markers[:, 2, 1] = 40.0
    corrupt = rng.random(T) < 0.6
    markers[corrupt, 1, 0] = rng.uniform(60, 140, corrupt.sum())
    ref = gl.learn_anchor_triangle(markers, (0, 1, 2))
    assert abs(ref[0] - 30.0) < 1.0
    assert abs(np.median(np.linalg.norm(markers[:, 1] - markers[:, 0], axis=1)) - 30.0) > 20.0


def test_locate_anchor_triangle_finds_and_repairs_swapped_anchors():
    rng = np.random.default_rng(22)
    markers = _plate_recording(rng, bad=(50, 60))
    ref = gl.learn_anchor_triangle(markers, (0, 1, 2))
    tri = gl.locate_anchor_triangle(markers, ref, (0, 1, 2), tol_mm=2.0)

    assert (tri[:, 0] >= 0).all()                       # located every frame
    good = np.r_[0:50, 60:200]
    assert (tri[good] == np.array([0, 1, 2])).all()     # labels kept when right
    # In the swapped block the labelled triple no longer matches the
    # reference triangle's *ordered* distances, so a different (repaired)
    # triple is chosen.
    assert not (tri[50:60] == np.array([0, 1, 2])).all()


def test_locate_anchor_triangle_reports_minus_one_when_nothing_matches():
    rng = np.random.default_rng(23)
    markers = rng.normal(scale=200.0, size=(5, 6, 3))   # no rigid structure
    tri = gl.locate_anchor_triangle(markers, np.array([30.0, 40.0, 50.0]),
                                     (0, 1, 2), tol_mm=0.05)
    assert (tri == -1).all()


def test_rigid_frames_from_triangle_matches_fixed_anchor_version():
    rng = np.random.default_rng(24)
    markers = _plate_recording(rng, bad=(0, 0))         # no corruption
    tri = np.tile(np.array([0, 1, 2]), (markers.shape[0], 1))
    R_tri, o_tri = gl.rigid_frames_from_triangle(markers, tri)
    R_fix, o_fix = gl.rigid_frames(markers, 0, 1, 2)
    np.testing.assert_allclose(R_tri, R_fix, atol=1e-9)
    np.testing.assert_allclose(o_tri, o_fix, atol=1e-9)


def test_rigid_frames_from_triangle_is_nan_where_no_triangle_located():
    markers = np.zeros((3, 4, 3))
    tri = np.array([[0, 1, 2], [-1, -1, -1], [0, 1, 2]])
    R, o = gl.rigid_frames_from_triangle(markers, tri)
    assert np.isnan(R[1]).all() and np.isnan(o[1]).all()
    assert not np.isnan(o[0]).any()


def test_mask_untrustworthy_frames_nans_out_bad_anchor_frames_only():
    local = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    anchor_valid = np.array([True, False])
    out = gl.mask_untrustworthy_frames(local, anchor_valid)
    np.testing.assert_array_equal(out[0], local[0])
    assert np.isnan(out[1]).all()
    # input untouched
    assert not np.isnan(local).any()


def test_local_frame_is_invariant_to_rigid_motion():
    # A marker with a fixed offset from the anchor triangle must land at the
    # same local coordinates regardless of how the whole rigid body is
    # translated/rotated in world space -- this is the entire point of
    # working in the local frame.
    rng = np.random.default_rng(2)
    anchors_frame0 = rng.normal(size=(3, 3)) * 20
    offset = np.array([5.0, -3.0, 2.0])

    def rotate(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    frames = []
    for theta, shift in [(0.0, np.zeros(3)), (0.7, np.array([30.0, -10.0, 5.0]))]:
        Rw = rotate(theta)
        anchors = anchors_frame0 @ Rw.T + shift
        marker = anchors[0] + offset @ Rw.T   # rigidly attached to the anchor triangle
        frames.append(np.vstack([anchors, marker]))
    markers = np.stack(frames)   # (2, 4, 3): anchors 0-2, marker 3

    R, o = gl.rigid_frames(markers, origin_idx=0, x_idx=1, y_idx=2)
    local = gl.to_local(markers, R, o)
    np.testing.assert_allclose(local[0, 3], local[1, 3], atol=1e-6)


# ---------------------------------------------------------------------------
# Step 2: per-marker GMM fitting
# ---------------------------------------------------------------------------


def test_fit_gmm_diag_recovers_single_cluster_mean():
    rng = np.random.default_rng(3)
    true_mean = np.array([10.0, -5.0, 2.0])
    samples = rng.normal(loc=true_mean, scale=0.5, size=(500, 3))
    gmm = gl.fit_gmm_diag(samples, n_components=1, n_iter=30)
    np.testing.assert_allclose(gmm.means[0], true_mean, atol=0.2)


def test_fit_gmm_diag_recovers_two_well_separated_clusters():
    rng = np.random.default_rng(4)
    mean_a, mean_b = np.array([0.0, 0.0, 0.0]), np.array([50.0, 50.0, 50.0])
    a = rng.normal(loc=mean_a, scale=1.0, size=(300, 3))
    b = rng.normal(loc=mean_b, scale=1.0, size=(300, 3))
    samples = np.vstack([a, b])
    gmm = gl.fit_gmm_diag(samples, n_components=2, n_iter=50)

    recovered = sorted(gmm.means.tolist(), key=lambda m: m[0])
    expected = sorted([mean_a.tolist(), mean_b.tolist()], key=lambda m: m[0])
    np.testing.assert_allclose(recovered, expected, atol=1.0)


def test_gmm_loglik_is_higher_near_the_mean():
    gmm = gl.GMMParams(
        weights=np.array([1.0]),
        means=np.array([[0.0, 0.0, 0.0]]),
        variances=np.array([[1.0, 1.0, 1.0]]),
    )
    near = gl.gmm_loglik(np.array([0.1, 0.0, 0.0]), gmm)
    far = gl.gmm_loglik(np.array([10.0, 0.0, 0.0]), gmm)
    assert near > far


def test_fit_marker_gmms_skips_markers_with_too_few_samples():
    T, N = 30, 2
    local = np.full((T, N, 3), np.nan)
    rng = np.random.default_rng(5)
    local[:, 0] = rng.normal(size=(T, 3))     # plenty of samples
    local[:5, 1] = rng.normal(size=(5, 3))    # below min_samples default (20)

    gmms = gl.fit_marker_gmms(local, [0, 1], np.arange(T), min_samples=20)
    assert 0 in gmms
    assert 1 not in gmms


# ---------------------------------------------------------------------------
# Step 3: top-N assignment (Murty's algorithm)
# ---------------------------------------------------------------------------


def _brute_force_top_n(cost: np.ndarray, n: int):
    """Reference implementation: enumerate every M-subset-of-K injection."""
    M, K = cost.shape
    best = []
    for cols in itertools.permutations(range(K), M):
        total = sum(cost[i, c] for i, c in enumerate(cols))
        best.append((total, np.array(cols)))
    best.sort(key=lambda x: x[0])
    return best[:n]


def test_top_n_assignments_matches_hungarian_for_n1():
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(6)
    cost = rng.normal(size=(4, 4))
    (assign, total), = gl.top_n_assignments(cost, 1)
    r, c = linear_sum_assignment(cost)
    expected = np.full(4, -1, dtype=int)
    expected[r] = c
    np.testing.assert_array_equal(assign, expected)
    np.testing.assert_allclose(total, cost[r, c].sum())


def test_top_n_assignments_matches_brute_force_ranking():
    rng = np.random.default_rng(7)
    cost = rng.normal(size=(4, 4))
    got = gl.top_n_assignments(cost, 6)
    expected = _brute_force_top_n(cost, 6)

    got_costs = [round(c, 6) for _, c in got]
    exp_costs = [round(c, 6) for c, _ in expected]
    assert got_costs == exp_costs


def test_top_n_assignments_handles_rectangular_more_observations_than_markers():
    rng = np.random.default_rng(8)
    cost = rng.normal(size=(3, 5))   # 3 markers, 5 observations
    got = gl.top_n_assignments(cost, 4)
    assert all(a.shape == (3,) for a, _ in got)
    # scores strictly non-decreasing (best-first ordering)
    costs = [c for _, c in got]
    assert costs == sorted(costs)


def test_top_n_assignments_is_non_decreasing_in_cost():
    rng = np.random.default_rng(9)
    cost = rng.normal(size=(6, 6))
    got = gl.top_n_assignments(cost, 15)
    costs = [c for _, c in got]
    assert costs == sorted(costs)
    # all distinct assignments
    keys = [tuple(a.tolist()) for a, _ in got]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Step 4: Kalman filter + Viterbi hypothesis selection
# ---------------------------------------------------------------------------


def test_marker_kalman_converges_to_a_stationary_marker():
    kf = gl.MarkerKalman(np.array([0.0, 0.0, 0.0]), process_var=0.01, obs_var=1.0)
    true_pos = np.array([5.0, -2.0, 1.0])
    for _ in range(50):
        kf.predict()
        kf.update(true_pos)
    np.testing.assert_allclose(kf.x[:3], true_pos, atol=0.5)


def test_marker_kalman_residual_loglik_prefers_closer_observation():
    kf = gl.MarkerKalman(np.array([0.0, 0.0, 0.0]), process_var=0.01, obs_var=1.0)
    kf.predict()
    ll_near, _ = kf.residual_loglik(np.array([0.1, 0.0, 0.0]))
    ll_far, _ = kf.residual_loglik(np.array([10.0, 0.0, 0.0]))
    assert ll_near > ll_far


def test_viterbi_select_resolves_a_swap_that_spatial_hypotheses_alone_cannot():
    # Two markers, A and B, sit close enough that at t=1 both possible
    # pairings look spatially equal (a synthetic stand-in for the
    # Middle1/Ring1-style ambiguity the cascade's distance checks miss).
    # Temporal continuity from t=0 must be what breaks the tie: A and B
    # barely move, so the identity-preserving hypothesis should win even
    # though top_n_assignments hands both hypotheses to the selector.
    marker_order = [0, 1]
    obs_t0 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    obs_t1 = np.array([[0.2, 0.0, 0.0], [10.2, 0.0, 0.0]])

    hyps_t0 = [(np.array([0, 1]), 0.0)]  # identity at t0 (seed)
    # t1: hypothesis 0 keeps identity; hypothesis 1 swaps A<->B. Spatially
    # near-identical cost since the two observations barely moved from a
    # symmetric-looking configuration once GMMs are ignored here.
    hyps_t1 = [(np.array([0, 1]), 0.0), (np.array([1, 0]), 0.0)]

    result = gl.viterbi_select(
        marker_order,
        [hyps_t0, hyps_t1],
        [obs_t0, obs_t1],
        init_positions={0: obs_t0[0], 1: obs_t0[1]},
        process_var=0.001,
        obs_var=1.0,
    )
    np.testing.assert_array_equal(result[1], np.array([0, 1]))


# ---------------------------------------------------------------------------
# Step 5: relabel_sequence / GMMLabeler end-to-end on synthetic data
# ---------------------------------------------------------------------------


def _synthetic_two_finger_recording(rng, T=40, swap_start=15, swap_len=6):
    """3 anchors (rigid triangle, moving) + 2 "finger" markers (Index1,
    Middle1) each orbiting a fixed local offset -- close enough together
    that a naive nearest-neighbour match at the swap frames is ambiguous,
    but each keeps its own small-radius orbit, which is what the GMM
    should have learned from the (swap-free) reference frames.
    """
    labels = ["Wrist", "IndexMCPref", "MiddleMCPref", "Index1", "Middle1"]
    off_index = np.array([20.0, 5.0, 0.0])
    off_middle = np.array([20.0, -5.0, 0.0])   # 10mm apart -- deliberately close

    markers = np.zeros((T, 5, 3))
    for t in range(T):
        # slowly translating/rotating rigid triangle
        theta = 0.05 * t
        c, s = np.cos(theta), np.sin(theta)
        Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([t * 2.0, 0.0, 0.0])
        anchors = (np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], dtype=float) @ Rw.T) + shift
        jitter_i = rng.normal(scale=1.0, size=3)
        jitter_m = rng.normal(scale=1.0, size=3)
        idx_world = anchors[0] + (off_index + jitter_i) @ Rw.T
        mid_world = anchors[0] + (off_middle + jitter_m) @ Rw.T
        markers[t, :3] = anchors
        markers[t, 3] = idx_world
        markers[t, 4] = mid_world

    # Ground truth identity is Index1/Middle1 throughout; inject an actual
    # column swap during [swap_start, swap_start+swap_len) to simulate a
    # mislabelling episode the cascade's distance checks would miss (both
    # markers stay plausible relative to the rigid anchors).
    swapped = markers.copy()
    for t in range(swap_start, swap_start + swap_len):
        swapped[t, 3], swapped[t, 4] = markers[t, 4].copy(), markers[t, 3].copy()

    ref_frames = np.array([t for t in range(T) if not (swap_start <= t < swap_start + swap_len)])
    return labels, swapped, markers, ref_frames


def test_relabel_sequence_recovers_a_synthetic_finger_swap():
    rng = np.random.default_rng(11)
    labels, swapped, truth, ref_frames = _synthetic_two_finger_recording(rng)
    marker_idxs = [3, 4]   # Index1, Middle1

    result = gl.relabel_sequence(
        swapped, labels, ("Wrist", "IndexMCPref", "MiddleMCPref"),
        ref_frames, marker_idxs,
        n_hypotheses=2, n_components=1, process_var=5.0, obs_var=4.0,
    )
    # result[t, i] indexes into marker_idxs: 0 keeps Index1, 1 means "the
    # slot's occupant is actually the other column's observation".
    corrected_is_swap = result[:, 0] == 1
    true_is_swap = np.zeros(swapped.shape[0], dtype=bool)
    true_is_swap[15:21] = True
    # Allow the boundary frames to disagree (Viterbi may lag by a frame or
    # two before committing to a transition); the bulk of the swapped
    # window must be recovered.
    core = slice(16, 20)
    np.testing.assert_array_equal(corrected_is_swap[core], true_is_swap[core])
    assert not corrected_is_swap[:14].any()   # no false positives before the swap
    assert not corrected_is_swap[22:].any()   # or after it


def test_gmm_labeler_fit_and_relabel_matches_relabel_sequence():
    rng = np.random.default_rng(12)
    labels, swapped, _truth, ref_frames = _synthetic_two_finger_recording(rng)
    marker_idxs = [3, 4]
    anchor_labels = ("Wrist", "IndexMCPref", "MiddleMCPref")

    direct = gl.relabel_sequence(
        swapped, labels, anchor_labels, ref_frames, marker_idxs,
        n_hypotheses=2, n_components=1, process_var=5.0, obs_var=4.0,
    )
    labeler = gl.GMMLabeler.fit(swapped, labels, anchor_labels, ref_frames,
                                marker_idxs, n_components=1)
    via_class = labeler.relabel(swapped, labels, n_hypotheses=2,
                                process_var=5.0, obs_var=4.0)
    np.testing.assert_array_equal(direct, via_class)
