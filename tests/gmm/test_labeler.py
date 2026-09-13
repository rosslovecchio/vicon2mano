"""Tests for vicon2mano.gmm_labeler (strategyGMM).

Built up step by step alongside the module: this file currently covers
step 1 (rigid local frame) and step 2 (per-marker GMM fitting). Hypothesis
generation (top-N assignment) and temporal selection (Kalman + Viterbi)
get their own test sections as those pieces land.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vicon2mano.strategies.gmm import labeler as gl


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


@pytest.mark.parametrize("good_first", [True, False])
def test_viterbi_select_uses_the_emission_score_to_break_a_temporal_tie(good_first):
    # The dual of the test above: there, emission was equal and transition
    # had to decide. Here transition is *deliberately symmetric* -- both
    # filters are seeded at the midpoint between the two observations, so
    # every pairing has an identical Kalman residual -- and only the
    # emission (spatial GMM) score separates the hypotheses.
    #
    # Regression guard for the paper's Eq. 1 term being dropped from the
    # trellis. With emission ignored, the winner was decided by the
    # hypothesis' *position in the list* (argmax over tied scores returns
    # the first), so this passed for one ordering and failed for the other
    # -- hence parametrising over both orderings rather than trusting one.
    marker_order = [0, 1]
    mid = np.array([5.0, 0.0, 0.0])
    obs = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]) for _ in range(4)]

    identity = (np.array([0, 1]), 0.0)        # cost 0 -> emission 0 (best)
    swapped = (np.array([1, 0]), 1e6)         # huge cost -> terrible emission
    hyps = [identity, swapped] if good_first else [swapped, identity]

    result = gl.viterbi_select(
        marker_order, [list(hyps)] * 4, obs,
        init_positions={0: mid, 1: mid},
    )
    for t, assign in enumerate(result):
        np.testing.assert_array_equal(
            assign, np.array([0, 1]),
            err_msg=f"frame {t}: spatially implausible hypothesis won")


def _tie_hyps(n=5):
    """n frames of two hypotheses over two well-separated observations."""
    obs = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]) for _ in range(n)]
    hyps = [[(np.array([0, 1]), 0.0), (np.array([1, 0]), 1e6)] for _ in range(n)]
    return hyps, obs


@pytest.mark.parametrize("seeds", [
    {0: np.full(3, np.nan), 1: np.array([10.0, 0.0, 0.0])},   # one unknown
    {0: np.full(3, np.nan), 1: np.full(3, np.nan)},           # all unknown
])
def test_viterbi_select_survives_non_finite_seed_positions(seeds):
    # relabel_local seeds the filter bank from local_mm[init_frame], which
    # is NaN whenever that frame has a Vicon gap -- routine on real data.
    # The NaN propagated through predict/update into every transition
    # score, and because `nan > -inf` is False the node selected no bank at
    # all and crashed with "'NoneType' object is not subscriptable".
    hyps, obs = _tie_hyps()
    result = gl.viterbi_select([0, 1], hyps, obs, init_positions=seeds)
    assert len(result) == len(hyps)
    for assign in result:
        assert np.isfinite(assign).all()


def test_viterbi_select_survives_a_non_finite_observation():
    hyps, obs = _tie_hyps()
    obs[2] = np.array([[np.nan] * 3, [10.0, 0.0, 0.0]])
    result = gl.viterbi_select(
        [0, 1], hyps, obs,
        init_positions={0: np.zeros(3), 1: np.array([10.0, 0.0, 0.0])})
    assert len(result) == len(hyps)


def test_marker_kalman_treats_a_non_finite_seed_as_unknown_not_as_origin():
    # Seeding 0 at the normal uncertainty would assert the marker sits at
    # the local origin. An unknown seed must instead be dominated by the
    # first real observations.
    kf = gl.MarkerKalman(np.full(3, np.nan), process_var=0.5, obs_var=1.0)
    assert kf.P[0, 0] > 1e4
    assert np.isfinite(kf.x).all()
    for _ in range(5):
        kf.predict()
        kf.update(np.array([7.0, 0.0, 0.0]))
    np.testing.assert_allclose(kf.x[:3], [7.0, 0.0, 0.0], atol=0.1)


def test_marker_kalman_update_skips_a_non_finite_observation():
    # The paper's Section 3.2 occlusion handling: extrapolate from the
    # predicted state, omit the innovation update.
    kf = gl.MarkerKalman(np.array([1.0, 2.0, 3.0]))
    kf.predict()
    before = kf.x.copy()
    kf.update(np.array([np.nan, 0.0, 0.0]))
    np.testing.assert_array_equal(kf.x, before)
    assert np.isfinite(kf.x).all()


def test_viterbi_select_rejects_an_empty_first_frame():
    with pytest.raises(ValueError, match="no hypotheses"):
        gl.viterbi_select([0, 1], [[]], [np.zeros((2, 3))], init_positions={})


def test_viterbi_select_flags_an_unseeded_unmatched_marker_as_unknown_not_origin():
    # Marker 1 has no entry in init_positions AND is unmatched (-1) in the
    # only t=0 hypothesis -- there is no observation to fall back to either.
    # The old fallback silently seeded it at np.zeros(3) with the filter's
    # *normal* starting confidence, asserting "this marker is at the local
    # origin" rather than flagging it as genuinely unknown. Unreachable via
    # relabel_local (which always supplies a complete init_positions), but
    # nothing enforced that for a direct viterbi_select caller.
    mo = [0, 1]
    obs0 = np.array([[0.0, 0.0, 0.0]])          # only 1 observation present
    hyp0 = [(np.array([0, -1]), 0.0)]           # marker 1 necessarily unmatched
    result = gl.viterbi_select(mo, [hyp0], [obs0], init_positions={0: obs0[0]})
    # Must not crash, and the frame must resolve normally.
    np.testing.assert_array_equal(result[0], np.array([0, -1]))


def test_marker_kalman_gate_loglik_raises_on_a_non_positive_definite_covariance():
    # (I - KH)P is a numerically fragile covariance update form; if P ever
    # loses positive-definiteness, slogdet's sign silently flips logdet's
    # meaning rather than erroring. gate_loglik is now called on every
    # occluded marker in every hypothesis every frame, far more often than
    # residual_loglik alone ever was, so this must fail loudly rather than
    # returning a nonsense score that could silently win a comparison.
    kf = gl.MarkerKalman(np.zeros(3), obs_var=1.0)
    kf.P = np.diag([1.0, -1000.0, 1.0, 1.0, 1.0, 1.0])   # indefinite, dominates R
    with pytest.raises(np.linalg.LinAlgError, match="not positive-definite"):
        kf.gate_loglik()


def test_marker_kalman_residual_loglik_raises_on_a_non_positive_definite_covariance():
    kf = gl.MarkerKalman(np.zeros(3), obs_var=1.0)
    kf.P = np.diag([1.0, -1000.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError, match="not positive-definite"):
        kf.residual_loglik(np.array([0.0, 0.0, 0.0]))


def test_transition_score_prefers_a_plausible_match_over_dropping():
    # Regression guard: unmatched markers used to score a fixed 0, which
    # beat *any* real match (a Gaussian log-density is generically negative
    # for non-trivial covariance) regardless of fit quality. A hypothesis
    # matching a marker to an observation squarely inside the filter's own
    # 95% gate must now beat one that drops it.
    mo = [0]
    obs0 = np.array([[0.0, 0.0, 0.0]])
    hyp0 = [(np.array([0]), 0.0)]
    obs1_good = np.array([[2.0, 0.0, 0.0]])       # small, plausible move
    hyps1 = [(np.array([0]), 0.0), (np.array([-1]), 0.0)]

    result = gl.viterbi_select(mo, [hyp0, hyps1], [obs0, obs1_good],
                               init_positions={0: obs0[0]})
    np.testing.assert_array_equal(result[1], np.array([0]))


def test_transition_score_prefers_dropping_an_implausible_match():
    # The complementary case: a match landing well outside the filter's
    # 95% gate (obviously the wrong marker, or a bad detection) must still
    # lose to leaving the marker unassigned -- the fix must not overcorrect
    # into always preferring a match.
    mo = [0]
    obs0 = np.array([[0.0, 0.0, 0.0]])
    hyp0 = [(np.array([0]), 0.0)]
    obs1_bad = np.array([[500.0, 0.0, 0.0]])      # absurd for obs_var=25 default
    hyps1 = [(np.array([0]), 0.0), (np.array([-1]), 0.0)]

    result = gl.viterbi_select(mo, [hyp0, hyps1], [obs0, obs1_bad],
                               init_positions={0: obs0[0]})
    np.testing.assert_array_equal(result[1], np.array([-1]))


def test_marker_kalman_gate_loglik_gap_is_invariant_to_the_filters_scale():
    # gate_loglik and residual_loglik share the same logdet(S) term, so it
    # cancels in their difference: how much a plausible match beats
    # dropping the marker should depend only on how far inside the 95%
    # gate the residual sits (in Mahalanobis terms), not on whether the
    # filter happens to be tightly converged or still wide after a recent
    # gap. A fixed-constant miss score (the pre-fix behaviour) would instead
    # make that comparison arbitrarily easier or harder to win purely as a
    # side effect of the filter's unrelated current uncertainty.
    residual = np.array([0.05, 0.0, 0.0])
    gaps = []
    for scale in (1.0, 10.0, 50.0):
        kf = gl.MarkerKalman(np.zeros(3), process_var=0.1, obs_var=0.1)
        kf.P *= scale
        match_ll, _ = kf.residual_loglik(residual)
        gaps.append(match_ll - kf.gate_loglik())
    # R does not scale with P, so the invariance is only approximate --
    # exact would require R = 0, which isn't a realistic filter.
    np.testing.assert_allclose(gaps, gaps[0], atol=1e-3)


# ---------------------------------------------------------------------------
# Step 4b: relabel_local's init_frame -- forward/backward split
# ---------------------------------------------------------------------------


def test_relabel_local_labels_frames_before_a_mid_window_init_frame():
    # init_frame need not be (and in real trials usually isn't) frame 0 --
    # relabel_local is handed the whole local array and picks whichever
    # frame is first known-good. The old implementation unconditionally
    # started its Viterbi trellis at array index 0 while seeding the filter
    # bank from local_mm[init_frame] -- physically incoherent whenever
    # init_frame != 0, and it left frames before init_frame covered only by
    # whatever the (wrongly-seeded) forward pass produced.
    T = 7
    local = np.zeros((T, 2, 3))
    for t in range(T):
        local[t, 0] = [t * 1.0, 0, 0]
        local[t, 1] = [t * 1.0, 50, 0]      # far apart -- spatially unambiguous
    local[0] = np.nan                        # frame 0 itself is unusable as a seed

    gmms = gl.fit_marker_gmms(local, [0, 1], np.arange(1, T), n_components=1,
                              min_samples=1)
    mapping = gl.relabel_local(local, gmms, [0, 1], n_hypotheses=3, init_frame=3)

    assert (mapping[0] == -1).all()          # genuinely unusable frame stays -1
    for t in range(1, T):
        np.testing.assert_array_equal(mapping[t], [0, 1])


def test_relabel_local_recovers_a_swap_entirely_before_init_frame():
    # The case the old seeding bug could not get right even in principle:
    # a swap confined to frames that precede init_frame, in a region the
    # old single forward pass covered only by accident.
    T = 10
    local = np.zeros((T, 2, 3))
    for t in range(T):
        local[t, 0] = [0, 0, 0]
        local[t, 1] = [0, 60, 0]
    local[2:5, [0, 1]] = local[2:5, [1, 0]]

    gmms = gl.fit_marker_gmms(local, [0, 1], np.array([0, 1, 6, 7, 8, 9]),
                              n_components=1, min_samples=1)
    mapping = gl.relabel_local(local, gmms, [0, 1], n_hypotheses=3, init_frame=7)

    for t in list(range(2)) + list(range(5, T)):
        np.testing.assert_array_equal(mapping[t], [0, 1])
    for t in range(2, 5):
        np.testing.assert_array_equal(mapping[t], [1, 0])


@pytest.mark.parametrize("init_frame", [0, 4])
def test_relabel_local_handles_init_frame_at_either_boundary(init_frame):
    local = np.zeros((5, 1, 3))
    gmms = gl.fit_marker_gmms(local, [0], np.arange(5), n_components=1, min_samples=1)
    mapping = gl.relabel_local(local, gmms, [0], init_frame=init_frame)
    np.testing.assert_array_equal(mapping.ravel(), [0, 0, 0, 0, 0])


def test_relabel_local_handles_a_single_frame_window():
    local = np.zeros((1, 1, 3))
    gmms = gl.fit_marker_gmms(local, [0], np.array([0]), n_components=1, min_samples=1)
    mapping = gl.relabel_local(local, gmms, [0], init_frame=0)
    np.testing.assert_array_equal(mapping.ravel(), [0])


def test_relabel_local_rejects_an_out_of_range_init_frame():
    local = np.zeros((5, 1, 3))
    gmms = gl.fit_marker_gmms(local, [0], np.arange(5), n_components=1, min_samples=1)
    with pytest.raises(ValueError, match="out of range"):
        gl.relabel_local(local, gmms, [0], init_frame=5)


def test_viterbi_w_emission_zero_disables_the_spatial_term():
    # w_emission is the paper's Section 5 knob for trading spatial
    # likelihood against temporal smoothness; 0 must fully disable the
    # spatial term, leaving the transition-only behaviour intact.
    marker_order = [0, 1]
    mid = np.array([5.0, 0.0, 0.0])
    obs = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]) for _ in range(3)]
    hyps = [(np.array([1, 0]), 1e6), (np.array([0, 1]), 0.0)]

    weighted = gl.viterbi_select(marker_order, [hyps] * 3, obs,
                                 init_positions={0: mid, 1: mid})
    unweighted = gl.viterbi_select(marker_order, [hyps] * 3, obs,
                                   init_positions={0: mid, 1: mid},
                                   w_emission=0.0)
    np.testing.assert_array_equal(weighted[0], np.array([0, 1]))
    # transition is symmetric here, so with no spatial term the tie falls
    # back to list order -- the pre-fix behaviour.
    np.testing.assert_array_equal(unweighted[0], np.array([1, 0]))


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


# ---------------------------------------------------------------------------
# Step 6: regime-segmented anchor location
# ---------------------------------------------------------------------------


def _regime_recording(rng, T=600, swap_at=200, swap_until=400):
    """4 markers; the anchor triple is columns 0,1,2 with a 3-4-5 geometry.
    Between ``swap_at`` and ``swap_until`` column 2 is swapped with the
    distractor column 3, so that regime's geometry is wrong -- but the
    physical triple is still present, under columns (0, 1, 3).
    """
    base = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 0.0],
                     [70.0, 70.0, 15.0]])
    out = np.zeros((T, 4, 3))
    for t in range(T):
        th = 0.01 * t
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[t] = base @ R.T + np.array([t * 0.8, 0.0, 0.0])
        out[t] += rng.normal(scale=0.05, size=(4, 3))
    for t in range(swap_at, swap_until):
        out[t, 2], out[t, 3] = out[t, 3].copy(), out[t, 2].copy()
    return out


def test_segment_by_jumps_splits_at_regime_boundaries():
    rng = np.random.default_rng(30)
    markers = _regime_recording(rng)
    segs = gl.segment_by_jumps(markers, (0, 1, 2), jump_mm=3.0, min_len=20)
    starts = sorted(int(s[0]) for s in segs)
    # boundaries at the swap in and the swap out
    assert any(abs(s - 200) <= 1 for s in starts)
    assert any(abs(s - 400) <= 1 for s in starts)


def test_learn_reference_geometry_picks_the_recurring_one():
    # The correct geometry (3-4-5) appears in two separate regimes; the
    # corrupted one appears in a single (longer) regime. Recurrence across
    # independent regimes must win over raw length.
    rng = np.random.default_rng(31)
    markers = _regime_recording(rng, T=900, swap_at=150, swap_until=750)
    segs = gl.segment_by_jumps(markers, (0, 1, 2), jump_mm=3.0, min_len=20)
    ref = gl.learn_reference_geometry(markers, (0, 1, 2), segs)
    np.testing.assert_allclose(ref, [30.0, 40.0, 50.0], atol=1.5)


def test_solve_segment_anchors_repairs_a_swapped_regime():
    rng = np.random.default_rng(32)
    markers = _regime_recording(rng)
    segs = gl.segment_by_jumps(markers, (0, 1, 2), jump_mm=3.0, min_len=20)
    ref = gl.learn_reference_geometry(markers, (0, 1, 2), segs)
    tri = gl.solve_segment_anchors(markers, segs, ref, (0, 1, 2), tol_mm=2.0)

    # clean regimes keep their own labels
    np.testing.assert_array_equal(tri[100], [0, 1, 2])
    np.testing.assert_array_equal(tri[500], [0, 1, 2])
    # the swapped regime is repaired onto the physical triple (0, 1, 3)
    assert set(tri[300]) == {0, 1, 3}


def test_solve_segment_anchors_refuses_when_no_triple_persists():
    # Pure noise: no triple holds the reference geometry across a regime,
    # so every frame must come back -1 rather than taking a coincidental
    # per-frame match (this is the frame-13410 failure mode).
    rng = np.random.default_rng(33)
    markers = rng.normal(scale=60.0, size=(300, 8, 3))
    segs = [np.arange(300)]
    tri = gl.solve_segment_anchors(markers, segs, np.array([30.0, 40.0, 50.0]),
                                    (0, 1, 2), tol_mm=1.0, min_persist=0.8)
    assert (tri == -1).all()


def test_solve_segment_anchors_prefers_labelled_columns_when_persistent():
    rng = np.random.default_rng(34)
    markers = _regime_recording(rng, swap_at=0, swap_until=0)   # never swapped
    segs = gl.segment_by_jumps(markers, (0, 1, 2), jump_mm=3.0, min_len=20)
    ref = gl.learn_reference_geometry(markers, (0, 1, 2), segs)
    tri = gl.solve_segment_anchors(markers, segs, ref, (0, 1, 2), tol_mm=2.0)
    found = tri[:, 0] >= 0
    assert found.any()
    assert (tri[found] == np.array([0, 1, 2])).all()


def test_solve_segment_anchors_refuses_a_short_regime_repair():
    # A 40-frame regime whose labelled triple is wrong: the correct triple
    # is present and matches 100% of the regime, but 40 frames is far too
    # little evidence to justify a wholesale relabel against ~9k candidate
    # triples (the P7 frame 38315 failure). Keeping the labelled triple
    # needs no such evidence, so only the *repair* is length-gated.
    rng = np.random.default_rng(35)
    markers = _regime_recording(rng, T=500, swap_at=200, swap_until=240)
    segs = gl.segment_by_jumps(markers, (0, 1, 2), jump_mm=3.0, min_len=20)
    ref = gl.learn_reference_geometry(markers, (0, 1, 2), segs)

    lax = gl.solve_segment_anchors(markers, segs, ref, (0, 1, 2),
                                    tol_mm=2.0, min_repair_len=10)
    assert set(lax[220]) == {0, 1, 3}          # repair taken when ungated

    strict = gl.solve_segment_anchors(markers, segs, ref, (0, 1, 2),
                                       tol_mm=2.0, min_repair_len=200)
    assert (strict[220] == -1).all()           # refused: regime too short
    np.testing.assert_array_equal(strict[100], [0, 1, 2])   # clean regime kept


def test_segment_by_jumps_bridges_short_gaps():
    # A 2-frame dropout in the anchors is not a regime change. With
    # bridge=0 it splits the recording; with the default it must not.
    rng = np.random.default_rng(40)
    markers = _regime_recording(rng, T=400, swap_at=0, swap_until=0)
    markers[150:152, :3] = np.nan          # brief anchor gap, no relabel

    unbridged = gl.segment_by_jumps(markers, (0, 1, 2), min_len=20, bridge=0)
    bridged = gl.segment_by_jumps(markers, (0, 1, 2), min_len=20, bridge=5)
    assert len(unbridged) > len(bridged)
    assert len(bridged) == 1               # one continuous regime
    assert len(bridged[0]) == 400


def test_segment_by_jumps_still_splits_on_a_long_gap():
    rng = np.random.default_rng(41)
    markers = _regime_recording(rng, T=400, swap_at=0, swap_until=0)
    markers[150:170, :3] = np.nan          # 20 frames -- beyond the bridge
    segs = gl.segment_by_jumps(markers, (0, 1, 2), min_len=20, bridge=5)
    assert len(segs) > 1


def test_segment_by_jumps_still_splits_on_a_real_jump_inside_a_gap_run():
    # Bridging must not paper over a genuine relabel that happens to sit
    # next to a short gap: the carried geometry is compared against the
    # frame after the gap, so the jump is still seen.
    rng = np.random.default_rng(42)
    markers = _regime_recording(rng, T=400, swap_at=200, swap_until=400)
    markers[199:201, :3] = np.nan
    segs = gl.segment_by_jumps(markers, (0, 1, 2), min_len=20, bridge=5)
    assert len(segs) > 1
