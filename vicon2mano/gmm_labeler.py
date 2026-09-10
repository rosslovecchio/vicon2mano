"""Spatial-GMM + Viterbi marker labeler ("strategyGMM").

Model-free alternative to the MANO-based relabelling in ``mano_relabel.py``,
following Alexanderson, O'Sullivan & Beskow, "Real-time labeling of
non-rigid motion capture marker sets" (Computers & Graphics, 2017):

1. **Rigid local frame.** Every marker position is expressed relative to a
   frame rigidly attached to the hand (built from 3 stable anchor markers),
   so a finger marker's distribution is pose-of-the-hand invariant even
   though the finger itself is non-rigid.
2. **Per-marker GMM (training).** Each marker's spatial distribution in
   that local frame is modelled independently with a small Gaussian
   Mixture Model, fit on trusted/reference frames only.
3. **Hypothesis generation (spatial, per frame).** The log-likelihood of
   every observation under every marker's GMM forms a cost matrix; the
   top-N assignments (Murty's algorithm reduction to repeated calls of the
   Hungarian algorithm) are the frame's assignment hypotheses.
4. **Hypothesis selection (temporal).** A bank of Kalman filters (one per
   marker per live hypothesis) scores transitions between consecutive
   frames' hypotheses; the Viterbi algorithm picks the most probable path
   through the hypothesis trellis over time.

No third-party ML dependency is added: the GMM is a small hand-rolled
diagonal-covariance EM (numpy), and the Kalman filter is a standard
constant-velocity filter (numpy). Both mirror this repo's existing
preference for dependency-free numeric routines (see
``correspondence._cluster_two``).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# 1. Rigid local frame
# ---------------------------------------------------------------------------


def rigid_frames(
    markers_mm: np.ndarray,     # (T, N, 3)
    origin_idx: int,
    x_idx: int,
    y_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rigid orthonormal frame from 3 anchor marker indices.

    Modelled on ``mano_relabel._palm_frames`` (same cross-product
    construction), but built directly from raw marker columns rather than
    fitted MANO joints, so it needs no MANO fit -- only 3 already-trusted
    anchor markers (e.g. wrist/Palm2, Index-MCP, Middle-MCP).

    Returns ``(R, origin)``: ``R`` is ``(T, 3, 3)`` with rows as the local
    x/y/z axes (world -> local is ``R @ (p - origin)``); ``origin`` is
    ``(T, 3)``. Frames for frames where an anchor is non-finite are NaN.
    """
    o = markers_mm[:, origin_idx]
    xp = markers_mm[:, x_idx]
    yp = markers_mm[:, y_idx]
    x = _normalise(xp - o)
    y_raw = yp - o
    z = _normalise(np.cross(x, y_raw))
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)     # (T, 3, 3): rows are axes
    return R, o


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def to_local(markers_mm: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """World -> local. ``markers_mm``: (T, N, 3); ``R``/``origin``: (T, 3)/(T,3,3)."""
    delta = markers_mm - origin[:, None, :]
    return np.einsum("tab,tnb->tna", R, delta)


def learn_anchor_triangle(
    markers_mm: np.ndarray,       # (T, N, 3)
    anchor_idxs: tuple[int, int, int],
    *,
    bin_mm: float = 0.5,
    refine_mm: float = 1.5,
) -> np.ndarray:
    """The 3 mutual distances of the anchor plate, learned from the data.

    Returns ``[d01, d02, d12]`` in mm. Uses the *modal* value of each
    pairwise distance (tallest ``bin_mm`` histogram bin, then the mean of
    everything within ``refine_mm`` of it) rather than the median: on a
    recording where the anchor markers are themselves mislabelled for a
    substantial share of frames, the median lands between the correct and
    incorrect configurations and matches neither. The correct geometry is
    the single most *concentrated* value -- the plate is rigid, so its
    true distances repeat to within measurement noise, while mislabelled
    configurations scatter over tens of mm.

    Needs no external verdict (no cascade, no manual reference frames):
    the plate's own rigidity is the ground truth.
    """
    pairs = ((0, 1), (0, 2), (1, 2))
    out = []
    for a, b in pairs:
        v = np.linalg.norm(markers_mm[:, anchor_idxs[a]] - markers_mm[:, anchor_idxs[b]], axis=1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out.append(np.nan)
            continue
        hist, edges = np.histogram(v, bins=np.arange(0, v.max() + bin_mm, bin_mm))
        mode = edges[int(np.argmax(hist))] + bin_mm / 2
        near = v[np.abs(v - mode) < refine_mm]
        out.append(float(near.mean()) if near.size else float(mode))
    return np.array(out)


def locate_anchor_triangle(
    markers_mm: np.ndarray,        # (T, N, 3)
    ref_d: np.ndarray,             # [d01, d02, d12] from learn_anchor_triangle
    labelled: tuple[int, int, int],
    *,
    tol_mm: float = 2.5,
) -> np.ndarray:
    """Find, per frame, which 3 markers actually form the anchor plate.

    Returns ``(T, 3)`` int array of marker indices (``-1`` where no triple
    in the cloud matches the reference triangle within ``tol_mm``).

    Rather than trusting that the columns named ``Palm1/2/3`` really hold
    the palm plate -- on real data they demonstrably do not, see the
    P7/Trial1_handsonly rolling-mislabelling episode -- this searches the
    whole frame's point cloud for a triple whose three mutual distances
    match the rigid plate's. That both *validates* the frame and *repairs*
    the anchors when their labels were swapped, with no dependency on the
    quality cascade or any other external verdict.

    ``labelled`` (the nominal anchor columns) is preferred whenever it
    matches, so a correctly-labelled frame keeps its own anchors and the
    frame-to-frame anchor identity stays stable; only when the labelled
    triple fails does the best-matching alternative get used.
    """
    T, N, _ = markers_mm.shape
    out = np.full((T, 3), -1, dtype=int)

    # All ordered triples, precomputed once. N is ~20 for a hand marker
    # set, so this is ~8k rows -- small enough to score by brute force per
    # frame with numpy, and the whole search costs well under a minute for
    # a 47k-frame trial.
    tri = np.array([(a, b, c)
                    for a in range(N) for b in range(N) for c in range(N)
                    if a != b and b != c and a != c])
    ta, tb, tc = tri[:, 0], tri[:, 1], tri[:, 2]

    for t in range(T):
        pts = markers_mm[t]
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        err = (np.abs(d[ta, tb] - ref_d[0])
               + np.abs(d[ta, tc] - ref_d[1])
               + np.abs(d[tb, tc] - ref_d[2]))
        ok = ((np.abs(d[ta, tb] - ref_d[0]) <= tol_mm)
              & (np.abs(d[ta, tc] - ref_d[1]) <= tol_mm)
              & (np.abs(d[tb, tc] - ref_d[2]) <= tol_mm))
        if not ok.any():
            continue
        lab_row = np.flatnonzero((ta == labelled[0]) & (tb == labelled[1])
                                  & (tc == labelled[2]))
        if lab_row.size and ok[lab_row[0]]:
            out[t] = labelled                 # labels were right -- keep them
        else:
            cand = np.flatnonzero(ok)
            out[t] = tri[cand[np.argmin(err[cand])]]
    return out


def rigid_frames_from_triangle(
    markers_mm: np.ndarray,        # (T, N, 3)
    triangle: np.ndarray,          # (T, 3) from locate_anchor_triangle
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rigid frame built from a *per-frame* anchor triple.

    Same construction as :func:`rigid_frames`, but the 3 anchor markers may
    differ frame to frame (because ``locate_anchor_triangle`` re-identified
    them). Frames with no located triangle come back NaN.
    """
    T = markers_mm.shape[0]
    R = np.full((T, 3, 3), np.nan)
    origin = np.full((T, 3), np.nan)
    found = triangle[:, 0] >= 0
    idx = np.flatnonzero(found)
    if idx.size == 0:
        return R, origin
    o = markers_mm[idx, triangle[idx, 0]]
    xp = markers_mm[idx, triangle[idx, 1]]
    yp = markers_mm[idx, triangle[idx, 2]]
    x = _normalise(xp - o)
    z = _normalise(np.cross(x, yp - o))
    R[idx] = np.stack([x, np.cross(z, x), z], axis=1)
    origin[idx] = o
    return R, origin


def mask_untrustworthy_frames(local_mm: np.ndarray, anchor_valid: np.ndarray) -> np.ndarray:
    """NaN out every marker's local position at frames with bad anchors.

    ``rigid_frames``/``to_local`` compute a coordinate frame from 3 anchor
    markers unconditionally -- if those anchors are themselves mislabelled
    that frame (a real failure mode: see the P7/Trial1_handsonly
    Ring1<->Pinky1 investigation, where a rolling mislabelling cascade
    across Palm1/Palm2/Thumb1 corrupted the local frame for ~1600 frames),
    every other marker's local coordinate silently becomes garbage with no
    signal that anything is wrong.

    Pass a per-frame boolean ``anchor_valid`` (e.g. from
    ``label_quality_cascade``'s CORRECT verdict on the anchor labels) to
    NaN those frames out here, upstream of both training
    (``fit_marker_gmms`` already skips NaN samples) and evaluation
    (``_build_hypotheses`` already treats a NaN observation as an
    occlusion, extrapolated via the Kalman predict step) -- no separate
    handling needed in either path.
    """
    out = local_mm.copy()
    out[~anchor_valid] = np.nan
    return out


def to_world(local_mm: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Local -> world, inverse of :func:`to_local`.

    ``R`` is orthonormal (rows are the local axes), so its inverse is its
    transpose: ``delta = R^T @ local``, i.e. contract R's *first* non-batch
    axis (the axis index) against local's last axis, leaving R's second
    axis (the world-component index) as output -- not the naive
    letter-swap ``"tba,..."``, which silently applies ``R`` instead of
    ``R^T`` and only agrees with it when ``R`` happens to be symmetric.
    """
    world = np.einsum("tab,tna->tnb", R, local_mm)
    return world + origin[:, None, :]


# ---------------------------------------------------------------------------
# 2. Per-marker GMM (hand-rolled diagonal-covariance EM)
# ---------------------------------------------------------------------------


@dataclass
class GMMParams:
    weights: np.ndarray   # (C,)
    means: np.ndarray     # (C, 3)
    variances: np.ndarray  # (C, 3) diagonal


def _gaussian_logpdf_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(x; mean, diag(var)) for x: (..., 3)."""
    var = np.maximum(var, 1e-12)
    d = x.shape[-1]
    diff2 = (x - mean) ** 2 / var
    return -0.5 * (d * np.log(2 * np.pi) + np.log(var).sum() + diff2.sum(axis=-1))


def fit_gmm_diag(
    samples: np.ndarray,       # (F, 3)
    *,
    n_components: int = 3,
    n_iter: int = 50,
    reg: float = 1e-6,
    seed: int = 0,
) -> GMMParams:
    """Fit a diagonal-covariance GMM via EM. Deterministic given ``seed``.

    Small and dependency-free (no scikit-learn), matching this repo's
    convention of hand-rolling small numeric routines rather than adding a
    library for one algorithm. Falls back to a single component if there
    are fewer samples than components.
    """
    samples = np.asarray(samples, dtype=float)
    F = samples.shape[0]
    C = max(1, min(n_components, F))
    rng = np.random.default_rng(seed)

    # k-means++-lite init: pick C samples spread apart.
    idx = [int(rng.integers(F))]
    for _ in range(1, C):
        d = np.min(
            [np.linalg.norm(samples - samples[i], axis=-1) for i in idx], axis=0
        )
        idx.append(int(np.argmax(d)))
    means = samples[idx].copy()
    variances = np.tile(samples.var(axis=0, keepdims=True) + reg, (C, 1))
    weights = np.full(C, 1.0 / C)

    for _ in range(n_iter):
        # E-step
        logp = np.stack(
            [_gaussian_logpdf_diag(samples, means[c], variances[c]) + np.log(weights[c] + 1e-300)
             for c in range(C)],
            axis=1,
        )  # (F, C)
        m = logp.max(axis=1, keepdims=True)
        resp = np.exp(logp - m)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-300  # (F, C)

        # M-step
        nk = resp.sum(axis=0) + 1e-12       # (C,)
        weights = nk / F
        means = (resp.T @ samples) / nk[:, None]
        diff2 = samples[:, None, :] - means[None, :, :]
        variances = np.einsum("fc,fca->ca", resp, diff2 ** 2) / nk[:, None] + reg

    return GMMParams(weights=weights, means=means, variances=variances)


def gmm_loglik(x: np.ndarray, gmm: GMMParams) -> np.ndarray:
    """Log-likelihood of point(s) ``x`` (..., 3) under ``gmm``."""
    comp = np.stack(
        [_gaussian_logpdf_diag(x, gmm.means[c], gmm.variances[c]) + np.log(gmm.weights[c] + 1e-300)
         for c in range(len(gmm.weights))],
        axis=-1,
    )
    m = comp.max(axis=-1)
    return m + np.log(np.exp(comp - m[..., None]).sum(axis=-1))


def fit_marker_gmms(
    local_mm: np.ndarray,          # (T, N, 3) marker positions in a rigid local frame
    marker_idxs: list[int],
    ref_frames: np.ndarray,        # frame indices trusted for training
    *,
    n_components: int = 3,
    n_iter: int = 50,
    min_samples: int = 20,
) -> dict[int, GMMParams]:
    """One GMM per marker index, trained on ``ref_frames`` only.

    Markers with fewer than ``min_samples`` finite reference samples are
    skipped (no spatial prior can be trusted from that little data) --
    mirrors ``mano_relabel.calibrate_marker_offsets``'s ``min_samples`` gate.
    """
    out: dict[int, GMMParams] = {}
    for m in marker_idxs:
        pts = local_mm[ref_frames, m]
        good = np.isfinite(pts).all(axis=1)
        pts = pts[good]
        if len(pts) < min_samples:
            continue
        out[m] = fit_gmm_diag(pts, n_components=n_components, n_iter=n_iter)
    return out


def loglik_matrix(
    obs_local: np.ndarray,             # (K, 3) observations in the local frame
    gmms: dict[int, GMMParams],
    marker_order: list[int],
    *,
    theta_min: float = -30.0,
) -> np.ndarray:
    """(M, K) log-likelihood matrix, ``M = len(marker_order)``.

    Mirrors the paper's ghost-marker tolerance: entries below ``theta_min``
    are clamped to it rather than left arbitrarily negative, so a wildly
    implausible pairing doesn't dominate cost comparisons through its
    magnitude alone (Section 3.1's ``theta_min`` filter).
    """
    M, K = len(marker_order), obs_local.shape[0]
    out = np.full((M, K), theta_min)
    for i, m in enumerate(marker_order):
        gmm = gmms.get(m)
        if gmm is None:
            continue
        ll = gmm_loglik(obs_local, gmm)
        out[i] = np.maximum(ll, theta_min)
    return out


# ---------------------------------------------------------------------------
# 3. Hypothesis generation: top-N assignments (Murty's algorithm)
# ---------------------------------------------------------------------------


def _lap_on_subset(
    cost: np.ndarray, forced: frozenset, forbidden: frozenset, M: int, K: int
):
    """Solve the LAP with ``forced`` pairs fixed and ``forbidden`` pairs banned.

    Returns ``(total_cost, pairs)`` or ``None`` if infeasible (a forbidden
    pair was unavoidable given the remaining rows/cols).
    """
    forced_rows = {r for r, _ in forced}
    forced_cols = {c for _, c in forced}
    avail_rows = [r for r in range(M) if r not in forced_rows]
    avail_cols = [c for c in range(K) if c not in forced_cols]
    pairs = list(forced)

    if avail_rows and avail_cols:
        sub = cost[np.ix_(avail_rows, avail_cols)].copy()
        BIG = 1e9
        for fr, fc in forbidden:
            if fr in avail_rows and fc in avail_cols:
                sub[avail_rows.index(fr), avail_cols.index(fc)] = BIG
        sr, sc = linear_sum_assignment(sub)
        for i, j in zip(sr, sc):
            if sub[i, j] >= BIG:
                return None
            pairs.append((avail_rows[i], avail_cols[j]))

    total = sum(cost[r, c] for r, c in pairs)
    return total, pairs


def top_n_assignments(
    cost: np.ndarray,      # (M, K), lower is better (e.g. negative log-lik)
    n: int,
) -> list[tuple[np.ndarray, float]]:
    """Return up to ``n`` best assignments, best (lowest-cost) first.

    Standard Murty's-algorithm reduction: repeatedly partition the
    best-so-far unexpanded node into subproblems that force a prefix of its
    pairs and forbid the next one, re-solving each subproblem's LAP with
    ``scipy.optimize.linear_sum_assignment`` -- no dedicated k-best solver
    is implemented or required.

    Each result is ``(assign, cost)`` where ``assign`` has length ``M`` and
    ``assign[i]`` is the observation index matched to marker ``i`` (or -1 if
    ``K < M`` and marker ``i`` was left unmatched).
    """
    cost = np.asarray(cost, dtype=float)
    M, K = cost.shape
    heap: list[tuple] = []
    counter = 0

    def push(forced: frozenset, forbidden: frozenset) -> None:
        nonlocal counter
        res = _lap_on_subset(cost, forced, forbidden, M, K)
        if res is None:
            return
        total, pairs = res
        heapq.heappush(heap, (total, counter, tuple(sorted(pairs)), forced, forbidden))
        counter += 1

    push(frozenset(), frozenset())
    results: list[tuple[np.ndarray, float]] = []
    while heap and len(results) < n:
        total, _, pairs, forced, forbidden = heapq.heappop(heap)
        assign = np.full(M, -1, dtype=int)
        for r, c in pairs:
            assign[r] = c
        results.append((assign, total))

        # Partition this accepted node: force an increasing prefix of its
        # own pairs (beyond what was already forced) and forbid the next.
        free_pairs = [p for p in pairs if p not in forced]
        prefix = set(forced)
        for p in free_pairs:
            push(frozenset(prefix), frozenset(forbidden | {p}))
            prefix = prefix | {p}
    return results


# ---------------------------------------------------------------------------
# 4. Hypothesis selection: Kalman filter + Viterbi
# ---------------------------------------------------------------------------


class MarkerKalman:
    """Constant-velocity Kalman filter for one marker's local-frame position.

    Simplified from the paper's position/velocity/acceleration state to
    position/velocity -- adequate for the ~frame-rate-scale motion here and
    keeps the filter bank (``N`` hypotheses x ``M`` markers) cheap to copy
    per Viterbi node.
    """

    def __init__(self, pos0: np.ndarray, *, process_var: float = 1.0, obs_var: float = 25.0):
        self.x = np.concatenate([pos0, np.zeros(3)])
        self.P = np.eye(6) * 100.0
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3)
        self.Q = np.eye(6) * process_var
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)
        self.R = np.eye(3) * obs_var

    def predict(self) -> np.ndarray:
        """Advance one step; returns the predicted position (3,)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def residual_loglik(self, obs: np.ndarray) -> tuple[float, np.ndarray]:
        """Log-likelihood of ``obs`` under the current (already-predicted)
        state, without committing an update. Returns ``(loglik, residual)``.
        """
        y = obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (3 * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(S, y))
        return float(ll), y

    def update(self, obs: np.ndarray) -> None:
        y = obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def copy(self) -> "MarkerKalman":
        out = MarkerKalman.__new__(MarkerKalman)
        out.x, out.P = self.x.copy(), self.P.copy()
        out.F, out.Q, out.H, out.R = self.F, self.Q, self.H, self.R
        return out


# Extrapolation cap for an occluded marker within a live hypothesis, mirroring
# the paper's tuned "10 frames" default (Section 3.2).
MAX_OCCLUSION_FRAMES = 10


def _transition_score(
    filters: dict[int, MarkerKalman],
    marker_order: list[int],
    obs_local: np.ndarray,      # (K, 3)
    assign: np.ndarray,         # (M,) obs index per marker, -1 = unmatched
) -> tuple[float, dict[int, np.ndarray]]:
    """Sum of per-marker Kalman residual log-likelihoods for one candidate
    hypothesis, given filters already ``predict()``-ed to this frame.

    Returns ``(score, predictions)`` where ``predictions`` is what each
    filter predicted this step (used for extrapolating occluded markers).
    """
    score = 0.0
    preds: dict[int, np.ndarray] = {}
    for i, m in enumerate(marker_order):
        kf = filters[m]
        pred = kf.x[:3]
        preds[m] = pred
        j = assign[i]
        if j < 0:
            continue
        ll, _ = kf.residual_loglik(obs_local[j])
        score += ll
    return score, preds


def viterbi_select(
    marker_order: list[int],
    hypotheses_per_frame: list[list[tuple[np.ndarray, float]]],
    observations_per_frame: list[np.ndarray],
    init_positions: dict[int, np.ndarray],
    *,
    process_var: float = 1.0,
    obs_var: float = 25.0,
) -> list[np.ndarray]:
    """Pick the most probable sequence of hypotheses via Viterbi + Kalman.

    ``hypotheses_per_frame[t]`` is the list of ``(assign, cost)`` candidates
    for frame ``t`` from :func:`top_n_assignments` (``cost`` here is unused;
    hypothesis scoring is spatial-only and already baked into which
    candidates were generated -- selection combines emission (spatial) and
    transition (temporal) scores per the paper's Viterbi trellis).

    Maintains one filter bank (dict[marker] -> MarkerKalman) per *live
    Viterbi path*, one per hypothesis slot at the current frame -- not one
    shared bank per hypothesis index, since two different incoming paths to
    the same hypothesis slot would otherwise silently share filter state.

    Returns the winning per-frame assignment array (length ``M``, marker ->
    observation index or -1), one per frame.
    """
    T = len(hypotheses_per_frame)
    if T == 0:
        return []

    # Initialise one filter bank per hypothesis slot at frame 0.
    banks: list[dict[int, MarkerKalman]] = []
    emission0 = []
    for assign, _cost in hypotheses_per_frame[0]:
        bank = {
            m: MarkerKalman(init_positions.get(m, observations_per_frame[0][assign[i]]
                             if assign[i] >= 0 else np.zeros(3)),
                             process_var=process_var, obs_var=obs_var)
            for i, m in enumerate(marker_order)
        }
        for i, m in enumerate(marker_order):
            j = assign[i]
            if j >= 0:
                bank[m].update(observations_per_frame[0][j])
        banks.append(bank)
        emission0.append(0.0)   # no prior transition at t=0

    path_score = emission0
    backptr: list[list[int]] = [[]]
    chosen_assign: list[list[np.ndarray]] = [[a for a, _ in hypotheses_per_frame[0]]]

    for t in range(1, T):
        obs = observations_per_frame[t]
        new_banks: list[dict[int, MarkerKalman]] = []
        new_scores: list[float] = []
        new_backptr: list[int] = []
        for assign, _cost in hypotheses_per_frame[t]:
            best_score, best_src, best_bank = -np.inf, 0, None
            for src, bank in enumerate(banks):
                predicted = {m: kf.copy() for m, kf in bank.items()}
                for kf in predicted.values():
                    kf.predict()
                trans, _preds = _transition_score(predicted, marker_order, obs, assign)
                total = path_score[src] + trans
                if total > best_score:
                    best_score, best_src, best_bank = total, src, predicted
            for i, m in enumerate(marker_order):
                j = assign[i]
                if j >= 0:
                    best_bank[m].update(obs[j])
            new_banks.append(best_bank)
            new_scores.append(best_score)
            new_backptr.append(best_src)
        banks, path_score = new_banks, new_scores
        backptr.append(new_backptr)
        chosen_assign.append([a for a, _ in hypotheses_per_frame[t]])

    # Backtrack from the best-scoring final hypothesis.
    best_final = int(np.argmax(path_score))
    result = [None] * T
    slot = best_final
    for t in range(T - 1, -1, -1):
        result[t] = chosen_assign[t][slot]
        slot = backptr[t][slot] if t > 0 else slot
    return result


# ---------------------------------------------------------------------------
# 5. Sequence-level driver: relabel already-present, fixed-column markers
# ---------------------------------------------------------------------------
#
# Unlike DeepLabeler (which voxelises the whole raw, order-agnostic point
# cloud), this operates on a recording whose columns are already labelled
# and mostly correct -- the anchor markers used for the rigid local frame
# must themselves be trustworthy every frame. That fits this repo's actual
# failure mode (documented in CLAUDE.md): the cascade's distance checks
# already establish which markers are the stable wrist/palm anchors, and
# the open problem is two *already-labelled* adjacent-finger markers
# swapping identity, not recovering labels from a wholly anonymous cloud.


def _build_hypotheses(
    local: np.ndarray,             # (T, N, 3)
    gmms: dict[int, GMMParams],
    kept: list[int],
    *,
    n_hypotheses: int,
    theta_min: float,
    init_frame: int,
) -> tuple[list[list[tuple[np.ndarray, float]]], list[np.ndarray]]:
    """Per-frame hypotheses + observation arrays, occlusion-safe.

    A marker occluded at frame ``t`` (non-finite local position -- a real
    Vicon gap) is dropped from that frame's *cost matrix columns* before
    calling ``top_n_assignments`` (``linear_sum_assignment`` rejects NaN
    entries outright), then remapped back to ``kept``-index space so
    hypothesis assignment values always mean "index into ``kept``",
    consistently across frames regardless of which markers were visible.
    An occluded marker's own slot simply comes back -1 that frame, which
    ``viterbi_select`` treats as "extrapolate via the Kalman predict step"
    -- the paper's Section 3.2 occlusion handling.

    ``obs_per_frame`` keeps the *full* (possibly-NaN) ``(K, 3)`` row per
    frame (not the compacted present-only subset) so hypothesis assignment
    values -- already in ``kept``-index space -- index directly into it.
    """
    hyps_per_frame: list[list[tuple[np.ndarray, float]]] = []
    obs_per_frame: list[np.ndarray] = []
    T = local.shape[0]
    for t in range(T):
        obs_full = local[t, kept]                       # (K, 3), K == len(kept)
        finite = np.isfinite(obs_full).all(axis=1)
        present = np.flatnonzero(finite)
        cost = -loglik_matrix(obs_full[present], gmms, kept, theta_min=theta_min)
        n_here = 1 if t == init_frame else n_hypotheses
        raw = top_n_assignments(cost, n_here)
        remapped = []
        for a, c in raw:
            if present.size:
                idx_arr = present[np.clip(a, 0, len(present) - 1)]
            else:
                idx_arr = np.full_like(a, -1)
            remapped.append((np.where(a >= 0, idx_arr, -1), c))
        hyps_per_frame.append(remapped)
        obs_per_frame.append(obs_full)
    return hyps_per_frame, obs_per_frame


def relabel_local(
    local_mm: np.ndarray,          # (T, N, 3) already in the rigid local frame
    gmms: dict[int, GMMParams],
    marker_idxs: list[int],
    *,
    n_hypotheses: int = 5,
    theta_min: float = -30.0,
    process_var: float = 1.0,
    obs_var: float = 25.0,
    init_frame: int = 0,
) -> np.ndarray:
    """Hypothesis generation + Viterbi selection on precomputed local coords.

    The lowest-level public entry point, for callers that build the local
    frame themselves -- notably one built from a *per-frame* anchor triple
    (``locate_anchor_triangle`` + ``rigid_frames_from_triangle``), which
    the fixed-anchor-label API cannot express.

    Returns ``(T, len(kept))``; ``[t, i]`` is the index into ``kept`` of
    the marker whose observed position belongs in slot ``i`` at frame
    ``t``, or -1 where that slot had no usable observation.
    """
    kept = [m for m in marker_idxs if m in gmms]
    if not kept:
        return np.zeros((local_mm.shape[0], 0), dtype=int)
    hyps_per_frame, obs_per_frame = _build_hypotheses(
        local_mm, gmms, kept, n_hypotheses=n_hypotheses, theta_min=theta_min,
        init_frame=init_frame)
    init_positions = {m: local_mm[init_frame, m] for m in kept}
    return np.stack(viterbi_select(
        kept, hyps_per_frame, obs_per_frame, init_positions,
        process_var=process_var, obs_var=obs_var))


def relabel_sequence(
    markers_mm: np.ndarray,        # (T, N, 3)
    labels: list[str],
    anchor_labels: tuple[str, str, str],   # (origin, x, y) label names
    ref_frames: np.ndarray,
    marker_idxs: list[int],        # columns to model + potentially swap
    *,
    n_hypotheses: int = 5,
    n_components: int = 3,
    theta_min: float = -30.0,
    process_var: float = 1.0,
    obs_var: float = 25.0,
    init_frame: int = 0,
    anchor_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame identity assignment for ``marker_idxs`` via GMM + Viterbi.

    Returns ``(T, len(marker_idxs))``: entry ``[t, i]`` is the index (into
    ``marker_idxs``) of the column whose *observed position* should occupy
    slot ``i`` at frame ``t`` (-1 if occluded that frame). Feed the result
    through ``mano_relabel.apply_relabel`` (same ``{slot: source}``
    convention, after converting to marker-index space) to actually permute
    positions.

    Markers with no fitted GMM (too few reference samples -- see
    ``fit_marker_gmms``) are dropped from ``marker_idxs`` before assignment;
    check the returned array's second dimension against the input.

    ``anchor_valid``: optional ``(T,)`` bool, per-frame trustworthiness of
    the 3 anchor markers themselves (e.g. from a cascade CORRECT verdict on
    ``anchor_labels``). Frames where they are *not* trustworthy have their
    local coordinates NaN'd via ``mask_untrustworthy_frames`` before fitting
    or assigning -- see that function's docstring for why this matters (a
    real rolling-mislabelling episode found on P7/Trial1_handsonly
    corrupted the anchors themselves for ~1600 frames, which silently wrecks
    every other marker's local coordinate if left unguarded).
    """
    name_to_idx = {l: i for i, l in enumerate(labels)}
    origin_i, x_i, y_i = (name_to_idx[a] for a in anchor_labels)
    R, o = rigid_frames(markers_mm, origin_i, x_i, y_i)
    local = to_local(markers_mm, R, o)
    if anchor_valid is not None:
        local = mask_untrustworthy_frames(local, anchor_valid)

    gmms = fit_marker_gmms(local, marker_idxs, ref_frames, n_components=n_components)
    return relabel_local(local, gmms, marker_idxs, n_hypotheses=n_hypotheses,
                         theta_min=theta_min, process_var=process_var,
                         obs_var=obs_var, init_frame=init_frame)


class GMMLabeler:
    """Fitted per-participant GMM model, reusable across trials.

    Usage::

        labeler = GMMLabeler.fit(markers_mm, labels, anchor_labels,
                                  ref_frames, marker_idxs)
        mapping_seq = labeler.relabel(markers_mm, labels)   # (T, M) source idx
    """

    def __init__(self, gmms: dict[int, GMMParams], marker_idxs: list[int],
                 anchor_labels: tuple[str, str, str]):
        self.gmms = gmms
        self.marker_idxs = marker_idxs
        self.anchor_labels = anchor_labels

    @classmethod
    def fit(
        cls,
        markers_mm: np.ndarray,
        labels: list[str],
        anchor_labels: tuple[str, str, str],
        ref_frames: np.ndarray,
        marker_idxs: list[int],
        *,
        n_components: int = 3,
        anchor_valid: np.ndarray | None = None,
    ) -> "GMMLabeler":
        """See ``relabel_sequence`` for the ``anchor_valid`` contract."""
        name_to_idx = {l: i for i, l in enumerate(labels)}
        origin_i, x_i, y_i = (name_to_idx[a] for a in anchor_labels)
        R, o = rigid_frames(markers_mm, origin_i, x_i, y_i)
        local = to_local(markers_mm, R, o)
        if anchor_valid is not None:
            local = mask_untrustworthy_frames(local, anchor_valid)
        gmms = fit_marker_gmms(local, marker_idxs, ref_frames, n_components=n_components)
        return cls(gmms, [m for m in marker_idxs if m in gmms], anchor_labels)

    def relabel(
        self,
        markers_mm: np.ndarray,
        labels: list[str],
        *,
        n_hypotheses: int = 5,
        theta_min: float = -30.0,
        process_var: float = 1.0,
        obs_var: float = 25.0,
        anchor_valid: np.ndarray | None = None,
        init_frame: int = 0,
    ) -> np.ndarray:
        name_to_idx = {l: i for i, l in enumerate(labels)}
        origin_i, x_i, y_i = (name_to_idx[a] for a in self.anchor_labels)
        R, o = rigid_frames(markers_mm, origin_i, x_i, y_i)
        local = to_local(markers_mm, R, o)
        if anchor_valid is not None:
            local = mask_untrustworthy_frames(local, anchor_valid)
        return relabel_local(local, self.gmms, self.marker_idxs,
                             n_hypotheses=n_hypotheses, theta_min=theta_min,
                             process_var=process_var, obs_var=obs_var,
                             init_frame=init_frame)
