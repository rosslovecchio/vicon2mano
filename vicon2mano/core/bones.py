"""Bone-length self-consistency: a model-free way to score whether a
frame's marker labelling is geometrically plausible.

Moved here from ``scripts/gmm/relabel_trial.py`` (where it was first
written for strategyGMM's training-frame bootstrap) because a second,
independent consumer now needs the same reference-length/deviation
machinery: ``scripts/shared/geometric_self_consistency.py``, which scores
*every* trial's raw labelling this way, with no GMM, no manual export, and
no cascade involved. Logic is unchanged from the original -- see
``docs/strategies/gmm.md`` and CLAUDE.md's 2026-09-11/12 session notes for
how ``consensus_bone_lengths`` was arrived at and validated.

Bone topology (*which* marker pairs count as a bone) is not part of this
module -- pass in whatever list of ``(marker_i, marker_j)`` index pairs
suits the caller. :func:`vicon2mano.strategies.cascade.quality_cascade.infer_bones`
derives one such list from marker label strings (finger chains + palm/
forearm rigid-plate pairs) and is the topology
``geometric_self_consistency.py`` uses; strategyGMM's own
``scripts/gmm/relabel_trial.py`` keeps its separate, narrower
``_finger_chains`` topology unchanged, since the two were tuned for
different purposes and swapping one for the other was not asked for.
"""

from __future__ import annotations

import warnings

import numpy as np


def modal_bone_lengths(markers: np.ndarray, bones: list[tuple[int, int]],
                        *, bin_mm: float = 0.5, refine_mm: float = 2.0) -> np.ndarray:
    """Reference length per bone from each bone's *marginal* mode.

    Works only where a bone is correctly labelled in most frames. Kept for
    comparison and diagnostics; :func:`consensus_bone_lengths` is what real
    pipelines use, because that assumption fails badly on real data (see
    its docstring).
    """
    out = np.full(len(bones), np.nan)
    for k, (i, j) in enumerate(bones):
        v = np.linalg.norm(markers[:, i] - markers[:, j], axis=1)
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        hist, edges = np.histogram(v, bins=np.arange(0, v.max() + bin_mm, bin_mm))
        mode = edges[int(np.argmax(hist))] + bin_mm / 2
        near = v[np.abs(v - mode) < refine_mm]
        out[k] = near.mean() if near.size else mode
    return out


def consensus_bone_lengths(
    markers: np.ndarray, bones: list[tuple[int, int]], *,
    tol_mm: float = 4.0, n_hypotheses: int = 400, n_compare: int = 3000,
    seed: int = 0, min_bones: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference lengths from the largest mutually-consistent set of frames.

    RANSAC over frames: every candidate frame's whole bone-length vector is
    one hypothesis, scored by how many other frames agree with it on *every*
    bone at once; the winner's inliers are then averaged.

    Why not the per-bone mode (:func:`modal_bone_lengths`): that assumes each
    bone is labelled correctly in most frames, and on real data it is not.
    On P7/Trial1_handsonly, `Ring1` is mislabelled in the *majority* of
    frames, so its marginal mode locks onto the wrong configuration --
    Palm2-Ring1 reads 34mm, which is shorter than Palm2-Pinky1 (54mm) and
    anatomically impossible. Consensus fixes it (50mm) because wrong
    configurations do not agree with *each other*: each different swap
    produces a different distance vector, so only correctly-labelled frames
    pile up into one large mutually-consistent set.

    Returns ``(ref_lengths, inlier_frames)``. The inliers are frames whose
    every bone matches the reference -- i.e. frames that are geometrically
    self-consistent, and so a natural training set for a model-based
    relabeller, with no external quality verdict involved.
    """
    if len(bones) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    i_idx = np.array([b[0] for b in bones])
    j_idx = np.array([b[1] for b in bones])
    D = np.linalg.norm(markers[:, i_idx] - markers[:, j_idx], axis=2)   # (T, B)
    finite = np.isfinite(D)
    usable = np.flatnonzero(finite.sum(axis=1) >= min_bones)
    if usable.size == 0:
        return np.full(len(bones), np.nan), usable

    # A hypothesis is drawn from the same `min_bones`-or-more pool as the
    # comparison set (`usable`), not "every bone must be finite" -- a
    # marker missing for an entire trial (real data: P1's Forearm3/4 and
    # Palm3 are permanently absent) makes every bone touching it NaN in
    # *every* frame, so requiring literal 100% bone coverage to even
    # nominate a hypothesis previously meant NO hypothesis ever qualified,
    # and every OTHER marker's reference silently came back all-NaN too --
    # confirmed on P1: 19/22 markers had 87-100% availability, yet
    # `consensus_bone_lengths` returned 0 inlier frames trial-wide. Scoring
    # in `agrees` below already handles a hypothesis missing some bones
    # correctly (a bond neither the hypothesis nor a comparison frame has
    # is bypassed via `~ok` on both sides; a bond the hypothesis lacks but
    # a comparison frame DOES have simply fails to match, same as any other
    # disagreement) -- so a fully-populated hypothesis still wins outright
    # whenever one exists, and this only changes the outcome when none do.
    rng = np.random.default_rng(seed)
    hyp = usable if usable.size <= n_hypotheses else usable[
        rng.choice(usable.size, size=n_hypotheses, replace=False)]
    comp = usable if usable.size <= n_compare else usable[
        np.linspace(0, usable.size - 1, n_compare).astype(int)]

    def agrees(frames: np.ndarray, h: int) -> np.ndarray:
        dev = np.abs(D[frames] - D[h])
        ok = finite[frames]
        return ((dev <= tol_mm) | ~ok).all(axis=1) & (ok.sum(axis=1) >= min_bones)

    best_score, best_h = -1, int(hyp[0])
    for h in hyp:
        score = int(agrees(comp, h).sum())
        if score > best_score:
            best_score, best_h = score, int(h)

    inliers = usable[agrees(usable, best_h)]
    if inliers.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ref = np.nanmean(np.where(finite[inliers], D[inliers], np.nan), axis=0)
        ref = np.where(np.isfinite(ref), ref, D[best_h])
    else:
        ref = D[best_h]
    return ref, inliers


def bone_error(markers: np.ndarray, bones: list[tuple[int, int]],
               ref: np.ndarray) -> np.ndarray:
    """(T, n_bones) absolute deviation from each bone's reference length."""
    err = np.full((markers.shape[0], len(bones)), np.nan)
    for k, (i, j) in enumerate(bones):
        if not np.isfinite(ref[k]):
            continue
        err[:, k] = np.abs(np.linalg.norm(markers[:, i] - markers[:, j], axis=1) - ref[k])
    return err
