"""Infer soft correspondences between Vicon markers and MANO joints.

Strategy
--------
We do NOT assume a fixed labelling.  Instead we use an iterative closest-point
(ICP) inspired assignment:

1. Normalise both point sets (zero-mean, unit scale).
2. Solve the linear assignment problem (Hungarian) on pairwise distances.
3. Discard spurious assignments whose distance exceeds a z-score threshold.

For a sequence we can either:
  - use a single assignment computed from the mean pose (fast), or
  - re-solve per frame (robust to missing markers / label swaps).

The module also provides a heuristic seed based on anatomical ordering that
works when Vicon labels follow a consistent naming convention
(e.g. RWRB, RIDX1 ... RIDX4, RMID1 ... etc.).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment


# MANO joint order (21 joints): wrist + 4 joints per finger (thumb→pinky)
MANO_JOINT_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp",   "ring_pip",   "ring_dip",   "ring_tip",
    "pinky_mcp",  "pinky_pip",  "pinky_dip",  "pinky_tip",
]  # 21 entries

# Keyword fragments that hint at each MANO joint — used for label-based seeding.
#
# Primary naming (from marker_map.json in clean_kinematics):
#   Palm2 → wrist proxy, Index1-3, Middle1-3, Pinky1-3, Ring1-3, Thumb1-3
# Secondary naming (Nexus Rx prefixed, e.g. RIDX1):
#   RWRB/WRA/WRI, RTHB1-4, RIDX1-4, RMID1-4, RRNG1-4, RLIT1-4
#
# MANO has 16 internal joints + 5 fingertip joints = 21 total.
# The Vicon setup covers the 16 internal joints; fingertips are unobserved.
_LABEL_HINTS: dict[str, list[str]] = {
    "wrist":      ["palm2", "wrist", "wrb", "wra", "wri"],
    "thumb_mcp":  ["thumb1", "thb1", "thmcp"],
    "thumb_pip":  ["thumb2", "thb2", "thpip"],
    "thumb_dip":  ["thumb3", "thb3", "thdip"],
    "thumb_tip":  ["thumb4", "thb4", "thtip"],
    "index_mcp":  ["index1", "idx1", "ind1", "indmcp"],
    "index_pip":  ["index2", "idx2", "ind2", "indpip"],
    "index_dip":  ["index3", "idx3", "ind3", "inddip"],
    "index_tip":  ["index4", "idx4", "ind4", "indtip"],
    "middle_mcp": ["middle1", "mid1", "midmcp"],
    "middle_pip": ["middle2", "mid2", "midpip"],
    "middle_dip": ["middle3", "mid3", "middip"],
    "middle_tip": ["middle4", "mid4", "midtip"],
    "ring_mcp":   ["ring1",   "rng1", "rngmcp"],
    "ring_pip":   ["ring2",   "rng2", "rngpip"],
    "ring_dip":   ["ring3",   "rng3", "rngdip"],
    "ring_tip":   ["ring4",   "rng4", "rngtip"],
    "pinky_mcp":  ["pinky1",  "pnk1", "lit1",  "litmcp"],
    "pinky_pip":  ["pinky2",  "pnk2", "lit2",  "litpip"],
    "pinky_dip":  ["pinky3",  "pnk3", "lit3",  "litdip"],
    "pinky_tip":  ["pinky4",  "pnk4", "lit4",  "littip"],
}


def _normalise_label(label: str) -> str:
    """Canonicalise a Vicon label for hint matching.

    Lowercases, drops side designators, and strips non-alphanumerics so
    that e.g. "Index_Left1" → "index1" and "RIDX2" → "ridx2".
    """
    s = label.lower().replace("left", "").replace("right", "")
    return re.sub(r"[^a-z0-9]", "", s)


def label_seed(
    marker_labels: list[str],
    *,
    side: str | None = None,
) -> dict[str, int] | None:
    """Try to seed correspondences from marker label strings.

    Args:
        marker_labels: raw Vicon label strings.
        side:          "left" or "right".  When given, labels naming the
                       opposite side (e.g. "Index_Right1" while fitting the
                       left hand) are excluded — needed for two-hand exports.

    Returns a dict {joint_name: marker_index} for each joint that could be
    matched, or None if fewer than 10 joints were resolved (too uncertain).
    """
    opposite = {"left": "right", "right": "left"}.get(side or "")
    labels_norm: list[str | None] = []
    for l in marker_labels:
        if opposite and opposite in l.lower():
            labels_norm.append(None)
        else:
            labels_norm.append(_normalise_label(l))

    assignment: dict[str, int] = {}
    for joint, hints in _LABEL_HINTS.items():
        for idx, lbl in enumerate(labels_norm):
            if lbl is not None and any(h in lbl for h in hints):
                assignment[joint] = idx
                break
    return assignment if len(assignment) >= 10 else None


def hungarian_assignment(
    markers: np.ndarray,       # (N, 3)
    mano_joints: np.ndarray,   # (J, 3)  J ≤ 21
    *,
    z_thresh: float = 2.5,
) -> np.ndarray:
    """Return index array `assign` of length J.

    assign[j] = index into markers (or -1 if no confident match).

    Markers with non-finite coordinates (Vicon gaps) are excluded; returned
    indices always refer to the original markers array.
    """
    assign = np.full(len(mano_joints), -1, dtype=int)

    finite = np.isfinite(markers).all(axis=1)
    orig_idx = np.flatnonzero(finite)
    if len(orig_idx) == 0:
        return assign
    markers = markers[finite]

    # Normalise both to zero-mean unit-variance
    m = markers - markers.mean(0)
    m /= np.linalg.norm(m) + 1e-8
    j = mano_joints - mano_joints.mean(0)
    j /= np.linalg.norm(j) + 1e-8

    cost = np.linalg.norm(m[None, :, :] - j[:, None, :], axis=-1)  # (J, N)
    row_ind, col_ind = linear_sum_assignment(cost)

    dists = cost[row_ind, col_ind]
    # Add a small absolute floor so that near-zero matched distances
    # (e.g. when markers == joints up to floating-point noise) are not
    # spuriously rejected by a near-zero threshold.
    thresh = dists.mean() + z_thresh * dists.std() + 1e-6
    for r, c, d in zip(row_ind, col_ind, dists):
        if d <= thresh:
            assign[r] = orig_idx[c]
    return assign


def sequence_assignment(
    markers_seq: np.ndarray,   # (T, N, 3)
    mano_joints_seq: np.ndarray,  # (T, J, 3)
    *,
    per_frame: bool = False,
) -> np.ndarray:
    """Compute marker↔joint assignment for a whole sequence.

    If per_frame=False (default), uses the temporal mean (fast and stable
    unless there are frequent label swaps in the Vicon data).
    Returns assign array of shape (J,) or (T, J).
    """
    if per_frame:
        T = markers_seq.shape[0]
        assigns = np.stack([
            hungarian_assignment(markers_seq[t], mano_joints_seq[t])
            for t in range(T)
        ])
        return assigns
    # nanmean tolerates per-frame Vicon gaps; markers missing in every frame
    # come out all-NaN and are dropped inside hungarian_assignment.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        markers_mean = np.nanmean(markers_seq, axis=0)
    return hungarian_assignment(markers_mean, mano_joints_seq.mean(0))


# ---------------------------------------------------------------------------
# Unified assignment entry point
# ---------------------------------------------------------------------------

def get_assignment(
    markers_seq: np.ndarray,
    mano_joints_seq: np.ndarray,
    *,
    labeler=None,
    per_frame: bool = False,
    marker_labels: list[str] | None = None,
    side: str | None = None,
) -> np.ndarray:
    """Return a marker↔joint assignment, trying strategies in priority order.

    Priority:
    1. Label-seed heuristic (if *marker_labels* are provided and ≥15 match).
       Trusted Vicon labels beat any learned predictor.
    2. DeepLabeler (if *labeler* is provided and succeeds).
    3. Hungarian assignment on the mean pose (fallback).

    Args:
        markers_seq:     (T, N, 3) marker positions in metres.
        mano_joints_seq: (T, 21, 3) MANO joint positions in metres (used only
                         for the Hungarian fallback).
        labeler:         Optional DeepLabeler instance.  Pass None to skip.
        per_frame:       When True, return (T, 21) per-frame assignments.
                         When False (default), return a single (21,)
                         assignment (majority vote for the labeler path).
        marker_labels:   Optional list of N label strings (Vicon naming).
                         Used for the label-seed path.
        side:            "left"/"right" — excludes opposite-side labels when
                         the export contains both hands.

    Returns:
        (21,) int array (or (T, 21) when per_frame=True).
        Unmatched joints have value -1.
    """
    # ------------------------------------------------------------------
    # Path 1: label-seed heuristic — explicit labels are the most reliable
    # ------------------------------------------------------------------
    if marker_labels is not None:
        seed = label_seed(marker_labels, side=side)
        if seed is not None and len(seed) >= 15:
            assign = np.full(len(MANO_JOINT_NAMES), -1, dtype=int)
            for j_idx, jname in enumerate(MANO_JOINT_NAMES):
                if jname in seed:
                    assign[j_idx] = seed[jname]
            if per_frame:
                return np.tile(assign[None, :], (markers_seq.shape[0], 1))
            return assign

    # ------------------------------------------------------------------
    # Path 2: deep labeler
    # ------------------------------------------------------------------
    if labeler is not None:
        try:
            assign_seq = labeler.label_sequence(markers_seq)  # (T, 21)
            if per_frame:
                return assign_seq
            return _majority_vote(assign_seq, n_markers=markers_seq.shape[1])
        except Exception as exc:
            warnings.warn(
                f"DeepLabeler failed ({exc}); falling back to Hungarian.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Path 3: Hungarian on mean pose
    # ------------------------------------------------------------------
    return sequence_assignment(markers_seq, mano_joints_seq, per_frame=per_frame)


def _distance_descriptor(
    markers: np.ndarray,   # (T, N, 3)
    *,
    stride: int = 20,
    normalize: bool = True,
) -> np.ndarray:
    """Rigid-motion-invariant per-marker signature.

    For each marker, the sorted vector of its mean distances to every other
    marker (averaged over frames). Invariant to global rotation/translation;
    with ``normalize`` the whole descriptor is divided by its mean so it is
    also scale-invariant (matches hands of different size). NaN-safe: each
    pairwise mean ignores frames where either marker is missing.

    Returns (N, N) — row i is marker i's sorted distance signature.
    """
    sample = markers[::stride]                       # (F, N, 3)
    d = np.linalg.norm(sample[:, :, None, :] - sample[:, None, :, :], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_d = np.nanmean(d, axis=0)               # (N, N); NaN if never co-present
    mean_d = np.nan_to_num(mean_d, nan=0.0)
    sig = np.sort(mean_d, axis=1)
    if normalize and sig.mean() > 0:
        sig = sig / sig.mean()
    return sig


def relabel_by_template(
    markers: np.ndarray,            # (T, N, 3) unlabeled, complete cloud
    template_markers: np.ndarray,   # (Tt, M, 3) labeled reference
    template_labels: list[str],     # length M
    *,
    stride: int = 20,
    normalize: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Assign template labels to an unlabeled marker cloud by geometry only.

    Matches each of the N input markers to one of the M template markers via
    a rigid-/scale-invariant distance descriptor and the Hungarian algorithm,
    then copies that template marker's label across. Marker *positions are
    never modified* — this only names columns.

    Works when the input cloud is complete and shares the template's marker
    protocol (same physical marker set), even across different poses, hand
    sizes, and column orderings.

    Limitation — chirality: pairwise-distance descriptors are reflection-
    invariant, so a left and right hand (near mirror images) have nearly
    identical signatures. On a two-hand cloud the matcher therefore swaps
    markers between hands (validated: 22/22 single-hand, 16/44 two-hand).
    Separate the cloud into per-hand clusters and relabel each against a
    single-hand template; resolve which cluster is which from its spatial
    side. ``relabel_two_hands`` does this.

    Returns:
        new_labels: length-N list; input marker i gets ``new_labels[i]``.
                    If N > M, surplus markers get ``"Unlabeled_<i>"``.
        cost:       length-N array of the match cost per marker (lower = more
                    confident; large values flag ambiguous/foreign markers).
    """
    sig_in = _distance_descriptor(markers, stride=stride, normalize=normalize)
    sig_tm = _distance_descriptor(template_markers, stride=stride,
                                  normalize=normalize)
    N, M = sig_in.shape[0], sig_tm.shape[0]
    # descriptors must share length to compare; pad the shorter row dim
    width = max(N, M)
    a = np.zeros((N, width)); a[:, :N] = sig_in
    b = np.zeros((M, width)); b[:, :M] = sig_tm
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)   # (N, M)

    row, col = linear_sum_assignment(cost)           # match min(N, M) markers
    new_labels = [f"Unlabeled_{i}" for i in range(N)]
    out_cost = np.full(N, np.inf)
    for i, j in zip(row, col):
        new_labels[i] = template_labels[j]
        out_cost[i] = cost[i, j]
    return new_labels, out_cost


def _cluster_two(points: np.ndarray, n_iter: int = 25) -> np.ndarray:
    """Dependency-free 2-means on (N, 3) points → boolean mask (cluster 0/1).

    Seeded with the two most-distant points, so the two spatially-separated
    hands fall into different clusters."""
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    c = np.stack([points[i], points[j]])
    mask = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        dist = np.linalg.norm(points[:, None] - c[None], axis=-1)  # (N, 2)
        new = dist.argmin(1)
        if np.array_equal(new, mask):
            break
        mask = new
        for k in (0, 1):
            if (mask == k).any():
                c[k] = points[mask == k].mean(0)
    return mask.astype(bool)


def _hand_side(label: str) -> str | None:
    l = label.lower()
    return "left" if "left" in l else ("right" if "right" in l else None)


def relabel_two_hands(
    markers: np.ndarray,            # (T, N, 3) unlabeled two-hand cloud
    template_markers: np.ndarray,   # (Tt, M, 3) labeled two-hand reference
    template_labels: list[str],     # length M, with left/right designators
    *,
    stride: int = 20,
) -> tuple[list[str], np.ndarray]:
    """Relabel a complete two-hand cloud, chirality-safe.

    Splits both clouds into two hands (template by its label sides; input by
    spatial 2-means), matches each input cluster to the template hand it best
    fits by descriptor cost, then relabels each hand with the single-hand
    :func:`relabel_by_template` (where the distance descriptor is unambiguous).
    Positions are never modified.

    Returns ``(new_labels, cost)`` like :func:`relabel_by_template`.
    """
    # split template by its labels
    sides = {s: [i for i, l in enumerate(template_labels) if _hand_side(l) == s]
             for s in ("left", "right")}
    if not (sides["left"] and sides["right"]):
        # not actually two-handed → fall back to the plain matcher
        return relabel_by_template(markers, template_markers, template_labels,
                                   stride=stride)

    mean_pos = np.nanmean(markers, axis=0)               # (N, 3)
    cl = _cluster_two(np.nan_to_num(mean_pos))
    clusters = {0: np.flatnonzero(~cl), 1: np.flatnonzero(cl)}

    def side_match(idx, side_idx):
        return relabel_by_template(
            markers[:, idx], template_markers[:, side_idx],
            [template_labels[i] for i in side_idx], stride=stride)

    # try both cluster→side pairings, keep the lower total cost
    best = None
    for assign in ({0: "left", 1: "right"}, {0: "right", 1: "left"}):
        total, parts = 0.0, {}
        for cidx, side in assign.items():
            lbls, c = side_match(clusters[cidx], sides[side])
            parts[cidx] = (lbls, c)
            total += np.nansum(c[np.isfinite(c)])
        if best is None or total < best[0]:
            best = (total, parts)

    N = markers.shape[1]
    new_labels = [f"Unlabeled_{i}" for i in range(N)]
    cost = np.full(N, np.inf)
    for cidx, (lbls, c) in best[1].items():
        for local, orig in enumerate(clusters[cidx]):
            new_labels[orig] = lbls[local]
            cost[orig] = c[local]
    return new_labels, cost


def _majority_vote(assign_seq: np.ndarray, n_markers: int) -> np.ndarray:
    """Collapse (T, 21) per-frame assignments to a single (21,) assignment.

    For each joint, picks the most-frequent non-(-1) marker index across all
    frames.  If every frame gave -1 for a joint, that joint stays -1.
    """
    T, J = assign_seq.shape
    result = np.full(J, -1, dtype=int)
    for j in range(J):
        col = assign_seq[:, j]
        valid = col[col >= 0]
        if len(valid) == 0:
            continue
        # Guard against index >= n_markers
        valid = valid[valid < n_markers]
        if len(valid) == 0:
            continue
        result[j] = int(np.bincount(valid, minlength=n_markers).argmax())
    return result
