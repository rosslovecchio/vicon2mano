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

# Keyword fragments that hint at each MANO joint — used for label-based seeding
_LABEL_HINTS: dict[str, list[str]] = {
    "wrist":      ["wrist", "wrb", "wra", "wri"],
    "thumb_mcp":  ["thb1", "thumb1", "thmcp"],
    "thumb_pip":  ["thb2", "thumb2", "thpip"],
    "thumb_dip":  ["thb3", "thumb3", "thdip"],
    "thumb_tip":  ["thb4", "thumb4", "thtip", "thtop"],
    "index_mcp":  ["idx1", "index1", "indmcp", "ind1"],
    "index_pip":  ["idx2", "index2", "indpip", "ind2"],
    "index_dip":  ["idx3", "index3", "inddip", "ind3"],
    "index_tip":  ["idx4", "index4", "indtip", "ind4"],
    "middle_mcp": ["mid1", "middle1", "midmcp"],
    "middle_pip": ["mid2", "middle2", "midpip"],
    "middle_dip": ["mid3", "middle3", "middip"],
    "middle_tip": ["mid4", "middle4", "midtip"],
    "ring_mcp":   ["rng1", "ring1",   "rngmcp"],
    "ring_pip":   ["rng2", "ring2",   "rngpip"],
    "ring_dip":   ["rng3", "ring3",   "rngdip"],
    "ring_tip":   ["rng4", "ring4",   "rngtip"],
    "pinky_mcp":  ["pnk1", "pinky1",  "litmcp", "lit1"],
    "pinky_pip":  ["pnk2", "pinky2",  "litpip", "lit2"],
    "pinky_dip":  ["pnk3", "pinky3",  "litdip", "lit3"],
    "pinky_tip":  ["pnk4", "pinky4",  "littip", "lit4"],
}


def label_seed(marker_labels: list[str]) -> dict[str, int] | None:
    """Try to seed correspondences from marker label strings.

    Returns a dict {joint_name: marker_index} for each joint that could be
    matched, or None if fewer than 10 joints were resolved (too uncertain).
    """
    labels_lower = [l.lower() for l in marker_labels]
    assignment: dict[str, int] = {}
    for joint, hints in _LABEL_HINTS.items():
        for idx, lbl in enumerate(labels_lower):
            if any(h in lbl for h in hints):
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
    """
    # Normalise both to zero-mean unit-variance
    m = markers - markers.mean(0)
    m /= np.linalg.norm(m) + 1e-8
    j = mano_joints - mano_joints.mean(0)
    j /= np.linalg.norm(j) + 1e-8

    cost = np.linalg.norm(m[None, :, :] - j[:, None, :], axis=-1)  # (J, N)
    row_ind, col_ind = linear_sum_assignment(cost)

    assign = np.full(len(mano_joints), -1, dtype=int)
    dists = cost[row_ind, col_ind]
    thresh = dists.mean() + z_thresh * dists.std()
    for r, c, d in zip(row_ind, col_ind, dists):
        if d <= thresh:
            assign[r] = c
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
    return hungarian_assignment(markers_seq.mean(0), mano_joints_seq.mean(0))
