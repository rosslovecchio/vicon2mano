"""Trajectory continuity: does a marker move smoothly frame-to-frame, or
does it jump -- a signal distinct from, and never combined with, geometric
self-consistency (bone-length plausibility) or anchor-frame validity.
A marker can have a perfectly plausible bone length while still having
teleported there from an implausible speed, and vice versa; keeping this
its own measurement is why raw_vicon_quality_summary.csv reports it as its
own column rather than folding it into either of the others.
"""

from __future__ import annotations

import numpy as np


def flag_discontinuities(markers: np.ndarray, *, k: float = 6.0,
                          min_jump_mm: float = 25.0) -> np.ndarray:
    """(T, N) bool: frame t is flagged for marker n if the frame-to-frame
    displacement arriving at t is an outlier for that marker.

    Per marker: ``threshold = max(median(speed) + k * MAD(speed), min_jump_mm)``.
    Same zero-inflation guard used throughout this repo
    (``vicon2mano.core.agreement.flag_temporal_jumps``,
    ``vicon2mano.core.geometric_consistency.flag_suspicious_events``): a
    marker that is mostly still has a median/MAD speed near 0, so without
    the floor almost any motion at all would be flagged. ``min_jump_mm``
    defaults to 25mm, matching this repo's existing
    ``quality_cascade.py --speed-tol-mm`` default, for the same reason that
    value was chosen there (comfortably above ordinary frame-to-frame
    motion, well below a teleport/mislabel-sized jump).

    Frame 0 is never flagged for any marker (no prior frame to compare
    against). A gap (either side non-finite) contributes no displacement
    and is never itself flagged as a discontinuity -- that is
    ``missing``'s job, not this module's.
    """
    T, N, _ = markers.shape
    out = np.zeros((T, N), dtype=bool)
    disp = np.linalg.norm(np.diff(markers, axis=0), axis=2)  # (T-1, N)
    for n in range(N):
        d = disp[:, n]
        finite = np.isfinite(d)
        if finite.sum() < 2:
            continue
        med = np.median(d[finite])
        mad = np.median(np.abs(d[finite] - med)) * 1.4826
        thresh = max(med + k * mad, min_jump_mm)
        flag = finite & (d > thresh)
        out[1:, n] = flag
    return out
