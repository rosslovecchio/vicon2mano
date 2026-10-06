"""Raw Vicon quality assessment: classify every marker-frame using six
separately-measured signals, without combining them into one score.

**assessment_type = "raw_vicon_quality"**

This is a *fourth*, separate assessment from the other three in this repo
(``core.agreement`` = P10 label retention, ``core.geometric_consistency``
= geometric self-consistency, ``core.validation`` = the 500-frame manual
identity validation). It draws on data from several of them, but the
six measurements below are reported as their own columns, never averaged
or weighted into a single number:

1. manual agreement with reviewed labels (``core.validation``'s
   ``manual_annotations.csv`` -- only ~500 frames have this)
2. geometric self-consistency (``core.geometric_consistency``'s
   ``events.csv`` -- whether a frame falls inside a flagged
   bone-length-deviation event)
3. anchor-frame validity (``core.validation.to_palm_local_frame``'s
   anchors -- present and non-degenerate)
4. trajectory continuity (``core.continuity.flag_discontinuities``)
5. suspicious swap events -- same source as (2), kept as its own column
   since a caller may want "inside a flagged event" without conflating it
   with the geometric-plausibility label
6. ambiguous cases (``core.validation``'s automatic anchor-occlusion
   ``ambiguous`` status, where reviewed)

:func:`classify_marker_frame` turns these into exactly one of
:data:`CATEGORIES` per marker-frame, via an explicit, documented
precedence order -- a categorical label for convenience, not a score:
nothing is weighted or averaged, and every input signal that produced the
label is still available in its own column in the output table for anyone
who wants to recompute a different precedence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASSESSMENT_TYPE = "raw_vicon_quality"

# "ghost_candidate", "anchor_invalid" etc. are the categories named in the
# request; "label_manually_wrong" is added here because the given list had
# no slot for a human-confirmed *wrong* label that isn't also a ghost --
# dropping that distinction would lose real information the manual review
# produced. Flagged explicitly rather than silently folded into
# "ghost_candidate" or "observed_suspicious".
CATEGORIES = (
    "label_manually_verified",
    "label_manually_wrong",
    "ghost_candidate",
    "ambiguous",
    "missing",
    "anchor_invalid",
    "observed_suspicious",
    "observed_plausible",
)

_MANUAL_STATUS_TO_CATEGORY = {
    "correct": "label_manually_verified",
    "wrong_label": "label_manually_wrong",
    "ghost": "ghost_candidate",
    "ambiguous": "ambiguous",
    "missing": "missing",
}


def classify_marker_frame(
    is_missing: bool,
    manual_status: str | None,
    anchor_valid: bool,
    geometric_suspicious: bool,
) -> str:
    """One marker-frame's category, by precedence:

    1. A human review verdict, if this exact marker-frame was reviewed,
       always wins -- it is the one signal with real independent ground
       truth, so an automated signal never overrides it.
    2. Otherwise, a raw-missing coordinate is reported as ``missing``
       regardless of any other signal (there is nothing to assess).
    3. Otherwise, an invalid/degenerate anchor frame is reported as
       ``anchor_invalid`` -- this comes before the geometric check because
       a geometric "plausible" verdict computed in an undefined or
       numerically unstable local frame is not trustworthy (see
       ``core.validation.to_palm_local_frame``'s docstring).
    4. Otherwise, a frame inside a flagged geometric self-consistency event
       is ``observed_suspicious``.
    5. Otherwise, ``observed_plausible`` -- not proof of a correct label
       (see every other module's "geometry does not prove label
       correctness" caveat), only "nothing here contradicted it".
    """
    if manual_status is not None:
        return _MANUAL_STATUS_TO_CATEGORY[manual_status]
    if is_missing:
        return "missing"
    if not anchor_valid:
        return "anchor_invalid"
    if geometric_suspicious:
        return "observed_suspicious"
    return "observed_plausible"


def classify_vectorized(
    is_missing: np.ndarray,
    anchor_valid: np.ndarray,
    geometric_suspicious: np.ndarray,
    manual_category: np.ndarray,
) -> np.ndarray:
    """Vectorised form of :func:`classify_marker_frame` for a whole (T, N)
    array at once -- used by the production pipeline, where a Python-level
    loop over tens of millions of marker-frames is not practical.
    ``manual_category`` already has each reviewed cell's resolved category
    string (or "" where unreviewed) -- see ``_MANUAL_STATUS_TO_CATEGORY``,
    applied by the caller before this function so the precedence logic
    here stays identical between the scalar and vectorised paths.
    """
    out = np.full(is_missing.shape, "observed_plausible", dtype=object)
    out[geometric_suspicious] = "observed_suspicious"
    out[~anchor_valid] = "anchor_invalid"
    out[is_missing] = "missing"
    has_manual = manual_category != ""
    out[has_manual] = manual_category[has_manual]
    return out


def resolve_manual_category(status: str | None) -> str:
    """Empty string (not None) for "unreviewed", matching
    :func:`classify_vectorized`'s sentinel convention for a pandas/numpy
    string array, which has no clean native null for object-dtype string
    comparisons at this scale."""
    if status is None:
        return ""
    return _MANUAL_STATUS_TO_CATEGORY[status]


def summarize_counts(classification: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, trial_id, marker), with a count column per
    category in :data:`CATEGORIES` -- the aggregate
    ``raw_vicon_quality_summary.csv`` this module produces, since the full
    per-marker-frame table is tens of millions of rows dataset-wide and is
    kept per-trial instead (see ``scripts/shared/build_raw_vicon_quality_summary.py``).
    """
    rows = []
    for (p, t, m), g in classification.groupby(["participant", "trial_id", "marker"]):
        row = {"participant": p, "trial_id": t, "marker": m, "n_frames": len(g)}
        counts = g["category"].value_counts()
        for cat in CATEGORIES:
            row[cat] = int(counts.get(cat, 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["participant", "trial_id", "marker"]).reset_index(drop=True)


def adaptive_anchor_threshold(sine_values: np.ndarray, *, k: float = 3.0,
                               absolute_floor: float = 0.1) -> float:
    """Lower bound on sine(Palm1-2-3 angle) for "valid", learned from one
    subject's own pooled distribution rather than assumed universal.

    Why this exists: the palm plate is not rigid (see CLAUDE.md -- "palm
    markers are taped to skin"), so there is no single geometrically
    correct angle to check against; each subject's own marker placement
    gives them their own typical angle. Measured directly across this
    dataset: per-subject median sine is 0.4-0.6, with a fixed
    ``absolute_floor=0.1`` essentially never triggering for anyone (it is
    far below every subject's normal range) -- so a fixed threshold was
    silently only ever catching "anchor missing", never "anchor
    geometrically implausible for this subject".

    One-sided: only a *narrower*-than-usual triangle is penalised. A wider
    one is not degenerate -- if anything it is numerically *more* stable --
    so there is no reason to flag it.

    ``k`` defaults to 3.0, tighter than this repo's usual ``k=6`` (used
    where the concern is suppressing false positives on a mostly-zero
    signal, e.g. ``core.continuity``, ``core.agreement.flag_temporal_jumps``).
    Here the goal is the opposite: real sensitivity to a subject whose
    anchor geometry drifts, so a looser k would under-detect. This is a
    median+MAD estimate, not a consensus/RANSAC one (contrast
    ``core.bones.consensus_bone_lengths``, which exists precisely because a
    median/mode can land on a contaminated value when a large fraction of
    frames are themselves bad) -- a subject whose own palm labelling is
    badly and pervasively wrong could inflate both the median and the MAD
    enough to mask the problem. The ``absolute_floor`` is the backstop for
    that case: the returned threshold never drops below it, so a subject's
    own statistics can only make the check *stricter*, never *looser*, than
    the universal minimum.
    """
    finite = sine_values[np.isfinite(sine_values)]
    if finite.size == 0:
        return absolute_floor
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    return max(absolute_floor, med - k * mad)
