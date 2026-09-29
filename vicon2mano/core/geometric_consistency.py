"""Geometric self-consistency: does a trial's raw Vicon labelling produce
anatomically plausible bone lengths, checked against nothing but its own
geometry.

**assessment_type = "geometric_self_consistency"**

This is a *different* thing from the Vicon-vs-manual agreement analysis in
``vicon2mano.core.agreement`` (see that module and
``scripts/shared/analyze_manual_agreement.py``), and the two are never
pooled:

- ``agreement`` compares two independent *labellings* of the same
  recording (Vicon's on-disk labels vs. a human-reviewed export) and can
  therefore speak to label *identity* -- was this marker's column the one
  a human would also have called "Index2".
- This module compares a *single* labelling against a reference derived
  from its own bone lengths (:func:`vicon2mano.core.bones.consensus_bone_lengths`).
  It can flag a frame as **anatomically implausible** (a bone reads far
  from its trial's own consensus length), but it has no independent
  identity check at all -- two markers swapped in a way that happens to
  preserve every bone length involved (e.g. two markers on a rigid plate,
  symmetric under exchange) would pass through this check undetected. It
  measures plausibility, not correctness, and never claims otherwise.

Bone topology comes from
:func:`vicon2mano.strategies.cascade.quality_cascade.infer_bones` (finger
chains + rigid palm/forearm plate pairs, derived from marker label
strings); the reference-length/deviation machinery is
:mod:`vicon2mano.core.bones`, moved there from
``scripts/gmm/relabel_trial.py`` specifically so this module could reuse it
instead of re-deriving its own. Nothing in this module was copied from
``vicon2mano.core.agreement`` even though several functions here mirror its
shape (per-marker table, threshold columns, event flagging/grouping,
missingness gaps) -- that similarity is deliberate (the same diagnostic
shape is useful for both questions), but every function, column name, and
the ``assessment_type`` marker are kept independent on purpose, so the two
kinds of report can never be silently merged or confused downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vicon2mano.core.bones import bone_error, consensus_bone_lengths

ASSESSMENT_TYPE = "geometric_self_consistency"


def marker_bone_map(n_markers: int, bones: list[tuple[int, int]]) -> dict[int, list[int]]:
    """marker index -> list of bone indices (into ``bones``) touching it."""
    out: dict[int, list[int]] = {i: [] for i in range(n_markers)}
    for k, (i, j) in enumerate(bones):
        out[i].append(k)
        out[j].append(k)
    return out


def build_marker_frame_table(
    markers: np.ndarray,
    labels: list[str],
    bones: list[tuple[int, int, str]],
    ref_lengths: np.ndarray,
    *,
    subject_id: str,
    trial_id: str,
) -> pd.DataFrame:
    """Long-format per-frame, per-marker geometric-deviation table.

    A marker's ``deviation_mm`` at a frame is the *worst* (max) absolute
    bone-length deviation among the bones touching it that frame (see
    :func:`marker_bone_map`) -- a marker is implicated in an implausible
    frame if *any* of its bones is off, not only if all of them are.
    ``associated_bone`` names whichever bone produced that max, for
    attributing a later flagged event to a specific marker pair.

    A marker with no bones at all (unlisted by
    ``quality_cascade.infer_bones`` -- shouldn't happen for a real hand
    marker, but not assumed) gets NaN deviation every frame and
    ``associated_bone`` of ``None``, not an error: this is a coverage gap
    in the bone topology, not a data problem, and the caller should not
    have that raise unrelated to the trial actually being analysed.

    ``vicon_available`` records whether the marker itself has a finite
    coordinate that frame -- independent of whether a bone deviation could
    be computed at all (a bone needs its *partner* marker present too).
    """
    T, N, _ = markers.shape
    bone_pairs = [(b[0], b[1]) for b in bones]
    bone_desc = [b[2] if len(b) > 2 else f"{labels[b[0]]}-{labels[b[1]]}" for b in bones]
    err = bone_error(markers, bone_pairs, ref_lengths) if bones else np.zeros((T, 0))
    m2b = marker_bone_map(N, bone_pairs)
    available = np.isfinite(markers).all(axis=2)  # (T, N)

    frames = np.arange(T)
    rows = []
    for m in range(N):
        bone_idxs = m2b[m]
        if not bone_idxs:
            rows.append(pd.DataFrame({
                "subject_id": subject_id, "trial_id": trial_id, "frame": frames,
                "marker": labels[m], "deviation_mm": np.nan,
                "associated_bone": None, "vicon_available": available[:, m],
            }))
            continue
        sub = err[:, bone_idxs]                      # (T, n_incident_bones)
        all_nan = np.isnan(sub).all(axis=1)
        with np.errstate(invalid="ignore"):
            best = np.nanargmax(np.where(np.isnan(sub), -np.inf, sub), axis=1)
        deviation = sub[np.arange(T), best]
        deviation = np.where(all_nan, np.nan, deviation)
        assoc = [None if all_nan[t] else bone_desc[bone_idxs[best[t]]] for t in range(T)]
        rows.append(pd.DataFrame({
            "subject_id": subject_id, "trial_id": trial_id, "frame": frames,
            "marker": labels[m], "deviation_mm": deviation,
            "associated_bone": assoc, "vicon_available": available[:, m],
        }))
    return pd.concat(rows, ignore_index=True)


def add_threshold_columns(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame:
    """Add ``within_<t>mm`` boolean columns (deviation <= threshold).

    NaN deviation (no incident bone could be evaluated that frame) compares
    False against every threshold -- "within X mm" is undefined, not
    satisfied, when there is nothing to compare.
    """
    out = df.copy()
    for t in thresholds_mm:
        col = f"within_{t:g}mm"
        out[col] = out["deviation_mm"] <= t
        out.loc[df["deviation_mm"].isna(), col] = False
    return out


def flag_suspicious_events(df: pd.DataFrame, *, k: float = 6.0,
                            min_deviation_mm: float = 4.0) -> pd.DataFrame:
    """Flag frames whose bone deviation is far above its marker's typical
    level: ``threshold = max(median(deviation) + k * MAD(deviation), min_deviation_mm)``.

    Same zero-inflation guard as
    ``vicon2mano.core.agreement.flag_temporal_jumps``: a marker that is
    almost always geometrically consistent has a median and MAD of ~0, so
    without ``min_deviation_mm`` any nonzero deviation at all -- including
    ordinary sub-millimetre jitter -- would be flagged.
    """
    out = df.copy()
    out["suspicious_flag"] = False
    for name, g in out.groupby("marker"):
        d = g["deviation_mm"].to_numpy()
        finite = np.isfinite(d)
        if finite.sum() < 2:
            continue
        med = np.median(d[finite])
        mad = np.median(np.abs(d[finite] - med)) * 1.4826
        thresh = max(med + k * mad, min_deviation_mm)
        flag = finite & (d > thresh)
        out.loc[g.index, "suspicious_flag"] = flag
    return out


def geometric_events(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``suspicious_flag`` rows into contiguous per-marker events.

    Mirrors ``vicon2mano.core.agreement.temporal_events`` in shape (see
    that module for the run-length / adjacency conventions), applied to
    geometric deviation instead of label-agreement error. ``event_type`` is
    ``"single_frame_discontinuity"`` or ``"sustained_geometric_deviation"``
    -- named for what the detector measures (an implausible bone length),
    never "mislabelling" or "incorrect", since a passing bone length here
    does not establish correct identity and a failing one does not prove
    which marker is at fault.
    """
    if "suspicious_flag" not in df.columns:
        raise ValueError("call flag_suspicious_events first")

    avail_by_marker: dict[str, dict[int, bool]] = {
        name: dict(zip(g["frame"], g["vicon_available"]))
        for name, g in df.groupby("marker")
    }

    events = []
    for name, g in df.sort_values("frame").groupby("marker"):
        flagged = g[g["suspicious_flag"]]
        if flagged.empty:
            continue
        frames = flagged["frame"].to_numpy()
        deviations = flagged["deviation_mm"].to_numpy()
        assoc = flagged["associated_bone"].to_numpy()
        breaks = np.flatnonzero(np.diff(frames) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(frames) - 1]))
        available = avail_by_marker[name]
        for s, e in zip(starts, ends):
            start_frame, end_frame = int(frames[s]), int(frames[e])
            n_obs = int(e - s + 1)
            peak = s + int(np.argmax(deviations[s:e + 1]))
            before = available.get(start_frame - 1, True)
            after = available.get(end_frame + 1, True)
            events.append({
                "subject_id": g["subject_id"].iloc[0],
                "trial_id": g["trial_id"].iloc[0],
                "marker": name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration": n_obs,
                "event_type": ("single_frame_discontinuity" if n_obs == 1
                               else "sustained_geometric_deviation"),
                "associated_bone": assoc[peak],
                "deviation_magnitude_mm": float(deviations[peak]),
                "adjacent_missing_vicon": bool((not before) or (not after)),
            })
    return pd.DataFrame(events, columns=[
        "subject_id", "trial_id", "marker", "start_frame", "end_frame", "duration",
        "event_type", "associated_bone", "deviation_magnitude_mm", "adjacent_missing_vicon"])


def missingness_events(df: pd.DataFrame, *, subject_id: str, trial_id: str) -> pd.DataFrame:
    """Contiguous per-marker runs of ``vicon_available == False`` (gaps).

    Same shape as ``vicon2mano.core.agreement.missingness_events``.
    """
    events = []
    for name, g in df.sort_values("frame").groupby("marker"):
        missing = g[~g["vicon_available"]]
        if missing.empty:
            continue
        frames = missing["frame"].to_numpy()
        breaks = np.flatnonzero(np.diff(frames) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(frames) - 1]))
        for s, e in zip(starts, ends):
            events.append({
                "subject_id": subject_id, "trial_id": trial_id, "marker": name,
                "start_frame": int(frames[s]), "end_frame": int(frames[e]),
                "duration": int(e - s + 1),
            })
    return pd.DataFrame(events, columns=[
        "subject_id", "trial_id", "marker", "start_frame", "end_frame", "duration"])


def per_marker_summary(df: pd.DataFrame, events: pd.DataFrame, gaps: pd.DataFrame,
                        thresholds_mm: list[float]) -> pd.DataFrame:
    """One row per marker: availability, bone-deviation stats, event and gap
    counts -- the geometric-self-consistency analogue of
    ``vicon2mano.core.agreement.per_marker_summary`` /
    ``per_marker_event_summary`` /  ``missingness_events`` combined into one
    table, since the caller asked for a single per-marker row rather than
    several files to cross-reference.
    """
    ev_by_marker = {name: g for name, g in events.groupby("marker")} if len(events) else {}
    gap_by_marker = {name: g for name, g in gaps.groupby("marker")} if len(gaps) else {}

    rows = []
    for name, g in df.groupby("marker"):
        total = len(g)
        n_available = int(g["vicon_available"].sum())
        dev = g["deviation_mm"].dropna()

        row = {
            "marker": name,
            "expected_frame_marker_pairs": total,
            "valid_vicon_observations": n_available,
            "availability_pct": 100.0 * n_available / total if total else np.nan,
            "n_deviation_observations": len(dev),
            "mean_deviation_mm": dev.mean() if len(dev) else np.nan,
            "median_deviation_mm": dev.median() if len(dev) else np.nan,
            "std_deviation_mm": dev.std() if len(dev) else np.nan,
            "p95_deviation_mm": dev.quantile(0.95) if len(dev) else np.nan,
            "max_deviation_mm": dev.max() if len(dev) else np.nan,
        }
        for t in thresholds_mm:
            tcol = f"within_{t:g}mm"
            row[f"pct_within_{t:g}mm"] = 100.0 * g[tcol].mean() if total else np.nan

        ev = ev_by_marker.get(name)
        row["suspicious_event_count"] = len(ev) if ev is not None else 0
        row["affected_observations"] = int(ev["duration"].sum()) if ev is not None else 0
        row["median_event_duration"] = ev["duration"].median() if ev is not None else np.nan
        row["max_event_duration"] = int(ev["duration"].max()) if ev is not None else 0

        gp = gap_by_marker.get(name)
        row["vicon_gap_count"] = len(gp) if gp is not None else 0
        row["affected_missing_frames"] = int(gp["duration"].sum()) if gp is not None else 0
        row["max_gap_duration"] = int(gp["duration"].max()) if gp is not None else 0

        rows.append(row)
    return pd.DataFrame(rows).sort_values("marker").reset_index(drop=True)


def trial_overall_summary(marker_summary: pd.DataFrame) -> dict:
    """Single-row trial-wide rollup of ``per_marker_summary`` (mean
    availability, worst median/p95 deviation, total events/gaps) -- what
    goes into the dataset-level CSV, one row per trial.
    """
    return {
        "assessment_type": ASSESSMENT_TYPE,
        "n_markers": len(marker_summary),
        "mean_availability_pct": marker_summary["availability_pct"].mean(),
        "min_availability_pct": marker_summary["availability_pct"].min(),
        "mean_median_deviation_mm": marker_summary["median_deviation_mm"].mean(),
        "max_p95_deviation_mm": marker_summary["p95_deviation_mm"].max(),
        "total_suspicious_events": int(marker_summary["suspicious_event_count"].sum()),
        "total_affected_observations": int(marker_summary["affected_observations"].sum()),
        "total_vicon_gap_count": int(marker_summary["vicon_gap_count"].sum()),
        "max_gap_duration": int(marker_summary["max_gap_duration"].max()),
    }


# ---------------------------------------------------------------------------
# Reference-source metadata
#
# Static or calibration trials may intentionally serve as reference data for
# estimating subject-specific bone lengths. The reference trial can differ
# from the movement trial being assessed. Nothing here excludes, invalidates,
# or specially skips a trial based on this classification -- it is descriptive
# metadata attached to the same analysis every trial already gets, not a
# gate on whether that analysis runs.
# ---------------------------------------------------------------------------

REFERENCE_SOURCE_TYPES = ("static", "calibration", "movement", "unknown")
REFERENCE_STATUSES = ("valid", "insufficient_bones", "no_valid_reference", "unknown")


def classify_reference_source(filename: str) -> tuple[str, bool]:
    """(reference_source_type, reference_source_is_static) from a filename.

    Purely a naming-convention read, same spirit as
    ``vicon2mano.core.dataset.find_static_csv`` using ``manual_frames.csv``'s
    "Static" session name -- neither this nor that is a claim about what the
    recording actually contains, only about how it is named/logged. Returns
    ``"unknown"`` rather than guessing when the name gives no signal either
    way, since a wrong static/movement guess would be worse than admitting
    the name doesn't say.
    """
    name = filename.lower()
    if "static" in name:
        return "static", True
    if "calib" in name:
        return "calibration", True
    if any(k in name for k in ("trial", "hoi", "handsonly", "hands_only", "hands only")):
        return "movement", False
    return "unknown", False


def reference_status(ref_lengths: np.ndarray) -> str:
    """Whether :func:`vicon2mano.core.bones.consensus_bone_lengths` produced
    a usable reference: ``"valid"`` (every bone got a finite length),
    ``"insufficient_bones"`` (some but not all -- e.g. a permanently-missing
    marker knocks out only the bones touching it, see
    ``vicon2mano.core.bones.consensus_bone_lengths``'s own docstring),
    ``"no_valid_reference"`` (none), or ``"unknown"`` (no bones were defined
    for this marker set at all, so there was nothing to evaluate).
    """
    if ref_lengths.size == 0:
        return "unknown"
    finite = np.isfinite(ref_lengths)
    if finite.all():
        return "valid"
    if finite.any():
        return "insufficient_bones"
    return "no_valid_reference"


def bone_reference_table(
    markers: np.ndarray,
    bones: list[tuple[int, int, str]],
    ref_lengths: np.ndarray,
    inlier_frames: np.ndarray,
) -> pd.DataFrame:
    """Per-bone reference detail: how many of the consensus inlier frames
    actually had that specific bone finite, and the reference length that
    resulted -- the "number of valid frames used per bone" /
    "reference length per bone" figures, broken out per bone rather than
    only summarised trial-wide.
    """
    bone_ij = [(b[0], b[1]) for b in bones]
    rows = []
    for k, (i, j) in enumerate(bone_ij):
        if inlier_frames.size:
            d = np.linalg.norm(markers[inlier_frames, i] - markers[inlier_frames, j], axis=1)
            n_valid = int(np.isfinite(d).sum())
        else:
            n_valid = 0
        rows.append({
            "associated_bone": bones[k][2] if len(bones[k]) > 2 else f"{i}-{j}",
            "reference_length_mm": float(ref_lengths[k]) if np.isfinite(ref_lengths[k]) else np.nan,
            "n_valid_frames_used": n_valid,
        })
    return pd.DataFrame(rows, columns=["associated_bone", "reference_length_mm", "n_valid_frames_used"])
