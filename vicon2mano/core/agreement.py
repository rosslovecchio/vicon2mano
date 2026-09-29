"""Frame-by-frame agreement between two labellings of the same recording.

Shared by ``scripts/shared/analyze_manual_agreement.py`` (Vicon labels vs. a
manually-labelled export) and anything else that needs "how far apart are
these two (T, N, 3) marker arrays, marker for marker" without assuming
anything about which one is "correct". No relabelling, thresholds, or
correction happens here -- this is measurement only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def base_name(label: str) -> str:
    """Strip a Vicon subject prefix: "P10:Palm1" -> "Palm1"."""
    return label.split(":")[-1]


@dataclass
class MarkerMatch:
    common: list[str]           # labels present in both, base-name form
    idx_a: dict[str, int]       # base name -> column index into markers_a
    idx_b: dict[str, int]       # base name -> column index into markers_b
    only_a: list[str]
    only_b: list[str]


def match_markers(labels_a: list[str], labels_b: list[str]) -> MarkerMatch:
    """Pair up two marker-column lists by base name (prefix stripped).

    A duplicate base name within one side is a genuine ambiguity -- silently
    picking one column would be a guess, so the first occurrence wins and a
    warning-worthy count is left in the caller's hands (``idx_a``/``idx_b``
    only ever hold the first index seen).
    """
    idx_a: dict[str, int] = {}
    for i, l in enumerate(labels_a):
        idx_a.setdefault(base_name(l), i)
    idx_b: dict[str, int] = {}
    for i, l in enumerate(labels_b):
        idx_b.setdefault(base_name(l), i)
    common = sorted(set(idx_a) & set(idx_b))
    only_a = sorted(set(idx_a) - set(idx_b))
    only_b = sorted(set(idx_b) - set(idx_a))
    return MarkerMatch(common, idx_a, idx_b, only_a, only_b)


def per_frame_table(
    markers_a: np.ndarray,
    markers_b: np.ndarray,
    match: MarkerMatch,
    *,
    label_a: str = "vicon",
    label_b: str = "manual",
) -> pd.DataFrame:
    """Long-format per-frame, per-marker comparison for the common markers.

    ``markers_a``/``markers_b`` must share the same frame axis (same T,
    same frame meaning) -- this never shifts or interpolates frames, so if
    the two recordings are not already frame-aligned the caller must fix
    that upstream, or the numbers here are meaningless.

    Every (frame, marker) pair in ``match.common`` gets a row, with
    ``match_status`` recording whether both sides had a finite coordinate,
    so no observation is silently dropped -- rows with missing coordinates
    stay in the table with NaN error columns.
    """
    Ta, Tb = markers_a.shape[0], markers_b.shape[0]
    T = min(Ta, Tb)
    if Ta != Tb:
        print(f"[warn] frame counts differ ({label_a}={Ta}, {label_b}={Tb}); "
              f"comparing the first {T} frames only")

    frames = np.arange(T)
    rows = []
    for name in match.common:
        a = markers_a[:T, match.idx_a[name], :]
        b = markers_b[:T, match.idx_b[name], :]
        ok_a = np.isfinite(a).all(axis=1)
        ok_b = np.isfinite(b).all(axis=1)
        both = ok_a & ok_b

        d = a - b
        euclid = np.linalg.norm(d, axis=1)

        status = np.full(T, "", dtype=object)
        status[both] = "matched"
        status[ok_a & ~ok_b] = "missing_manual_coordinates"
        status[~ok_a & ok_b] = "missing_vicon_coordinates"
        status[~ok_a & ~ok_b] = "missing_both_coordinates"

        rows.append(pd.DataFrame({
            "frame": frames,
            "marker": name,
            f"{label_a}_x": a[:, 0], f"{label_a}_y": a[:, 1], f"{label_a}_z": a[:, 2],
            f"{label_b}_x": b[:, 0], f"{label_b}_y": b[:, 1], f"{label_b}_z": b[:, 2],
            "dx_mm": np.where(both, d[:, 0], np.nan),
            "dy_mm": np.where(both, d[:, 1], np.nan),
            "dz_mm": np.where(both, d[:, 2], np.nan),
            "euclidean_error_mm": np.where(both, euclid, np.nan),
            "match_status": status,
        }))
    return pd.concat(rows, ignore_index=True)


def add_threshold_columns(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame:
    """Add ``within_<t>mm`` boolean columns for each configured threshold.

    NaN error rows compare False against every threshold, which is correct:
    "within X mm" is undefined, not satisfied, when there is nothing to
    compare.
    """
    out = df.copy()
    for t in thresholds_mm:
        col = f"within_{t:g}mm"
        out[col] = out["euclidean_error_mm"] <= t
        out.loc[df["euclidean_error_mm"].isna(), col] = False
    return out


def flag_temporal_jumps(df: pd.DataFrame, *, k: float = 6.0,
                         min_jump_mm: float = 2.0) -> pd.DataFrame:
    """Flag frames whose error jumps far from its own marker's typical level.

    Per marker: ``threshold = median(error) + k * MAD(error)``, MAD scaled
    by 1.4826 to be a normal-consistent estimate of standard deviation, the
    usual robust-outlier convention. Diagnostic only -- flagged rows are
    marked, never dropped or corrected.

    This data is heavily zero-inflated (a manually-labelled export mostly
    *reuses* the Vicon trajectory it agrees with -- see the module-level
    note on ``per_frame_table`` -- so the modal error is an exact 0), which
    collapses the per-marker MAD to 0 for essentially every marker here.
    Without a floor, ``thresh`` becomes 0 and *every* frame with any nonzero
    error at all gets flagged -- confirmed on P10/Trial2_handsonly, where
    this produced 20033/42238 "jumps", one per nonzero-error frame rather
    than one per genuine anomaly. ``min_jump_mm`` floors the threshold so a
    degenerate (near-zero) MAD can't turn ordinary small disagreements into
    flagged anomalies.
    """
    out = df.copy()
    out["error_jump_flag"] = False
    for name, g in out.groupby("marker"):
        e = g["euclidean_error_mm"].to_numpy()
        finite = np.isfinite(e)
        if finite.sum() < 2:
            continue
        med = np.median(e[finite])
        mad = np.median(np.abs(e[finite] - med)) * 1.4826
        thresh = max(med + k * mad, min_jump_mm)
        flag = finite & (e > thresh)
        out.loc[g.index, "error_jump_flag"] = flag
    return out


_MISSING_STATUSES = ("missing_vicon_coordinates", "missing_both_coordinates")


def temporal_events(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``error_jump_flag`` rows into contiguous per-marker events.

    A run of flagged frames only continues while the frame numbers are
    actually consecutive (``frame[t] - frame[t-1] == 1``); a gap in frame
    numbering always starts a new event. Distinguishes "how many distinct
    anomalies" (rows here) from "how many marker-frame observations they
    cover" (each row's ``n_observations``), which a single flat count of
    flagged rows conflates.

    Each event is also classified:

    - ``event_type``: ``"single_frame_discontinuity"`` (one flagged frame)
      vs. ``"sustained_label_disagreement"`` (>1 consecutive flagged frames).
      Both names describe what the detector measures -- a run of sharp
      coordinate disagreement between the two files -- not a claim about
      *why* it happened (e.g. one continuous manual reassignment operation
      vs. several separate ones that happen to be adjacent); that would
      need the manual-labelling process itself as evidence, which this
      module does not have.
    - ``adjacent_missing_vicon``: whether the frame immediately before the
      event's start or after its end (same marker) has no Vicon coordinate.
      This never *reclassifies* an event as caused by missingness -- the
      event itself is, by construction, a run of *matched, valid* frames
      with high disagreement (see ``flag_temporal_jumps``) -- but a "Yes"
      here says the surrounding trajectory has a gap, which is relevant
      context for whether the disagreement coincides with a Vicon dropout
      rather than sitting in an otherwise-clean stretch.
    """
    if "error_jump_flag" not in df.columns:
        raise ValueError("call flag_temporal_jumps first")

    # frame -> match_status per marker, for the adjacency check below.
    status_by_marker: dict[str, dict[int, str]] = {
        name: dict(zip(g["frame"], g["match_status"]))
        for name, g in df.groupby("marker")
    }

    events = []
    for name, g in df.sort_values("frame").groupby("marker"):
        flagged = g[g["error_jump_flag"]]
        if flagged.empty:
            continue
        frames = flagged["frame"].to_numpy()
        errors = flagged["euclidean_error_mm"].to_numpy()
        breaks = np.flatnonzero(np.diff(frames) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(frames) - 1]))
        statuses = status_by_marker[name]
        for s, e in zip(starts, ends):
            start_frame, end_frame = int(frames[s]), int(frames[e])
            n_obs = int(e - s + 1)
            before = statuses.get(start_frame - 1)
            after = statuses.get(end_frame + 1)
            adjacent_missing = before in _MISSING_STATUSES or after in _MISSING_STATUSES
            events.append({
                "marker": name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_observations": n_obs,
                "max_error_mm": float(errors[s:e + 1].max()),
                "event_type": ("single_frame_discontinuity" if n_obs == 1
                               else "sustained_label_disagreement"),
                "adjacent_missing_vicon": bool(adjacent_missing),
            })
    return pd.DataFrame(events, columns=[
        "marker", "start_frame", "end_frame", "n_observations", "max_error_mm",
        "event_type", "adjacent_missing_vicon"])


def missingness_events(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous per-marker runs of missing coordinates (trajectory gaps).

    Same run-length convention as :func:`temporal_events`, applied to
    ``match_status`` instead of the error-magnitude flag -- this reports
    *availability* (does a coordinate exist at all), which is a distinct
    failure mode from *agreement* (does an existing coordinate match).
    """
    events = []
    for name, g in df.sort_values("frame").groupby("marker"):
        missing = g[g["match_status"].isin(_MISSING_STATUSES)]
        if missing.empty:
            continue
        frames = missing["frame"].to_numpy()
        breaks = np.flatnonzero(np.diff(frames) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(frames) - 1]))
        for s, e in zip(starts, ends):
            events.append({
                "marker": name,
                "start_frame": int(frames[s]),
                "end_frame": int(frames[e]),
                "n_observations": int(e - s + 1),
            })
    return pd.DataFrame(events, columns=[
        "marker", "start_frame", "end_frame", "n_observations"])


def per_marker_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    """One row per marker: event count, durations, and type breakdown.

    ``duration`` is measured in frames (``n_observations``, since an event's
    frames are contiguous by construction). NaN durations mean the marker
    had zero events, not zero duration.
    """
    if events.empty:
        return pd.DataFrame(columns=[
            "marker", "event_count", "affected_observations",
            "median_event_duration", "max_event_duration", "total_event_duration",
            "n_single_frame_events", "n_sustained_events", "n_adjacent_missing_vicon"])
    rows = []
    for name, g in events.groupby("marker"):
        dur = g["n_observations"]
        rows.append({
            "marker": name,
            "event_count": len(g),
            "affected_observations": int(dur.sum()),
            "median_event_duration": dur.median(),
            "max_event_duration": int(dur.max()),
            "total_event_duration": int(dur.sum()),
            "n_single_frame_events": int((g["event_type"] == "single_frame_discontinuity").sum()),
            "n_sustained_events": int((g["event_type"] == "sustained_label_disagreement").sum()),
            "n_adjacent_missing_vicon": int(g["adjacent_missing_vicon"].sum()),
        })
    return (pd.DataFrame(rows)
            .sort_values("affected_observations", ascending=False)
            .reset_index(drop=True))


def per_marker_summary(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame:
    """One row per marker: match counts, missingness, and error stats.

    NaN is used wherever a statistic cannot be computed (e.g. no matched
    observations for that marker) rather than 0, which would silently read
    as "matched perfectly".
    """
    rows = []
    for name, g in df.groupby("marker"):
        total = len(g)
        matched = int((g["match_status"] == "matched").sum())
        miss_manual = int((g["match_status"] == "missing_manual_coordinates").sum())
        miss_vicon = int((g["match_status"] == "missing_vicon_coordinates").sum())
        miss_both = int((g["match_status"] == "missing_both_coordinates").sum())
        err = g["euclidean_error_mm"].dropna()
        n_valid = len(err)
        n_zero = int((err == 0).sum())
        n_nonzero = n_valid - n_zero

        row = {
            "marker": name,
            "n_frames": total,
            "matched": matched,
            "missing_manual_coordinates": miss_manual,
            "missing_vicon_coordinates": miss_vicon,
            "missing_both_coordinates": miss_both,
            "matched_pct": 100.0 * matched / total if total else np.nan,
            "missing_pct": 100.0 * (miss_manual + miss_vicon + miss_both) / total if total else np.nan,
            # Vicon-side availability specifically -- distinct from "matched",
            # which also requires the manual side to be present. On a gap-filled
            # manual export the manual side is never missing, so the two happen
            # to coincide there, but this is computed independently so it stays
            # correct against a manual export that has its own gaps too.
            "vicon_availability_pct": 100.0 * (total - miss_vicon - miss_both) / total if total else np.nan,
            # A "valid comparison" is a matched pair with a computable error
            # (== matched here, since per_frame_table already NaNs out any
            # matched-but-non-finite case) -- kept as its own count so a
            # reader doesn't have to infer it from matched vs. missing.
            "valid_comparisons": n_valid,
            "zero_error_comparisons": n_zero,
            "nonzero_error_comparisons": n_nonzero,
            "zero_error_pct": 100.0 * n_zero / n_valid if n_valid else np.nan,
            "mean_error_mm": err.mean() if len(err) else np.nan,
            "median_error_mm": err.median() if len(err) else np.nan,
            "std_error_mm": err.std() if len(err) else np.nan,
            "rmse_mm": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
            "p90_error_mm": err.quantile(0.90) if len(err) else np.nan,
            "p95_error_mm": err.quantile(0.95) if len(err) else np.nan,
            "p99_error_mm": err.quantile(0.99) if len(err) else np.nan,
            "max_error_mm": err.max() if len(err) else np.nan,
        }
        for axis in ("x", "y", "z"):
            col = f"d{axis}_mm"
            valid = g[col].dropna()
            row[f"mean_abs_{axis}_error_mm"] = valid.abs().mean() if len(valid) else np.nan
            row[f"std_{axis}_error_mm"] = valid.std() if len(valid) else np.nan
        for t in thresholds_mm:
            tcol = f"within_{t:g}mm"
            row[f"pct_within_{t:g}mm"] = 100.0 * g[tcol].mean() if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("marker").reset_index(drop=True)


def overall_summary(df: pd.DataFrame, thresholds_mm: list[float]) -> dict:
    """Single-row dict of trial-wide agreement stats (see ``per_marker_summary``
    for the same statistics broken out per marker)."""
    total = len(df)
    matched = int((df["match_status"] == "matched").sum())
    miss_manual = int((df["match_status"] == "missing_manual_coordinates").sum())
    miss_vicon = int((df["match_status"] == "missing_vicon_coordinates").sum())
    miss_both = int((df["match_status"] == "missing_both_coordinates").sum())
    err = df["euclidean_error_mm"].dropna()
    n_valid = len(err)
    n_zero = int((err == 0).sum())
    out = {
        "n_rows": total,
        "n_markers": df["marker"].nunique(),
        "n_frames": int(df["frame"].nunique()),
        "matched": matched,
        "missing_manual_coordinates": miss_manual,
        "missing_vicon_coordinates": miss_vicon,
        "missing_both_coordinates": miss_both,
        "matched_pct": 100.0 * matched / total if total else np.nan,
        "missing_pct": 100.0 * (miss_manual + miss_vicon + miss_both) / total if total else np.nan,
        "valid_comparisons": n_valid,
        "zero_error_comparisons": n_zero,
        "nonzero_error_comparisons": n_valid - n_zero,
        "zero_error_pct": 100.0 * n_zero / n_valid if n_valid else np.nan,
        "mean_error_mm": err.mean() if len(err) else np.nan,
        "median_error_mm": err.median() if len(err) else np.nan,
        "rmse_mm": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
        "p95_error_mm": err.quantile(0.95) if len(err) else np.nan,
        "max_error_mm": err.max() if len(err) else np.nan,
    }
    for t in thresholds_mm:
        tcol = f"within_{t:g}mm"
        out[f"pct_within_{t:g}mm"] = 100.0 * df[tcol].mean() if total else np.nan
    return out


def reliability_table(marker_summary: pd.DataFrame, event_summary: pd.DataFrame,
                       missingness: pd.DataFrame) -> pd.DataFrame:
    """One row per marker, combining availability, identity, and event burden
    into the three separately-meaningful quantities a reader actually needs:

    - ``availability_pct``: how often a Vicon coordinate exists at all
      (``vicon_availability_pct`` from ``per_marker_summary``).
    - ``label_retention_pct_valid``: of the frames where both sides have a
      coordinate, how often the manual export kept the original identity
      (``zero_error_pct``).
    - ``reassignment_pct_valid``: the complement of the above -- how often
      the labeller changed the identity (``100 - zero_error_pct``).
    - ``overall_usable_pct``: zero-error comparisons over *all expected*
      frame-marker pairs, not just the valid ones -- this is the one number
      that combines availability and retention, since a marker can only be
      "usable" if the coordinate both exists and was left unreassigned. It
      is a strictly more conservative number than either input alone --
      e.g. a marker at 76% availability and 95% retention-given-availability
      is only ~72% usable overall, not 95%.

    Event/gap columns (``event_count``, ``sustained_event_count``,
    ``max_event_duration``, ``max_gap_duration``) are 0 for a marker with no
    events or gaps, not NaN -- "zero events happened" is a real, known value,
    unlike the error statistics elsewhere in this module that use NaN for
    "cannot be computed".
    """
    out = marker_summary[["marker", "vicon_availability_pct", "zero_error_pct",
                          "valid_comparisons", "n_frames"]].copy()
    out = out.rename(columns={"vicon_availability_pct": "availability_pct",
                              "zero_error_pct": "label_retention_pct_valid"})
    out["reassignment_pct_valid"] = 100.0 - out["label_retention_pct_valid"]
    out["overall_usable_pct"] = (100.0 * marker_summary["zero_error_comparisons"]
                                 / marker_summary["n_frames"])

    ev = (event_summary.set_index("marker")[["event_count", "n_sustained_events",
                                             "max_event_duration"]]
          if len(event_summary) else pd.DataFrame(
              columns=["event_count", "n_sustained_events", "max_event_duration"]))
    ev = ev.rename(columns={"n_sustained_events": "sustained_event_count"})
    out = out.set_index("marker").join(ev).reset_index()
    for col in ("event_count", "sustained_event_count", "max_event_duration"):
        out[col] = out[col].fillna(0).astype(int)

    gaps = (missingness.groupby("marker")["n_observations"].max().rename("max_gap_duration")
            if len(missingness) else pd.Series(name="max_gap_duration", dtype=int))
    out = out.set_index("marker").join(gaps).reset_index()
    out["max_gap_duration"] = out["max_gap_duration"].fillna(0).astype(int)

    return out[["marker", "availability_pct", "label_retention_pct_valid",
               "reassignment_pct_valid", "event_count", "sustained_event_count",
               "max_event_duration", "max_gap_duration", "overall_usable_pct"]] \
        .sort_values("overall_usable_pct").reset_index(drop=True)


def cross_trial_summary(all_reliability: pd.DataFrame) -> pd.DataFrame:
    """One row per marker: how its :func:`reliability_table` numbers vary
    across several trials (``all_reliability`` = that table from each trial,
    concatenated, tagged by whatever trial-identifying columns the caller
    used -- only ``marker`` and the reliability columns are read here).

    With a single trial (the common case today -- manually-labelled exports
    are scarce) every ``*_std`` is NaN and every ``*_min``/``*_max`` pair is
    equal by construction. That is the correct way to represent "no
    between-trial evidence yet", not a bug to special-case around -- a
    reader comparing ``*_min`` to ``*_max`` sees immediately that there is
    nothing to compare.
    """
    rows = []
    for marker, g in all_reliability.groupby("marker"):
        row = {"marker": marker, "n_trials": len(g)}
        for col in ("availability_pct", "label_retention_pct_valid", "overall_usable_pct"):
            v = g[col]
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_min"] = v.min()
            row[f"{col}_max"] = v.max()
            row[f"{col}_std"] = v.std() if len(v) > 1 else np.nan
        row["total_event_count"] = int(g["event_count"].sum())
        row["total_sustained_event_count"] = int(g["sustained_event_count"].sum())
        row["max_gap_duration"] = int(g["max_gap_duration"].max())
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("overall_usable_pct_mean")
            .reset_index(drop=True))
