"""Marker-quality cascade: label every marker correct / incorrect / missing.

A chain of trust, most-rigid part of the hand first, each step trusting only
what the step before it confirmed. See ``docs/strategies/cascade.md`` for the
full reasoning and the list of real bugs it has caught.

Moved here from ``scripts/label_marker_quality.py``: this is library code
(1600 lines, imported by the MANO driver and the notebook alike), not a
script. Its dataset/naming helpers now live in :mod:`vicon2mano.core.dataset`.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from vicon2mano.core.loader import load_c3d, load_csv
from vicon2mano.core.dataset import (          # noqa: F401  (re-exported)
    REF_CSV, find_trial_csv, is_frame_number, load_ref_ranges_csv,
    load_trial_sessions, find_session_file, parse_frame_spec,
    participant_sort_key, resolve_participants, slug, sniff_delimiter,
)

_FINGER_RE = re.compile(r"(thumb|index|middle|ring|pinky)[_]?(\d+)", re.IGNORECASE)
_PLATE_RE = re.compile(r"(forearm|palm)[_]?(\d+)", re.IGNORECASE)



def _group_markers(labels: list[str]) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, int]]]:
    """Group marker indices by finger (thumb/index/.../pinky) and by plate
    (forearm/palm), each as {digit: marker_index}, from label strings."""
    fingers: dict[str, dict[int, int]] = {}
    plates: dict[str, dict[int, int]] = {}

    for i, lbl in enumerate(labels):
        m = _FINGER_RE.search(lbl)
        if m:
            finger, digit = m.group(1).lower(), int(m.group(2))
            fingers.setdefault(finger, {})[digit] = i
            continue
        m = _PLATE_RE.search(lbl)
        if m:
            plate, digit = m.group(1).lower(), int(m.group(2))
            plates.setdefault(plate, {})[digit] = i

    return fingers, plates


def infer_bones(labels: list[str]) -> list[tuple[int, int, str]]:
    """Infer anatomically-adjacent marker pairs from label strings.

    No marker is treated as "the wrist" — there is no such marker in this
    Vicon protocol. Every finger's base marker is instead connected to
    every present Palm-plate marker (all real, physical markers), since the
    palm plate is rigid and any of its markers is an equally valid anchor.

    Returns a list of (marker_i, marker_j, description), where the
    description always names the two actual marker labels involved.
    """
    fingers, plates = _group_markers(labels)

    bones: list[tuple[int, int, str]] = []

    # Consecutive segments within each finger.
    for finger, digits in fingers.items():
        ordered = sorted(digits)
        for a, b in zip(ordered, ordered[1:]):
            bones.append((digits[a], digits[b], f"{labels[digits[a]]}-{labels[digits[b]]}"))
        first = digits[ordered[0]]
        for palm_idx in plates.get("palm", {}).values():
            if palm_idx != first:
                bones.append((palm_idx, first, f"{labels[palm_idx]}-{labels[first]}"))

    # Rigid plates: every pairwise distance within the group is constant.
    for plate, digits in plates.items():
        idxs = sorted(digits.values())
        for k, i in enumerate(idxs):
            for j in idxs[k + 1:]:
                bones.append((i, j, f"{labels[i]}-{labels[j]}"))

    return bones


def align_markers_to_labels(
    markers: np.ndarray,        # (T, M, 3)
    labels: list[str],          # length M — this recording's own labels
    target_labels: list[str],   # length N — labels to reorder/pad to
) -> np.ndarray:
    """Reorder/select marker columns from ``labels`` order to
    ``target_labels`` order, by exact (case-insensitive) name match.

    Used to align a separate recording (e.g. a static/calibration trial)
    with the marker order of the trial being evaluated, since two CSV
    exports for the same participant aren't guaranteed to list markers in
    the same column order. Target labels with no match get an all-NaN
    column (so they're simply ignored downstream, same as a missing
    marker).
    """
    idx_by_label = {l.strip().lower(): i for i, l in enumerate(labels)}
    out = np.full((markers.shape[0], len(target_labels), 3), np.nan, dtype=markers.dtype)
    for j, tl in enumerate(target_labels):
        i = idx_by_label.get(tl.strip().lower())
        if i is not None:
            out[:, j] = markers[:, i]
    return out


def build_reference(
    markers: np.ndarray,   # (T, N, 3)
    ref_frames: np.ndarray,
    bones: list[tuple[int, int, str]],
    *,
    extra_markers: np.ndarray | None = None,   # (Te, N, 3), e.g. a static trial
) -> tuple[dict[tuple[int, int], tuple[float, float]], np.ndarray]:
    """Reference bone lengths (median, MAD) and per-marker speed samples.

    ``extra_markers`` (already aligned to the same N marker columns, see
    ``align_markers_to_labels``) contributes extra samples to the bone-length
    reference only — it's a separate recording, so it has no valid
    frame-to-frame relationship to ``markers`` and isn't used for the speed
    reference.
    """
    ref = markers[ref_frames]  # (R, N, 3)
    if extra_markers is not None:
        ref = np.concatenate([ref, extra_markers], axis=0)

    bone_ref: dict[tuple[int, int], tuple[float, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, j, _ in bones:
            d = np.linalg.norm(ref[:, i] - ref[:, j], axis=-1)
            d = d[np.isfinite(d)]
            if d.size < 3:
                continue
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med))) or 1.0  # avoid zero-width
            bone_ref[(i, j)] = (med, mad)

    # Per-marker speed samples: only over *consecutive* reference frames
    # (a gap in ref_frames would otherwise look like a huge, spurious jump).
    consecutive = np.where(np.diff(ref_frames) == 1)[0]
    speeds = np.full((len(consecutive), markers.shape[1]), np.nan)
    for k, start in enumerate(consecutive):
        f0, f1 = ref_frames[start], ref_frames[start + 1]
        speeds[k] = np.linalg.norm(markers[f1] - markers[f0], axis=-1)

    return bone_ref, speeds


def label_quality(
    markers: np.ndarray,           # (T, N, 3) mm
    labels: list[str],
    ref_frames: np.ndarray,
    *,
    static_markers: np.ndarray | None = None,  # (Te, N, 3), pre-aligned to `labels`
    bone_tol_mad: float = 8.0,
    bone_tol_mm: float = 15.0,
    speed_tol_mad: float = 8.0,
    speed_tol_mm: float = 25.0,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Return (T, N) status array: 0=missing, 1=incorrect, 2=correct.

    ``static_markers``, if given, supplements the bone-length reference with
    extra frames from a separate (e.g. static/calibration) recording — see
    ``align_markers_to_labels`` to prepare it and ``build_reference`` for how
    it's used.
    """
    T, N, _ = markers.shape
    bones = infer_bones(labels)
    bone_ref, speeds = build_reference(markers, ref_frames, bones, extra_markers=static_markers)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed_med = np.nanmedian(speeds, axis=0)              # (N,)
        speed_mad = np.nanmedian(np.abs(speeds - speed_med), axis=0)
    speed_med = np.nan_to_num(speed_med, nan=0.0)
    speed_mad = np.nan_to_num(speed_mad, nan=0.0)

    present = np.isfinite(markers).all(axis=-1)                # (T, N)
    status = np.where(present, 2, 0).astype(np.int8)           # start correct/missing

    # ---- bone-length check ----
    for (i, j), (med, mad) in bone_ref.items():
        d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)  # (T,)
        tol = max(bone_tol_mad * mad, bone_tol_mm)
        bad = np.isfinite(d) & (np.abs(d - med) > tol)
        status[bad, i] = 1
        status[bad, j] = 1

    # ---- temporal check ----
    # Track each marker's displacement from its own last *present* frame,
    # normalised per elapsed frame so gaps don't falsely look like jumps.
    last_pos = np.full((N, 3), np.nan)
    last_frame = np.full(N, -1, dtype=int)
    for t in range(T):
        for n in range(N):
            if not present[t, n]:
                continue
            if last_frame[n] >= 0:
                dt = t - last_frame[n]
                dist = np.linalg.norm(markers[t, n] - last_pos[n]) / dt
                tol = max(speed_tol_mad * speed_mad[n], speed_tol_mm)
                if dist > tol:
                    status[t, n] = 1
            last_pos[n] = markers[t, n]
            last_frame[n] = t

    return status, bones


def _pairwise_bad(
    markers: np.ndarray,           # (T, N, 3)
    present: np.ndarray,           # (T, N) bool
    bone_ref: dict[tuple[int, int], tuple[float, float]],
    idxs: list[int],
    tol_mad: float,
    tol_mm: float,
) -> np.ndarray:
    """(T, len(idxs)) bool: marker k involved in an out-of-tolerance
    pairwise distance to another present marker in idxs, this frame.

    A pair with no reference (``bone_ref`` has no entry, e.g. too few valid
    reference-frame samples) is silently skipped — it contributes no "bad"
    signal for either marker, even if the pair is actually anomalous. Use
    ``debug_frame_cascade`` to check whether a specific pair has a
    reference at all.
    """
    T = markers.shape[0]
    bad = np.zeros((T, len(idxs)), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k, i in enumerate(idxs):
            for l, j in enumerate(idxs[k + 1:], start=k + 1):
                med, mad = bone_ref.get((i, j), (None, None))
                if med is None:
                    continue
                d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)
                tol = max(tol_mad * mad, tol_mm)
                bad_pair = present[:, i] & present[:, j] & (~np.isfinite(d) | (np.abs(d - med) > tol))
                bad[bad_pair, k] = True
                bad[bad_pair, l] = True
    return bad


def _pairwise_bad_anchor_tiebreak(
    markers: np.ndarray,           # (T, N, 3)
    present: np.ndarray,           # (T, N) bool
    bone_ref: dict[tuple[int, int], tuple[float, float]],
    idxs: list[int],
    tol_mad: float,
    tol_mm: float,
    anchor_ok: dict[int, np.ndarray],  # idx -> (T,) bool, from the forearm-centroid anchor check
) -> np.ndarray:
    """Like ``_pairwise_bad``, but for each out-of-tolerance pair (i, j) use
    each marker's own anchor-to-forearm-centroid check to decide who's
    actually at fault, instead of always blaming both:

    - If exactly one of i/j has a bad anchor, only that one is blamed —
      the other's own pair-independent check backs it up, so it's
      exonerated even though it sits in a failing pair.
    - If i and j's anchor checks agree (both ok or both bad), there's no
      way to tell which one actually moved — blame both, same as
      ``_pairwise_bad``.
    """
    T = markers.shape[0]
    bad = np.zeros((T, len(idxs)), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k, i in enumerate(idxs):
            for l, j in enumerate(idxs[k + 1:], start=k + 1):
                med, mad = bone_ref.get((i, j), (None, None))
                if med is None:
                    continue
                d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)
                tol = max(tol_mad * mad, tol_mm)
                bad_pair = present[:, i] & present[:, j] & (~np.isfinite(d) | (np.abs(d - med) > tol))
                ai, aj = anchor_ok[i], anchor_ok[j]
                # Exonerate i only when i's anchor is ok and j's isn't (and vice versa);
                # otherwise (agreement either way) blame both.
                blame_i = bad_pair & ~(ai & ~aj)
                blame_j = bad_pair & ~(aj & ~ai)
                bad[blame_i, k] = True
                bad[blame_j, l] = True
    return bad


def _rolling_local_reference(
    markers: np.ndarray,           # (T, N, 3)
    pairs: list[tuple[int, int]],
    window: int,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Per-pair LOCAL median and residual-scale over a centered rolling
    window of nearby frames, instead of a single global reference built
    from a handful of (possibly far-away-in-time) ``ref_frames``.

    Markers sitting on skin, not a rigid plate, legitimately drift several
    mm over the course of a long recording as the limb rotates and muscle
    moves underneath — comparing every frame in a 35,000-frame trial
    against 2-3 spot-checked snapshots from elsewhere in the trial makes
    that ordinary drift indistinguishable from a real error. Comparing
    each frame to its own recent neighbourhood instead treats slow,
    smooth drift as the expected baseline, and reserves flagging for
    *abrupt* deviations from it — which is what an actual tracking error
    or occlusion-fill jump looks like.

    Returns ``{(i, j): (local_med, local_dev)}``, each ``(T,)``, NaN where
    the window had no valid samples at all (e.g. right at a long gap).

    ``local_dev`` is deliberately **not** a windowed MAD (the median of
    ``|x - median(x)|`` over each window's raw distances) — it is the
    rolling median of ``|d[t] - local_med[t]|``, i.e. of each frame's
    residual from its *own* centered local median. That is the exact
    statistic the caller then thresholds, so this estimates the null
    distribution of the quantity actually being tested, rather than the
    spread of the raw values.

    The distinction is invisible in a stationary window (both give the
    same number, and both are equally robust to a minority of bad frames)
    and matters under drift — which is the whole reason this local
    reference exists. Measured on synthetic data with 6mm of drift per
    301-frame window: this residual scale reads 0.33mm, a true windowed
    MAD reads 1.50mm, the latter dominated by the within-window trend
    rather than by the noise. Since the centered median already tracks
    that trend, the residual being tested does *not* grow with it, so a
    true MAD would inflate the tolerance (2.5mm -> 6.0mm here) purely as a
    function of how fast the forearm happens to be drifting. Downstream,
    over that same drift, it costs nearly all sensitivity to the small
    sustained errors this cascade is already worst at catching: recall on
    injected jumps goes 87.5% -> 0% at 3mm and 100% -> 1.7% at 5mm, with
    no reduction in false positives to show for it (0% either way). So
    don't "correct" this into a real MAD without re-tuning
    ``forearm_tol_mm`` / ``forearm_tol_mad`` to win that recall back.
    """
    import pandas as pd  # local import: only needed by this experimental path

    T = markers.shape[0]
    min_periods = max(5, window // 10)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i, j in pairs:
        d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)
        s = pd.Series(d)
        local_med = s.rolling(window, center=True, min_periods=min_periods).median()
        local_dev = (s - local_med).abs().rolling(window, center=True, min_periods=min_periods).median()
        out[(i, j)] = (local_med.to_numpy(), local_dev.to_numpy())
    return out


def label_quality_cascade(
    markers: np.ndarray,           # (T, N, 3) mm
    labels: list[str],
    ref_frames: np.ndarray,
    *,
    static_markers: np.ndarray | None = None,  # (Te, N, 3), pre-aligned to `labels`
    forearm_tol_mad: float = 4.0,
    forearm_tol_mm: float = 2.5,
    forearm_max_bad: int = 2,
    forearm_window: int = 301,
    forearm_sanity_mm: float = 10.0,
    forearm_dev_inflation: float = 3.0,
    anchor_tol_mad: float = 1.5,
    anchor_tol_mm: float = 12.0,
    palm_min_present: int = 2,
    palm_min_bonelength_ok: int = 2,
    palm_min_anchor_ok: int = 2,
    bone_tol_mad: float = 4.0,
    bone_tol_mm: float = 10.0,
    speed_tol_mad: float = 4.0,
    speed_tol_mm: float = 12.5,
    sticky_tol_mm: float = 5.0,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Cascading trust-chain marker quality check.

    Unlike ``label_quality`` (every bone checked independently against a
    fixed reference), this evaluates markers in a strict dependency chain —
    forearm gates palm, palm gates each finger's base marker, each finger
    marker gates the next one out — where a broken link forces everything
    downstream of it to "incorrect" rather than leaving it independently
    checkable:

    1. **Forearm gate.** The Forearm markers sit on skin, not a rigid
       plate — their mutual distances legitimately drift several mm over a
       long recording as the forearm rotates and the muscle underneath
       moves, so instead of comparing every frame to a handful of
       possibly-far-away-in-time ``ref_frames`` (indistinguishable from a
       real error at that scale), each pairwise distance is compared to
       its own *local* median/MAD over a ``forearm_window``-frame centered
       rolling window (see ``_rolling_local_reference``) — slow drift is
       absorbed into the local baseline, and only an *abrupt* deviation
       from a marker's own recent neighbourhood gets flagged. A purely
       local baseline has a blind spot, though: a genuine tracking failure
       that's *sustained* for longer than about half the window (a marker
       stuck at a wrong reconstructed position, not just a one-frame
       blip) looks just like a new, stable pose from inside the window —
       seen on real data, where a Forearm plate distance jumped from
       ~55mm to ~150mm in a single frame and then held there for hundreds
       of frames, and the local median duly adopted 150mm as "normal".
       So the local median is also sanity-checked against the *global*
       ``ref_frames``-based reference: if it strays more than
       ``forearm_sanity_mm`` from that fixed anchor, the local baseline
       itself is untrusted and the pair is flagged regardless of how well
       the current frame matches its own (drifted) local neighbourhood.
       The local *spread* gets a matching guard. That spread
       (``local_dev``) is the rolling median of each frame's residual from
       its own local median — deliberately not a windowed MAD; see
       ``_rolling_local_reference`` for why, and don't convert it into one
       without re-tuning the tolerances. Being a median, it survives a
       minority of bad frames in its window; past that it inflates, and
       since the tolerance is ``forearm_tol_mad`` times it, the check
       would silently widen its own tolerance exactly where the data is
       worst — failing permissive, the opposite of this cascade's intended
       bias. So it's capped at ``forearm_dev_inflation`` times the global
       reference MAD (real skin drift moves the median slowly and leaves
       the residual scale alone, so quiet and drifting windows alike are
       unaffected by the cap). Unlike an out-of-range median, an inflated
       spread doesn't make the frame unverifiable — it only withholds the
       extra slack, and the ``forearm_tol_mm`` floor still applies. Note
       the cap only binds past ~50% contamination of a window, where the
       median sanity check above is the other line of defence.
       (Note: a pairwise-distance check structurally can't catch two Forearm
       markers swapping labels with each other, since swapping doesn't
       change the distance between them — that job falls to the temporal
       speed check below instead, which cares about *identity/motion*
       plausibility, not raw geometry.) A Forearm marker is "bad" that
       frame if it's missing, or any such local deviation involves it. If
       ``forearm_max_bad`` or more of the Forearm markers are bad
       (default: 2 of 4), the whole frame fails the gate. A frame that
       passes has each Forearm marker individually marked
       "correct"/"incorrect" per its own bad flag.
    2. **Palm gate**, only evaluated in frames that passed stage 1. The
       "Palm group" is the 3 Palm-plate markers *plus* Thumb1, folded in as
       a de facto 4th member — it sits close enough to the palm to usefully
       anchor off it, and the extra vote means a single occluded/bad Palm
       marker doesn't as easily starve the gate below its required
       minimum (fewer whole-frame vetoes, at the cost of Thumb1 itself
       being resolved here instead of via its own finger-base check in
       stage 3). A Palm-group marker is "anchor-ok" if its distance to the
       Forearm centroid is within tolerance. It's "bone-length-ok" if
       present and every pairwise distance to another present Palm-group
       marker (the physical Palm plate is rigid, and Thumb1's distance to
       each Palm marker is treated the same way here) is within tolerance
       — except when a pairwise distance fails and the *other* marker in
       that pair is anchor-ok while this one isn't: a rigid pairing can't
       tell you which of the two moved, but the anchor check can, so only
       the anchor-bad one is blamed and the anchor-ok one is exonerated for
       that pair (if both are anchor-ok or both anchor-bad, there's no way
       to tell them apart and both are blamed, same as a plain pairwise
       check). The gate requires at least ``palm_min_present`` Palm-group
       markers present, at least ``palm_min_bonelength_ok`` bone-length-ok,
       and at least ``palm_min_anchor_ok`` anchor-ok — otherwise the whole
       frame fails. A Palm-group marker counts as individually "correct"
       (for stage 3) only if present AND bone-length-ok AND anchor-ok.
    3. **Finger chain**, only evaluated in frames that passed stages 1-2.
       Thumb1 was already resolved in stage 2 as part of the Palm group, so
       the thumb's chain starts directly from its stage-2 status. Every
       other finger's first marker is checked against every Palm-group
       marker it actually has a bone to (the 3 physical Palm markers —
       Thumb1 has no bone to another finger's base, so it's excluded from
       this specific check rather than letting its correctness count
       without ever being geometrically compared) that came out
       individually "correct" in stage 2; if there is no such correct,
       bonded marker to compare against, or any of those reference distances
       is out of tolerance, it is "incorrect". Each subsequent marker on
       the finger is checked against the *previous* marker's actual
       position, but is only eligible to be "correct" if the previous
       marker was itself "correct" — one broken link marks everything
       further out on that finger "incorrect" too, even if its own
       consecutive distance happens to look fine.
    4. **Gate veto.** A frame that failed the forearm gate (step 1) has
       *every* present marker in it — Forearm included — forced to
       "incorrect", since an untrustworthy plate means nothing in the
       frame is actually confirmed. A frame that passed the forearm gate
       but failed the palm gate (step 2) only has its Palm and finger
       markers forced to "incorrect" — the Forearm markers keep their own
       stage-1 status, since the plate itself was independently confirmed
       rigid regardless of what's wrong with the palm/fingers.
    5. **Reference-frame pin.** Every present marker in a ``ref_frames``
       frame is forced "correct" outright, overriding every check above —
       these were manually verified, so they're trusted unconditionally
       rather than being subject to the same checks they themselves helped
       build (a reference frame can otherwise fail the temporal check
       purely because the frame *before* it was bad, not because the
       reference frame itself is suspect).
    6. **Stickiness override**, run last, but — unlike every stage above —
       it can only reinstate an *unverified* "incorrect", never overrule a
       *verified* one. Every "incorrect" marking above is tagged internally
       as one or the other: "verified" means some check actually measured
       a real deviation against trustworthy data (a pairwise/anchor
       distance genuinely out of tolerance, an implausible speed, ...);
       "unverified" means it was marked incorrect only for lack of
       something to confirm it against — no individually-correct anchor to
       compare a finger-base to, an untrustworthy local Forearm reference,
       swept up in a whole-frame gate veto with no direct evidence of its
       own, and so on. A marker that was "correct" in an *adjacent* frame
       (before or after) and has moved at most ``sticky_tol_mm`` since
       stays "correct" too, but only when its current "incorrect" is
       unverified — a marker that barely moved can't have been swapped or
       relabelled, so an unverified incorrect isn't this marker's own
       problem, but a verified one is real, positive evidence this
       override must never paper over no matter how little the marker
       moved. (Seen on real data: a persistent label swap between two
       fingers held both physically still for hundreds of frames
       afterward, each step individually well under tolerance — comparing
       only to the adjacent frame without this distinction would otherwise
       resurrect "correct" status for the entire wrong stretch. A Forearm
       pairwise failure is a special case: it can only implicate a *pair*,
       never say which of the two moved, so only a marker implicated in
       *every one* of its pairs — the signature of the actual culprit — is
       treated as verified; a marker only partly implicated stays eligible
       for this override, same as before verified/unverified existed,
       since here stickiness is the only available way to tell a moving
       culprit from its stationary neighbours.)

       Runs forward (frame-by-frame in time order) then backward
       (frame-by-frame in reverse), each a full sweep, so a short
       unverified-bad stretch sandwiched between two confirmed-good frames
       gets rescued from whichever side reaches it, not just whichever
       comes chronologically first. A marker that goes missing breaks the
       chain — the frame right after it reappears gets no free pass, it
       has to earn "correct" again from the checks above first (from
       either direction).

    A per-marker temporal (frame-to-frame speed) check is applied on top,
    same as ``label_quality``, before the gate veto (step 4) has final say.

    ``static_markers``, if given, supplements the bone-length reference with
    extra frames from a separate (e.g. static/calibration) recording — see
    ``align_markers_to_labels``. Deliberately NOT used for the forearm-anchor-
    distance reference: an anchor distance depends on the palm's pose
    relative to the forearm (unlike a bone length, which is pose-invariant),
    so a static calibration pose can sit measurably off from a dynamic
    trial's anchor distances with no tracking error involved — see the
    comment above the anchor-reference block for a measured example.

    Returns (T, N) status array: 0=missing, 1=incorrect, 2=correct.
    """
    T, N, _ = markers.shape
    fingers, plates = _group_markers(labels)
    forearm_idxs = sorted(plates.get("forearm", {}).values())

    # Thumb1 sits close enough to the palm plate to usefully anchor off it
    # too — folding it in as a de facto 4th palm marker gives the stage-2
    # quorum an extra vote, so a single occluded/bad Palm marker doesn't as
    # easily starve the gate below its required minimum. (It's still the
    # base of the thumb's own finger chain in stage 3, just resolved here
    # instead of there.)
    thumb_digits = fingers.get("thumb")
    thumb_first_idx = thumb_digits[min(thumb_digits)] if thumb_digits else None
    palm_idxs = sorted(plates.get("palm", {}).values())
    if thumb_first_idx is not None:
        palm_idxs = sorted(palm_idxs + [thumb_first_idx])
    palm_group_set = {thumb_first_idx} if thumb_first_idx is not None else set()

    present = np.isfinite(markers).all(axis=-1)                # (T, N)
    status = np.where(present, 2, 0).astype(np.int8)

    # A marker marked "incorrect" for a real, positive reason (a measured
    # distance/anchor/speed that actually failed against trustworthy
    # reference data) is "verified bad" — the stickiness override below
    # must never reinstate it. A marker marked "incorrect" only because
    # there wasn't enough corroborating data to confirm it either way (no
    # individually-correct anchor to compare against, an untrustworthy
    # local reference, swept up in a gate veto with no direct evidence of
    # its own, ...) stays eligible for stickiness — "can't verify" is not
    # the same claim as "verified wrong".
    verified_bad = np.zeros((T, N), dtype=bool)

    all_bones = infer_bones(labels)  # for visualisation / return value only
    bone_ref, speeds = build_reference(markers, ref_frames, all_bones, extra_markers=static_markers)

    # ---- stage 1: forearm gate (local rolling reference — see docstring) ----
    if forearm_idxs:
        forearm_pairs = [(i, j) for k, i in enumerate(forearm_idxs) for j in forearm_idxs[k + 1:]]
        forearm_local_ref = _rolling_local_reference(markers, forearm_pairs, forearm_window)
        forearm_bad_marker = ~present[:, forearm_idxs]
        forearm_verified_bad_marker = np.zeros((T, len(forearm_idxs)), dtype=bool)
        # Per marker, per frame: how many of its pairs could actually be
        # judged this frame (both markers present, trustworthy local
        # reference) and how many of those it was implicated in. Needed to
        # tell the culprit ("bad in every pair it takes part in") from an
        # innocent partner ("bad only in the pair it shares with the
        # culprit") — see below.
        forearm_pair_evaluable = np.zeros((T, len(forearm_idxs)), dtype=np.int16)
        forearm_pair_real_bad = np.zeros((T, len(forearm_idxs)), dtype=np.int16)
        idx_pos = {idx: k for k, idx in enumerate(forearm_idxs)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for (i, j), (local_med, local_dev) in forearm_local_ref.items():
                d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)
                dev_floor = np.nan_to_num(local_dev, nan=0.0)
                dev_floor[dev_floor == 0] = 1.0  # avoid zero-width tolerance

                # Sanity-check the local baseline itself against the global,
                # ref_frames-based reference — a local median that has
                # drifted implausibly far from it means a sustained tracking
                # failure got mistaken for a new "normal", not real skin
                # drift (see docstring). Skip if there's no global reference
                # for this pair to sanity-check against.
                global_med, global_mad = bone_ref.get((i, j), (None, None))
                if global_med is not None:
                    local_untrustworthy = np.isfinite(local_med) & (np.abs(local_med - global_med) > forearm_sanity_mm)
                    # ...and the same treatment for the local *spread*
                    # (`local_dev` — a residual scale, not a windowed MAD;
                    # see `_rolling_local_reference`). Being a median it
                    # survives a minority of bad frames in the window, but
                    # past that it inflates, and since the tolerance is a
                    # multiple of it, the check would silently get *more*
                    # permissive exactly where the data is dirtiest —
                    # failing toward "wave it through", the opposite of this
                    # cascade's intended bias. So it's capped at
                    # `forearm_dev_inflation` times the global reference
                    # MAD; real skin drift is a slow shift of the median and
                    # leaves this residual scale alone, so quiet and
                    # drifting windows alike are unaffected by the cap.
                    # Capping (rather than declaring the pair unverifiable,
                    # as an out-of-range median does) keeps the frame
                    # *evaluable*: an inflated spread is no reason to stop
                    # measuring, only a reason to withhold the extra slack.
                    # The `forearm_tol_mm` floor below still guarantees a
                    # workable tolerance if the global MAD is itself tiny
                    # (ref frames close together in time). Measured: the cap
                    # only binds past ~50% contamination of a window.
                    dev_floor = np.minimum(dev_floor, forearm_dev_inflation * global_mad)
                else:
                    local_untrustworthy = np.zeros(T, dtype=bool)

                tol = np.maximum(forearm_tol_mad * dev_floor, forearm_tol_mm)

                both_present = present[:, i] & present[:, j]
                # A local reference is only trustworthy to measure a real
                # deviation against when it exists at all AND passed its own
                # sanity check — a distance vs. an untrustworthy local
                # baseline is "can't verify", not positive evidence, even
                # though the pair still gets marked incorrect defensively.
                has_trustworthy_local = np.isfinite(local_med) & ~local_untrustworthy
                real_bad = both_present & has_trustworthy_local & np.isfinite(d) & (np.abs(d - local_med) > tol)
                unverifiable_bad = both_present & (~has_trustworthy_local | ~np.isfinite(d))
                bad_pair = real_bad | unverifiable_bad

                forearm_bad_marker[bad_pair, idx_pos[i]] = True
                forearm_bad_marker[bad_pair, idx_pos[j]] = True
                forearm_verified_bad_marker[real_bad, idx_pos[i]] = True
                forearm_verified_bad_marker[real_bad, idx_pos[j]] = True

                evaluable = both_present & has_trustworthy_local & np.isfinite(d)
                for pos in (idx_pos[i], idx_pos[j]):
                    forearm_pair_evaluable[:, pos] += evaluable
                    forearm_pair_real_bad[:, pos] += real_bad
        # A pairwise check can only implicate a *pair*, not tell you which
        # of the two actually moved — unlike Palm, Forearm markers have no
        # independent anchor to break that tie with. The real culprit in a
        # single-marker failure shows up bad in *every* one of its pairs,
        # while an innocent partner only shows up bad in the one pair it
        # shares with the culprit — so only count a marker as genuinely
        # confirmed-bad (blocking stickiness) when it's implicated in all
        # of its pairs; a marker only partly implicated stays eligible,
        # same as before this real/unverifiable split existed.
        # Implicated in *every* pair it could be judged in this frame (and in
        # at least one) — counted per marker over its own pairs, not over
        # how many markers happen to be flagged in the frame.
        fully_implicated = (
            (forearm_pair_evaluable > 0)
            & (forearm_pair_real_bad == forearm_pair_evaluable)
        )
        forearm_confirmed_bad_marker = forearm_verified_bad_marker & fully_implicated
        forearm_gate_ok = forearm_bad_marker.sum(axis=1) < forearm_max_bad
    else:
        forearm_bad_marker = np.zeros((T, 0), dtype=bool)
        forearm_confirmed_bad_marker = np.zeros((T, 0), dtype=bool)
        forearm_gate_ok = np.zeros(T, dtype=bool)

    for k, idx in enumerate(forearm_idxs):
        bad = present[:, idx] & forearm_bad_marker[:, k]
        status[bad, idx] = 1
        verified_bad[present[:, idx] & forearm_confirmed_bad_marker[:, k], idx] = True

    # ---- forearm centroid + anchor-distance reference (Palm & finger bases) ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid = np.nanmean(markers[:, forearm_idxs], axis=1) if forearm_idxs else np.full((T, 3), np.nan)

    ref_ok = ref_frames[forearm_gate_ok[ref_frames]] if forearm_idxs else np.array([], dtype=int)

    # Deliberately NOT pooled with static_markers, unlike the bone-length
    # reference above. Bone lengths are pose-invariant (a rigid segment
    # measures the same regardless of how the hand is posed), so a static
    # recording is a legitimate extra sample of them. An anchor distance
    # (Palm/finger-base marker to the Forearm centroid) is not pose-invariant
    # in the same way — the palm's angle relative to the forearm changes
    # with wrist/thumb pose, so a static calibration pose and a dynamic HOI
    # trial can genuinely sit at different anchor distances even with no
    # tracking error at all. Measured on P4/Trial 1 HOI: pooling 362 static
    # frames against 3 trial ref-frames left Palm3's anchor reference
    # 9.5mm from the trial-only value (which itself was within ~1-7mm of
    # real HOI frames) and made every anchor_tol_mm setting wrong somewhere
    # — too tight and it fails on the static-pose bias throughout the trial,
    # too loose and it stops catching real Palm/finger-base errors. So the
    # anchor reference draws only from this trial's own (forearm-gate-passing)
    # ref_frames.
    finger_first_idxs = [digits[min(digits)] for digits in fingers.values()]
    anchor_ok: dict[int, np.ndarray] = {}  # idx -> (T,) bool, distance-to-centroid within tolerance
    # idx -> (T,) bool, a REAL measured deviation (not just "couldn't verify"
    # from too few reference samples) — see the verified_bad note above.
    anchor_verified_bad: dict[int, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for idx in palm_idxs + finger_first_idxs:
            d_ref = np.linalg.norm(markers[ref_ok, idx] - centroid[ref_ok], axis=-1)
            d_ref = d_ref[np.isfinite(d_ref)]
            if d_ref.size < 2:
                anchor_ok[idx] = np.zeros(T, dtype=bool)  # not enough reference data to verify
                anchor_verified_bad[idx] = np.zeros(T, dtype=bool)  # can't verify != verified wrong
                continue
            med = float(np.median(d_ref))
            mad = float(np.median(np.abs(d_ref - med))) or 1.0
            tol = max(anchor_tol_mad * mad, anchor_tol_mm)
            d = np.linalg.norm(markers[:, idx] - centroid, axis=-1)
            anchor_ok[idx] = np.isfinite(d) & (np.abs(d - med) <= tol)
            anchor_verified_bad[idx] = np.isfinite(d) & (np.abs(d - med) > tol)

    # ---- stage 2: palm gate ----
    if palm_idxs:
        palm_present = present[:, palm_idxs]                                    # (T, P)
        # Every "bad" flag _pairwise_bad_anchor_tiebreak returns is a real
        # measured deviation (a pair with no reference is silently skipped,
        # never counted "bad") — bone-length badness is always verified.
        palm_bonelength_bad = _pairwise_bad_anchor_tiebreak(
            markers, present, bone_ref, palm_idxs, bone_tol_mad, bone_tol_mm, anchor_ok)
        palm_bonelength_ok = palm_present & ~palm_bonelength_bad
        palm_anchor_ok = np.stack([anchor_ok[idx] for idx in palm_idxs], axis=1)  # (T, P)
        palm_anchor_verified_bad = np.stack([anchor_verified_bad[idx] for idx in palm_idxs], axis=1)

        palm_gate_ok = (
            forearm_gate_ok
            & (palm_present.sum(axis=1) >= palm_min_present)
            & (palm_bonelength_ok.sum(axis=1) >= palm_min_bonelength_ok)
            & (palm_anchor_ok.sum(axis=1) >= palm_min_anchor_ok)
        )
        # A Palm marker is individually "correct" only if it clears every
        # criterion itself, not just the frame-level >= counts.
        palm_marker_ok = palm_present & palm_bonelength_ok & palm_anchor_ok
        palm_marker_verified_bad = palm_present & (palm_bonelength_bad | palm_anchor_verified_bad)
    else:
        palm_gate_ok = np.zeros(T, dtype=bool)
        palm_marker_ok = np.zeros((T, 0), dtype=bool)
        palm_marker_verified_bad = np.zeros((T, 0), dtype=bool)

    for k, idx in enumerate(palm_idxs):
        bad = present[:, idx] & forearm_gate_ok & ~palm_marker_ok[:, k]
        status[bad, idx] = 1
        verified_bad[bad & palm_marker_verified_bad[:, k], idx] = True

    # ---- stage 3: finger chain, gated by the palm stage ----
    gate_ok = forearm_gate_ok & palm_gate_ok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for finger, digits in fingers.items():
            ordered = sorted(digits)
            first_idx = digits[ordered[0]]

            if first_idx in palm_group_set:
                # Thumb1 was already folded into the palm group and fully
                # resolved in stage 2 — reuse that status (and its
                # verified_bad) instead of re-deriving it here (it has no
                # bones to the *other* fingers' base markers, so the check
                # below wouldn't have anything meaningful to compare it
                # against anyway).
                # Gate explicitly: stages 1/2 only *write* status for frames
                # that passed the forearm gate, so a vetoed frame still reads
                # back as 2 here (the veto runs later). Without `gate_ok` the
                # chain below would propagate trust it never had and could
                # stamp a stickiness-blocking `verified_bad` on the thumb in
                # a frame where nothing was trustworthy.
                prev_idx = first_idx
                prev_ok = gate_ok & (status[:, first_idx] == 2)
                prev_verified_bad = verified_bad[:, first_idx]
            else:
                # Finger-base marker: must be within tolerance of every
                # individually-correct Palm-group marker it actually has a
                # bone to (Thumb1, folded into the palm group, has no bone
                # to another finger's base — restrict to markers that do,
                # so its correctness can't silently inflate the count
                # without ever being checked against). No individually-
                # correct, bonded marker to compare against means it can't
                # be confirmed, so it's marked incorrect (no fallback).
                relevant = [k for k, palm_idx in enumerate(palm_idxs)
                            if bone_ref.get((palm_idx, first_idx)) is not None]
                n_correct_palm = palm_marker_ok[:, relevant].sum(axis=1) if relevant else np.zeros(T, dtype=int)
                chain_ok = gate_ok & (n_correct_palm > 0)
                chain_verified_bad = np.zeros(T, dtype=bool)
                for k in relevant:
                    palm_idx = palm_idxs[k]
                    med, mad = bone_ref[(palm_idx, first_idx)]
                    d = np.linalg.norm(markers[:, palm_idx] - markers[:, first_idx], axis=-1)
                    tol = max(bone_tol_mad * mad, bone_tol_mm)
                    pair_ok = np.isfinite(d) & (np.abs(d - med) <= tol)
                    # Only this pair's failure matters where that Palm-group
                    # marker is actually one of the "correct" ones being
                    # compared against — and only counts as *real* evidence
                    # under that same condition.
                    chain_verified_bad |= palm_marker_ok[:, k] & ~pair_ok
                    chain_ok &= ~palm_marker_ok[:, k] | pair_ok

                bad = present[:, first_idx] & ~chain_ok
                status[bad, first_idx] = 1
                verified_bad[bad & chain_verified_bad, first_idx] = True

                prev_idx, prev_ok, prev_verified_bad = first_idx, chain_ok, chain_verified_bad
            for d in ordered[1:]:
                cur_idx = digits[d]
                med, mad = bone_ref.get((prev_idx, cur_idx), (None, None))
                if med is None:
                    # No reference for this pair at all — can't verify this
                    # link itself either way; only inherit real evidence
                    # already established further up the chain.
                    cur_ok = np.zeros(T, dtype=bool)
                    cur_verified_bad = prev_verified_bad.copy()
                else:
                    dist = np.linalg.norm(markers[:, prev_idx] - markers[:, cur_idx], axis=-1)
                    tol = max(bone_tol_mad * mad, bone_tol_mm)
                    dist_ok = np.isfinite(dist) & (np.abs(dist - med) <= tol)
                    cur_ok = prev_ok & dist_ok
                    # Only counts as real evidence against *this* marker
                    # when it's actually being compared against a trusted
                    # (prev_ok) predecessor; otherwise inherit whatever
                    # real evidence already indicted the chain upstream.
                    real_local_bad = prev_ok & ~dist_ok
                    cur_verified_bad = prev_verified_bad | real_local_bad

                bad = present[:, cur_idx] & ~cur_ok
                status[bad, cur_idx] = 1
                verified_bad[bad & cur_verified_bad, cur_idx] = True
                prev_idx, prev_ok, prev_verified_bad = cur_idx, cur_ok, cur_verified_bad

    # ---- temporal check (same as label_quality) ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed_med = np.nanmedian(speeds, axis=0)
        speed_mad = np.nanmedian(np.abs(speeds - speed_med), axis=0)
    speed_med = np.nan_to_num(speed_med, nan=0.0)
    speed_mad = np.nan_to_num(speed_mad, nan=0.0)

    last_pos = np.full((N, 3), np.nan)
    last_frame = np.full(N, -1, dtype=int)
    for t in range(T):
        for n in range(N):
            if not present[t, n]:
                continue
            if last_frame[n] >= 0:
                dt = t - last_frame[n]
                dist = np.linalg.norm(markers[t, n] - last_pos[n]) / dt
                tol = max(speed_tol_mad * speed_mad[n], speed_tol_mm)
                if dist > tol:
                    status[t, n] = 1
                    verified_bad[t, n] = True  # a real measured speed anomaly, not a lack of data
            last_pos[n] = markers[t, n]
            last_frame[n] = t

    # ---- gate veto ----
    # A failed forearm gate invalidates the *whole* frame, Forearm markers
    # included — an untrustworthy plate means nothing in the frame, not
    # even the plate's own markers, can actually be confirmed.
    bad_forearm_frame = ~forearm_gate_ok
    status[bad_forearm_frame] = np.where(present[bad_forearm_frame], 1, 0)

    # A failed palm gate (with the forearm gate itself OK) only invalidates
    # what depends on the palm — Palm and finger markers, forced to
    # "incorrect" rather than left at whatever stages 2/3 computed — but
    # leaves the Forearm markers' own stage-1 status alone, since the
    # forearm plate was independently confirmed rigid this frame regardless
    # of what's wrong with the palm/fingers.
    bad_palm_frame = forearm_gate_ok & ~palm_gate_ok
    non_forearm_idxs = [i for i in range(N) if i not in forearm_idxs]
    if non_forearm_idxs:
        sub_present = present[np.ix_(bad_palm_frame, non_forearm_idxs)]
        status[np.ix_(bad_palm_frame, non_forearm_idxs)] = np.where(sub_present, 1, 0)

    # ---- reference frames are ground truth ----
    # ``ref_frames`` were manually verified, so every present marker in one
    # is pinned "correct" outright, overriding every check above —
    # including the temporal check, which would otherwise compare a
    # reference frame against whatever came right before it in time and
    # flag it for a "jump" that's actually the *preceding* frame being bad
    # (e.g. a stretch of genuinely corrupted tracking that happens to end
    # right at the verified-good frame). This runs before the stickiness
    # override below, so a pinned reference frame can also anchor
    # stickiness into its clean, unmoving neighbours.
    present_ref = present[ref_frames]
    status[ref_frames] = np.where(present_ref, 2, 0)

    # ---- stickiness override ----
    # A marker that was "correct" in an adjacent frame and hasn't moved
    # more than ``sticky_tol_mm`` since stays "correct" too — but *only* if
    # its "incorrect" here was never *verified*, i.e. no check actually
    # measured a real deviation against trustworthy data; it was marked
    # incorrect purely for lack of something to confirm it against (no
    # individually-correct anchor, an untrustworthy local reference, swept
    # up in a gate veto with no direct evidence of its own, ...). A marker
    # that barely moved can't have been swapped with a different physical
    # marker or relabelled mid-air, so an *unverified* incorrect isn't this
    # marker's own problem — but a *verified* one is real, positive
    # evidence that this override must never paper over, no matter how
    # little the marker moved (seen on real data: a persistent label swap
    # between two fingers can hold both physically still for hundreds of
    # frames afterward, during which every step is individually well under
    # tolerance — comparing only to the adjacent frame would otherwise
    # resurrect "correct" status for the entire wrong stretch).
    #
    # Runs forward (t-1 -> t) then backward (t+1 -> t), each a full
    # frame-by-frame sweep so a whole run of static frames propagates from
    # whichever side reaches it — e.g. a short bad stretch sandwiched
    # between two confirmed-good frames gets rescued from both ends, not
    # just whichever one happens to come chronologically first. A marker
    # that goes missing breaks the chain there — the frame right after it
    # reappears gets no free pass, it has to earn "correct" again first
    # (from either direction).
    for t in range(1, T):
        moved = np.linalg.norm(markers[t] - markers[t - 1], axis=-1)  # (N,)
        stuck = (
            (status[t] != 2) & ~verified_bad[t]
            & (status[t - 1] == 2)
            & present[t] & present[t - 1]
            & np.isfinite(moved) & (moved <= sticky_tol_mm)
        )
        status[t, stuck] = 2

    for t in range(T - 2, -1, -1):
        moved = np.linalg.norm(markers[t] - markers[t + 1], axis=-1)  # (N,)
        stuck = (
            (status[t] != 2) & ~verified_bad[t]
            & (status[t + 1] == 2)
            & present[t] & present[t + 1]
            & np.isfinite(moved) & (moved <= sticky_tol_mm)
        )
        status[t, stuck] = 2

    return status, all_bones


def debug_frame_cascade(
    markers: np.ndarray,           # (T, N, 3) mm
    labels: list[str],
    ref_frames: np.ndarray,
    t: int,
    *,
    static_markers: np.ndarray | None = None,
    forearm_tol_mad: float = 4.0,
    forearm_tol_mm: float = 2.5,
    forearm_max_bad: int = 2,
    forearm_window: int = 301,
    forearm_sanity_mm: float = 10.0,
    forearm_dev_inflation: float = 3.0,
    anchor_tol_mad: float = 1.5,
    anchor_tol_mm: float = 12.0,
    palm_min_present: int = 2,
    palm_min_bonelength_ok: int = 2,
    palm_min_anchor_ok: int = 2,
    bone_tol_mad: float = 4.0,
    bone_tol_mm: float = 10.0,
) -> None:
    """Print exactly how ``label_quality_cascade`` reasoned about frame
    ``t`` — every pairwise/anchor distance it checked, the reference it
    checked against, and whether each gate/marker passed. Same keyword
    arguments as ``label_quality_cascade`` (pass the same values you used to
    produce the status array you're investigating); minus ``speed_tol_*``,
    since the temporal check is a simple per-marker before/after comparison
    that doesn't need this kind of breakdown.
    """
    T, N, _ = markers.shape
    fingers, plates = _group_markers(labels)
    forearm_idxs = sorted(plates.get("forearm", {}).values())

    thumb_digits = fingers.get("thumb")
    thumb_first_idx = thumb_digits[min(thumb_digits)] if thumb_digits else None
    palm_idxs = sorted(plates.get("palm", {}).values())
    if thumb_first_idx is not None:
        palm_idxs = sorted(palm_idxs + [thumb_first_idx])

    present = np.isfinite(markers).all(axis=-1)

    all_bones = infer_bones(labels)
    bone_ref, _speeds = build_reference(markers, ref_frames, all_bones, extra_markers=static_markers)

    def _pair_line(i, j, tol_mad, tol_mm):
        med, mad = bone_ref.get((i, j), (None, None))
        if med is None:
            return f"    {labels[i]} <-> {labels[j]}: NO REFERENCE (skipped — never flags either marker)"
        d = np.linalg.norm(markers[t, i] - markers[t, j])
        tol = max(tol_mad * mad, tol_mm)
        ok = np.isfinite(d) and abs(d - med) <= tol
        both_present = present[t, i] and present[t, j]
        verdict = "OK" if ok else "BAD"
        if not both_present:
            verdict += " (but not scored — a marker is missing this frame)"
        return (f"    {labels[i]} <-> {labels[j]}: d={d:.1f}mm  ref={med:.1f}±{mad:.2f}mm  "
                f"tol=±{tol:.1f}mm  -> {verdict}")

    print(f"===== Frame {t} cascade trace =====")

    # ---- stage 1: forearm (local rolling reference — skin, not a rigid plate) ----
    print(f"\n-- Stage 1: Forearm gate (fails if >= {forearm_max_bad} of "
          f"{len(forearm_idxs)} markers bad; local {forearm_window}-frame rolling "
          f"reference, not the global ref_frames) --")
    forearm_pairs = [(i, j) for k, i in enumerate(forearm_idxs) for j in forearm_idxs[k + 1:]]
    forearm_local_ref = _rolling_local_reference(markers, forearm_pairs, forearm_window) if forearm_idxs else {}
    # Vectorized over all T frames (not just t) — forearm_gate_ok_all is
    # needed at every ref_frames index below to build the anchor reference,
    # not only at the one frame we're printing.
    forearm_bad_marker = ~present[:, forearm_idxs] if forearm_idxs else np.zeros((T, 0), dtype=bool)
    idx_pos = {idx: k for k, idx in enumerate(forearm_idxs)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, j in forearm_pairs:
            local_med, local_dev = forearm_local_ref[(i, j)]
            d = np.linalg.norm(markers[:, i] - markers[:, j], axis=-1)
            dev_floor = np.nan_to_num(local_dev, nan=0.0)
            dev_floor[dev_floor == 0] = 1.0

            global_med, global_mad = bone_ref.get((i, j), (None, None))
            if global_med is not None:
                local_untrustworthy = np.isfinite(local_med) & (np.abs(local_med - global_med) > forearm_sanity_mm)
                dev_capped = dev_floor > forearm_dev_inflation * global_mad
                dev_floor = np.minimum(dev_floor, forearm_dev_inflation * global_mad)
            else:
                local_untrustworthy = np.zeros(T, dtype=bool)
                dev_capped = np.zeros(T, dtype=bool)

            tol = np.maximum(forearm_tol_mad * dev_floor, forearm_tol_mm)

            bad_pair = (
                present[:, i] & present[:, j]
                & (~np.isfinite(d) | ~np.isfinite(local_med) | (np.abs(d - local_med) > tol) | local_untrustworthy)
            )
            forearm_bad_marker[bad_pair, idx_pos[i]] = True
            forearm_bad_marker[bad_pair, idx_pos[j]] = True

            # Print just frame t's numbers.
            med_t, dev_t = local_med[t], local_dev[t]
            if not np.isfinite(med_t):
                print(f"    {labels[i]} <-> {labels[j]}: NO LOCAL REFERENCE "
                      f"(too few valid frames in the {forearm_window}-frame window here)")
                continue
            tol_t = tol[t]
            both_present = present[t, i] and present[t, j]
            verdict = "OK" if not bad_pair[t] else "BAD"
            if not both_present:
                verdict += " (but not scored — a marker is missing this frame)"
            elif local_untrustworthy[t]:
                verdict += (f" (local baseline itself is {abs(med_t - global_med):.1f}mm from the "
                            f"global reference {global_med:.1f}mm — untrusted, sanity cap ±{forearm_sanity_mm:.1f}mm)")
            cap_str = ""
            if dev_capped[t]:
                cap_str = (f"  [local spread capped: {dev_t:.2f}mm > "
                           f"{forearm_dev_inflation:.1f}x global MAD {global_mad:.2f}mm]")
            print(f"    {labels[i]} <-> {labels[j]}: d={d[t]:.1f}mm  local_ref={med_t:.1f}±{dev_t:.2f}mm  "
                  f"tol=±{tol_t:.1f}mm  -> {verdict}{cap_str}")
    forearm_gate_ok_all = forearm_bad_marker.sum(axis=1) < forearm_max_bad if forearm_idxs else np.zeros(T, bool)
    n_bad = int(forearm_bad_marker[t].sum()) if forearm_idxs else 0
    for k, idx in enumerate(forearm_idxs):
        present_str = "present" if present[t, idx] else "MISSING"
        bad_str = "bad" if forearm_bad_marker[t, k] else "ok"
        print(f"    {labels[idx]}: {present_str}, pairwise={bad_str}")
    print(f"  -> {n_bad} bad forearm marker(s); "
          f"gate {'PASSES' if forearm_gate_ok_all[t] else 'FAILS'}")

    # ---- anchor references (Palm + finger bases), same as label_quality_cascade ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid = np.nanmean(markers[:, forearm_idxs], axis=1) if forearm_idxs else np.full((T, 3), np.nan)
    ref_ok = ref_frames[forearm_gate_ok_all[ref_frames]] if forearm_idxs else np.array([], dtype=int)
    # Deliberately trial-only, not pooled with static_markers — see the
    # matching comment in label_quality_cascade for why an anchor distance
    # (unlike bone length) isn't pose-invariant enough to borrow from a
    # static recording.
    finger_first_idxs = [digits[min(digits)] for digits in fingers.values()]
    anchor_ok: dict[int, np.ndarray] = {}
    anchor_ref_info: dict[int, tuple[float, float, int]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for idx in palm_idxs + finger_first_idxs:
            d_ref = np.linalg.norm(markers[ref_ok, idx] - centroid[ref_ok], axis=-1)
            d_ref = d_ref[np.isfinite(d_ref)]
            if d_ref.size < 2:
                anchor_ok[idx] = np.zeros(T, dtype=bool)
                anchor_ref_info[idx] = (float("nan"), float("nan"), int(d_ref.size))
                continue
            med = float(np.median(d_ref))
            mad = float(np.median(np.abs(d_ref - med))) or 1.0
            anchor_ref_info[idx] = (med, mad, int(d_ref.size))
            tol = max(anchor_tol_mad * mad, anchor_tol_mm)
            d = np.linalg.norm(markers[:, idx] - centroid, axis=-1)
            anchor_ok[idx] = np.isfinite(d) & (np.abs(d - med) <= tol)

    # ---- stage 2: palm ----
    print(f"\n-- Stage 2: Palm gate (needs >= {palm_min_present} present, "
          f">= {palm_min_bonelength_ok} bone-length-ok, >= {palm_min_anchor_ok} anchor-ok "
          f"out of {len(palm_idxs)}) --")
    if not forearm_gate_ok_all[t]:
        print("  Forearm gate failed this frame, so palm can't be trusted either "
              "(still evaluated below for reference).")

    print("  Palm-to-Forearm-centroid anchor distance (used below to break ties):")
    for idx in palm_idxs:
        med, mad, n_ref = anchor_ref_info[idx]
        dist = np.linalg.norm(markers[t, idx] - centroid[t])
        ok = bool(anchor_ok[idx][t])
        if n_ref < 2:
            print(f"    {labels[idx]}: d={dist:.1f}mm  NO/INSUFFICIENT REFERENCE "
                  f"({n_ref} sample(s)) -> BAD (can't verify)")
        else:
            tol = max(anchor_tol_mad * mad, anchor_tol_mm)
            print(f"    {labels[idx]}: d={dist:.1f}mm  ref={med:.1f}±{mad:.2f}mm "
                  f"(n={n_ref})  tol=±{tol:.1f}mm  -> {'OK' if ok else 'BAD'}")

    print("  Palm-group pairwise (rigidity; Thumb1 is folded in as a 4th member), "
          "with anchor-based tie-break on failures:")
    palm_pairs = [(i, j) for k, i in enumerate(palm_idxs) for j in palm_idxs[k + 1:]]
    for i, j in palm_pairs:
        print(_pair_line(i, j, bone_tol_mad, bone_tol_mm))
        med, mad = bone_ref.get((i, j), (None, None))
        if med is None:
            continue
        d = np.linalg.norm(markers[t, i] - markers[t, j])
        tol = max(bone_tol_mad * mad, bone_tol_mm)
        pair_bad = present[t, i] and present[t, j] and (not np.isfinite(d) or abs(d - med) > tol)
        if not pair_bad:
            continue
        ai, aj = bool(anchor_ok[i][t]), bool(anchor_ok[j][t])
        if ai and not aj:
            print(f"      -> tie-break: {labels[i]} anchor OK, {labels[j]} anchor BAD: "
                  f"blame {labels[j]} only, exonerate {labels[i]}")
        elif aj and not ai:
            print(f"      -> tie-break: {labels[j]} anchor OK, {labels[i]} anchor BAD: "
                  f"blame {labels[i]} only, exonerate {labels[j]}")
        else:
            agree = "both anchor-OK" if ai else "both anchor-BAD"
            print(f"      -> tie-break: {agree}, can't tell which moved: blame both")

    palm_bonelength_ok_all = present[:, palm_idxs] & ~_pairwise_bad_anchor_tiebreak(
        markers, present, bone_ref, palm_idxs, bone_tol_mad, bone_tol_mm, anchor_ok) if palm_idxs \
        else np.zeros((T, 0), dtype=bool)

    print("  Per-marker summary:")
    for k, idx in enumerate(palm_idxs):
        present_str = "present" if present[t, idx] else "MISSING"
        bl_str = "ok" if palm_bonelength_ok_all[t, k] else "bad"
        an_str = "ok" if anchor_ok[idx][t] else "bad"
        individually_ok = present[t, idx] and palm_bonelength_ok_all[t, k] and anchor_ok[idx][t]
        print(f"    {labels[idx]}: {present_str}, bone-length={bl_str}, anchor={an_str} "
              f"-> individually {'CORRECT' if individually_ok else 'incorrect'}")

    n_present = int(present[t, palm_idxs].sum()) if palm_idxs else 0
    n_bl_ok = int(palm_bonelength_ok_all[t].sum()) if palm_idxs else 0
    n_an_ok = int(sum(bool(anchor_ok[idx][t]) for idx in palm_idxs))
    palm_gate_ok_t = (
        forearm_gate_ok_all[t]
        and n_present >= palm_min_present
        and n_bl_ok >= palm_min_bonelength_ok
        and n_an_ok >= palm_min_anchor_ok
    )
    print(f"  -> present={n_present}/{len(palm_idxs)}, bone-length-ok={n_bl_ok}/{len(palm_idxs)}, "
          f"anchor-ok={n_an_ok}/{len(palm_idxs)}; gate {'PASSES' if palm_gate_ok_t else 'FAILS'}")

    print(f"\n-- Final veto --")
    if not forearm_gate_ok_all[t]:
        print("  Forearm gate failed -> every present marker in this frame (Forearm included) "
              "is forced to INCORRECT, regardless of what stages 1-2 computed individually above.")
    elif not palm_gate_ok_t:
        print("  Palm gate failed (Forearm gate passed) -> every present Palm/finger marker is "
              "forced to INCORRECT, but Forearm markers keep their own stage-1 status (the plate "
              "was independently confirmed rigid this frame).")
    else:
        print("  Both gates passed -> each marker's status above (and the finger chain, "
              "not traced by this function) stands.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="Vicon Nexus CSV export")
    p.add_argument("--c3d", help="Vicon .c3d file (alternative to --csv)")
    p.add_argument("--ref-frames",
                    help='Known-correct frames, e.g. "1000-1500,20000-20500"')
    p.add_argument("--ref-csv", help="tab-separated manual-frames log (alternative to --ref-frames)")
    p.add_argument("--participant", help="participant name, matched against --ref-csv; "
                                          "also used to resolve --csv when neither --csv "
                                          "nor --c3d is given (see find_trial_csv)")
    p.add_argument("--trial", help="trial name, matched against --ref-csv; also used to "
                                    "resolve --csv when neither --csv nor --c3d is given")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--bone-tol-mad", type=float, default=8.0)
    p.add_argument("--bone-tol-mm", type=float, default=15.0)
    p.add_argument("--speed-tol-mad", type=float, default=8.0)
    p.add_argument("--speed-tol-mm", type=float, default=25.0)
    args = p.parse_args(argv)

    if not args.csv and not args.c3d:
        # Same lookup gmm/mano's relabel_trial.py scripts already use: the
        # trial CSV's own path lives in manual_frames.csv next to the
        # reference-frame ranges, so --participant/--trial alone (the same
        # two values --ref-csv already requires) are enough to find it --
        # no reason to also make the caller spell out the file path.
        if not (args.participant and args.trial):
            p.error("one of --csv / --c3d is required, unless both "
                     "--participant and --trial are given (to resolve the "
                     "CSV via manual_frames.csv/find_trial_csv)")
        resolved, *_ = find_trial_csv(args.participant, args.trial)
        if resolved is None:
            p.error(f"no CSV found for participant={args.participant!r} "
                     f"trial={args.trial!r} in manual_frames.csv "
                     "-- pass --csv/--c3d explicitly instead")
        args.csv = str(resolved)
    if not args.ref_frames and not args.ref_csv:
        p.error("one of --ref-frames / --ref-csv is required")
    if args.ref_csv and not (args.participant and args.trial):
        p.error("--ref-csv requires --participant and --trial")

    if args.csv:
        markers, labels = load_csv(args.csv)
    else:
        markers, labels, _fps = load_c3d(args.c3d)

    if args.ref_csv:
        ref_frames = load_ref_ranges_csv(args.ref_csv, args.participant, args.trial)
    else:
        ref_frames = parse_frame_spec(args.ref_frames)
    ref_frames = ref_frames[(ref_frames >= 0) & (ref_frames < markers.shape[0])]
    if ref_frames.size < 3:
        p.error(f"Reference frames resolved to only {ref_frames.size} valid frame(s); need more")

    status, bones = label_quality(
        markers, labels, ref_frames,
        bone_tol_mad=args.bone_tol_mad, bone_tol_mm=args.bone_tol_mm,
        speed_tol_mad=args.speed_tol_mad, speed_tol_mm=args.speed_tol_mm,
    )

    print(f"{markers.shape[0]} frames, {len(labels)} markers, "
          f"{len(ref_frames)} reference frames, {len(bones)} bones inferred")
    for i, j, name in bones:
        print(f"  bone: {name}  ({labels[i]} <-> {labels[j]})")

    names = np.array(["missing", "incorrect", "correct"])
    counts = {n: int((status == v).sum()) for v, n in enumerate(names)}
    total = status.size
    print(f"Overall: {counts['correct']}/{total} correct "
          f"({counts['correct']/total:.1%}), "
          f"{counts['incorrect']} incorrect, {counts['missing']} missing")

    per_marker = status
    print("\nPer-marker incorrect rate (excluding missing):")
    for n in range(len(labels)):
        col = per_marker[:, n]
        present_n = (col != 0).sum()
        bad_n = (col == 1).sum()
        rate = bad_n / present_n if present_n else 0.0
        if bad_n:
            print(f"  {labels[n]:24s} {bad_n:6d}/{present_n:<6d} ({rate:.1%})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("frame," + ",".join(labels) + "\n")
        for t in range(markers.shape[0]):
            f.write(f"{t}," + ",".join(names[status[t]]) + "\n")
    print(f"\nSaved per-frame per-marker labels to {out_path}")


if __name__ == "__main__":
    main()
