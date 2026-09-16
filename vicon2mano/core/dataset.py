"""Locating recordings on disk: participants, trials, statics, references.

Pure dataset plumbing, shared by every strategy: parsing manual_frames.csv
and trial_filename_map.csv, resolving a (participant, trial) pair to files
on disk, and normalising participant/trial names.

Consolidated here from two places that had grown into de-facto shared
libraries: ``scripts/relabel_with_mano.py`` (which the GMM tooling imported
purely to call :func:`find_trial_csv`) and the top third of
``scripts/label_marker_quality.py``. Neither was a sensible home -- the
first is a strategy driver, the second a strategy implementation.
"""

from __future__ import annotations

import csv as csv_mod
import re
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(r"C:\Users\RL000009\OneDrive - Vrije Universiteit Brussel\A-Skills\data_June25")
REF_CSV = DATA_ROOT / "manual_frames.csv"
TRIAL_MAP_CSV = DATA_ROOT / "trial_filename_map.csv"
MANO_DIR = REPO_ROOT.parent / "clean_kinematics" / "mano_v1_2" / "models"

# Static recordings usable for MANO calibration but NOT as the cascade's
# bone-length reference. P9's static was taken after a Forearm marker fell
# off and was reattached, so its forearm geometry no longer matches the
# trials -- folding it into the cascade reference collapsed 3 of P9's 4
# trials to 0% correct, so it is deliberately left unmapped in
# trial_filename_map.csv. MANO calibration only touches the 16 *hand*
# markers, so the forearm disturbance is irrelevant there.
MANO_ONLY_STATIC = {
    "P9": "static_after_forerm1_fall.csv",
}


def _normalise_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_ref_ranges_csv(
    path: str, participant: str, trial: str,
) -> np.ndarray:
    """Look up manually-verified good frames from the manual-frames log.

    The delimiter is sniffed per file: the log started out tab-separated and
    has since been re-saved as CSV, and parsing a comma file as TSV collapses
    every line into a single field, so *nothing* matches and the caller
    silently falls back to an automatic reference. That failure is quiet and
    expensive — on P7/Trial1_handsonly it cut the calibration set from 120
    frames across 12 time bins to 30 across 3, and the model's reliability
    from 0.43 (good) to 0.65 (marginal).

    Any non-numeric field is treated as a candidate trial name, so a row
    carrying both a display name and a canonical key
    ("P7,Trial 1 Hands only,Trial1_handsonly,1,15237,35569") matches on
    either.

    Columns: participant, trial (one or more name columns), then one or more
    individually-verified
    good frame numbers (NOT a start/end range — each number is its own
    spot-checked frame; a row with "1  15237  35569" means exactly those
    three frames are confirmed good, not "1 through 35569"), with any
    trailing non-numeric text ignored as a note. A blank participant cell
    means "same participant as the row above" (the log is written that way
    to avoid repeating it down a block of trials). Matching is case-/
    punctuation-insensitive substring matching on both participant and
    trial, since trial names in the log don't always match the recording
    filename exactly (e.g. typos, "HOI" vs "_hoi").

    Numbers in the log are the Vicon **Frame** number as read off the Nexus
    UI (1-indexed — the raw CSV's own "Frame" column starts at 1). The
    array ``load_csv`` returns is 0-indexed (``markers[0]`` is Frame 1), so
    the returned array is converted here (``frame - 1``) to be directly
    usable as an index into ``markers``. Getting this wrong silently pulls
    the *next* frame instead of the one actually verified — easy to miss,
    since it usually still looks plausible.

    Because these are isolated spot checks rather than a contiguous clean
    span, they rarely include *consecutive* frames — the temporal-speed
    reference in ``build_reference`` will likely end up empty, so the
    per-marker speed check falls back to the fixed ``speed_tol_mm`` floor
    instead of an adaptive per-marker threshold. Add more, or run through
    additional individual frames, if that's not tight enough.
    """
    import csv as csv_mod

    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
    # Pick whichever delimiter actually splits this file into fields.
    delim = "\t" if sample.count("\t") >= sample.count(",") else ","

    def _is_int(c: str) -> bool:
        return c.strip().lstrip("-").isdigit()

    want_p, want_t = _normalise_name(participant), _normalise_name(trial)
    frames: set[int] = set()
    matched_rows = []
    current_p = ""
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv_mod.reader(f, delimiter=delim):
            if not row or not any(c.strip() for c in row):
                continue
            p = (row + [""])[0]
            if p.strip():
                current_p = p.strip()
            if _normalise_name(current_p) != want_p:
                continue
            # Every non-numeric field after the participant is a candidate
            # name for this trial, so a row carrying both a display name and
            # the CSV stem ("P7,Trial 1 Hands only,Trial1_handsonly,...")
            # matches on either. Trailing free-text notes are harmless here:
            # they only ever add a name that fails to match.
            names = [c for c in row[1:] if c.strip() and not _is_int(c)]
            if not any(want_t in _normalise_name(n) or _normalise_name(n) in want_t
                       for n in names if _normalise_name(n)):
                continue
            row_frames = [int(c.strip()) for c in row[1:] if _is_int(c)]
            if not row_frames:
                continue
            frames.update(row_frames)
            matched_rows.append((current_p, " / ".join(n.strip() for n in names), row_frames))

    if not frames:
        raise ValueError(
            f"No rows in {path} matched participant={participant!r} trial={trial!r}. "
            "Check spelling, or pass --ref-frames directly instead."
        )
    print(f"Matched {len(matched_rows)} row(s), {len(frames)} good frame(s) from {path} "
          "(Vicon Frame numbers, 1-indexed):")
    for p, t, row_frames in matched_rows:
        print(f"  {p} / {t}: frames {row_frames}")

    # Convert 1-indexed Vicon Frame numbers to 0-indexed array positions.
    return np.array(sorted(f - 1 for f in frames), dtype=int)


# ---------------------------------------------------------------------------
# manual_frames.csv readers
#
# The log is the authority on which recording is which trial: its "Session"
# column is the trial's identity and its "CSV name" column + ".csv" is the
# file. Shared by visualize_data.py and scripts/relabel_with_mano.py so both
# resolve trials identically — they used to disagree, with the relabeller
# enumerating from trial_filename_map.csv (34 trials) while the cascade used
# the log (66).
# ---------------------------------------------------------------------------


def participant_sort_key(name: str) -> tuple[int, str]:
    """Sort "P2" before "P10" — plain string order puts P10 first, which
    makes the all-participants summary grid read wrong."""
    m = re.fullmatch(r"[A-Za-z]*(\d+)", name.strip())
    return (int(m.group(1)), name) if m else (10**9, name)


def resolve_participants(requested: list[str],
                         manual_sessions: dict[str, dict[str, str]],
                         source: str = "manual_frames.csv") -> list[str]:
    """Expand ["all"] to every participant with rows in manual_frames.csv,
    in numeric order. Any other list is returned as given (order preserved),
    with a warning for names the log doesn't know about — those would
    otherwise just silently produce an empty row in the summary grid."""
    if any(p.strip().lower() == "all" for p in requested):
        found = sorted(manual_sessions, key=participant_sort_key)
        print(f"PARTICIPANTS=all -> {len(found)} participant(s) from "
              f"{source}: {', '.join(found)}")
        return found
    for p in requested:
        if p not in manual_sessions:
            print(f"[warn] {p}: no rows in {source}")
    return list(requested)


def slug(name: str) -> str:
    """Filename-safe form of a Session name ("Trial 1 Hands only" ->
    "Trial_1_Hands_only"), for the per-trial output HTML."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def sniff_delimiter(path: Path) -> str:
    """Tab or comma, whichever actually splits this file into fields.

    manual_frames.csv started life tab-separated and has since been
    re-saved as true CSV; parsing a comma file as TSV collapses every line
    into one field, so nothing matches and every trial silently falls back
    to the AUTO reference. ``label_marker_quality.load_ref_ranges_csv``
    sniffs the same way — keep the two in step.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def is_frame_number(cell: str) -> bool:
    return cell.strip().lstrip("-").isdigit()


# Header cell -> which column it is. Matched on the normalised (alnum-only,
# lowercased) text, so "Partecipant" (the spelling actually in the file)
# and "CSV name" both land correctly.
_HEADER_PARTICIPANT = ("participant", "partecipant")
_HEADER_SESSION = ("session",)
_HEADER_CSVNAME = ("csvname", "filename", "file", "csv")


def load_trial_sessions(path: Path) -> dict[str, dict[str, str]]:
    """Read manual_frames.csv's (participant, session, CSV name) columns.

    The file is a header-row CSV with fixed columns::

        Partecipant,Session,CSV name,Ref frame 1..4,Notes
        P1,Trial 1 Hands only,P101,9952,26304,44016,,subject with missing markers

    Both name columns are taken **verbatim**: ``Session`` is the trial's
    identity (the strings in ``TRIALS``) and ``CSV name`` + ".csv" is its
    file. Neither is normalised or keyword-classified — the log is the
    authority, so a value that doesn't line up is a typo to fix at the
    source, not something to guess around.

    Frame and Notes columns are ignored here, including the rows where a
    note has slid into a frame column (P9's "too noisy with a lot of extra
    markers"). A blank participant cell means "same participant as the row
    above". Columns are located by header name, falling back to positions
    0/1/2 if the header is missing or renamed.

    Returns {participant: {session: csv_name}}. Sessions that appear twice
    for one participant keep the first row and warn; a deliberate retake is
    better given its own distinct Session name (as P17's "Trial2 Hands only
    extra" is), which then simply goes unused unless it's listed in
    ``TRIALS``.
    """
    import csv as csv_mod

    sessions: dict[str, dict[str, str]] = {}
    if not path.exists():
        return sessions
    delim = sniff_delimiter(path)
    i_p, i_session, i_csv = 0, 1, 2
    seen_header = False
    current_p = ""
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv_mod.reader(f, delimiter=delim):
            if not row or not any(c.strip() for c in row):
                continue
            cells = [c.strip() for c in row]
            norm = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in cells]

            if not seen_header and norm and norm[0] in _HEADER_PARTICIPANT:
                seen_header = True
                for i, n in enumerate(norm):
                    if n in _HEADER_PARTICIPANT:
                        i_p = i
                    elif n in _HEADER_SESSION:
                        i_session = i
                    elif n in _HEADER_CSVNAME:
                        i_csv = i
                continue

            def cell(i: int) -> str:
                return cells[i] if i < len(cells) else ""

            if cell(i_p):
                current_p = cell(i_p)
            session, csv_name = cell(i_session), cell(i_csv)
            if not current_p or not session or not csv_name:
                continue
            if is_frame_number(session) or is_frame_number(csv_name):
                continue  # not a real row (stray numbers in the name columns)
            slots = sessions.setdefault(current_p, {})
            if session in slots:
                print(f"[warn] {path.name}: {current_p}/{session!r} appears twice "
                      f"({csv_name!r} vs {slots[session]!r}) — keeping the first")
                continue
            slots[session] = csv_name
    return sessions


def find_session_file(participant_dir: Path, csv_name: str) -> Path | None:
    """Resolve a manual_frames.csv ``CSV name`` cell to its file: the stem
    plus ".csv", nothing cleverer.

    The log's ``CSV name`` column *is* the filename on disk, so no
    normalisation, fuzzy matching or keyword guessing is needed or wanted —
    a mismatch means a genuine typo on one side or the other, and should be
    fixed at the source rather than papered over by a guess that might pick
    the wrong recording. Currently 8 of 66 rows mismatch: P4 Trial1 HOI
    ("Trail1_HOI" logged, "Trial1_HOI.csv" on disk), P10's two Trial1 rows
    (the *files* are typo'd "Trail1_*"), P11 Trial2 hands-only
    ("Trial2_hands_only" vs "Trial2_handsonly.csv") and P12's four
    ("Staic_trial0N" vs "Static_trial0N.csv"). Those trials are skipped
    with a warning until the log or the filenames are corrected.

    Matching is case-sensitive in intent but tolerant of the filesystem's
    own casing: the real directory entry is returned, so a case difference
    doesn't silently produce a Path that only works on Windows.
    """
    if not csv_name:
        return None
    want = f"{csv_name}.csv"
    for csv_path in sorted(participant_dir.glob("*.csv")):
        if csv_path.name == want:
            return csv_path
        if csv_path.name.lower() == want.lower():
            print(f"[info] {participant_dir}: {csv_name!r} matched "
                  f"{csv_path.name!r} (case differs)")
            return csv_path
    return None


def parse_frame_spec(spec: str) -> np.ndarray:
    """Parse "100-200,500,900-950" into a sorted unique 0-indexed int array.

    Like ``load_ref_ranges_csv``, the input is 1-indexed Vicon Frame
    numbers (matching what you'd read off the Nexus UI or ``--ref-frames``
    written by a human); the returned array is converted (``frame - 1``) to
    be directly usable as an index into the ``markers`` array from
    ``load_csv``.
    """
    idx = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            idx.update(range(int(a), int(b) + 1))
        else:
            idx.add(int(part))
    idx = {i - 1 for i in idx}
    return np.array(sorted(idx), dtype=int)



def load_trial_map(path: Path) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv_mod.DictReader(f):
            p = (row.get("participant") or "").strip()
            fn = (row.get("filename") or "").strip()
            key = (row.get("trial_key") or "").strip()
            if p and fn:
                mapping[(p, fn.lower())] = key
    return mapping


def find_trial_csv(participant: str, trial: str
                    ) -> tuple[Path | None, Path | None, Path | None]:
    """Return (trial csv, cascade-static csv, mano-calibration-static csv).

    The two statics are usually the same file; they differ only where a
    static recording is trustworthy for the hand but not the forearm (see
    :data:`MANO_ONLY_STATIC`).

    The trial itself comes from manual_frames.csv: its "Session" column is
    the trial's identity and "CSV name" + ".csv" is the file. This used to
    read trial_filename_map.csv, which only covered 34 of 66 trials -- P1-P6
    and P17 were absent, so ``--all`` silently skipped them. The map remains
    the source for *statics*, which the log does not track.
    """
    pdir = DATA_ROOT / participant
    sessions = load_trial_sessions(REF_CSV).get(participant, {})
    csv_name = sessions.get(trial)
    trial_path = find_session_file(pdir, csv_name) if csv_name else None

    overrides = load_trial_map(TRIAL_MAP_CSV)
    static_path = None
    for c in sorted(pdir.glob("*.csv")):
        if overrides.get((participant, c.name.lower())) == "static":
            static_path = c

    mano_static = static_path
    extra = MANO_ONLY_STATIC.get(participant)
    if extra is not None:
        cand = pdir / extra
        if cand.exists():
            mano_static = cand
    return trial_path, static_path, mano_static


def find_static_csv(participant: str) -> Path | None:
    """The participant's dedicated static/calibration recording, if any --
    the *cascade-safe* one, not the MANO-calibration one (``find_trial_csv``'s
    third return value, used directly by MANO calibration callers).

    Resolved via manual_frames.csv itself: a static recording gets its own
    row there like any other trial, with ``Session`` == "Static" (matched
    case-insensitively via the same normalisation ``find_trial_csv`` uses
    for every other trial name -- no special-casing needed, "Static" is
    just another session name in that column). A participant with no such
    row -- P9, deliberately -- has no cascade-safe static: its static was
    taken after a Forearm marker fell off and was reattached, so its
    forearm geometry no longer matches the trials, and folding it into the
    cascade reference once collapsed 3 of P9's 4 trials to 0% correct. The
    log not carrying a "Static" row for P9 *is* that exclusion now, rather
    than a separate hardcoded list.

    This used to read ``trial_filename_map.csv``'s "static" entries, a
    second file tracking exactly the same fact by filename instead of by
    manual_frames.csv row -- redundant, and it no longer exists at the
    current ``DATA_ROOT`` at all, so that path always returned ``None``.
    """
    trial_path, _static_path, _mano_static = find_trial_csv(participant, "Static")
    return trial_path


def auto_reference(markers: np.ndarray, need: int = 300) -> np.ndarray:
    """Fallback reference frames: the longest fully-present stretches."""
    present = np.isfinite(markers).all(axis=(1, 2))
    runs, start = [], None
    for t, ok in enumerate(present):
        if ok and start is None:
            start = t
        elif not ok and start is not None:
            runs.append((start, t)); start = None
    if start is not None:
        runs.append((start, len(present)))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    idx: list[int] = []
    for a, b in runs:
        idx.extend(range(a, b))
        if len(idx) >= need:
            break
    return np.array(sorted(idx), dtype=int)
