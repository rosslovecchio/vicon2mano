"""Debug-friendly script version of visualize_data.ipynb.

Same logic as the notebook, but as a plain script split into ``# %%`` cells
(VS Code's Python extension runs/debugs each one individually — "Run Cell" /
"Debug Cell" appear above each ``# %%`` marker) so you can set breakpoints
and step through with the regular debugger instead of a notebook kernel.

Run the whole thing with F5 (or `python visualize_data.py`), or run/debug
cell by cell from the top.
"""

# %% Setup
import re
import sys
import webbrowser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vicon2mano.loader import load_csv
import label_marker_quality as lmq

# %% Configuration
DATA_ROOT = Path(r"D:\ExperimentsJune25")
REF_CSV = DATA_ROOT / "manual_frames.csv"
TRIAL_MAP_CSV = DATA_ROOT / "trial_filename_map.csv"
# Either an explicit list, or ["all"] for every participant that has rows
# in manual_frames.csv (resolved by _resolve_participants below).
PARTICIPANTS = ["all"]  # e.g. ["P7", "P8"] to narrow it down
# Trials are named exactly as manual_frames.csv's "Session" column spells
# them — that column is the trial's identity, so no canonicalisation or
# keyword guessing is involved anywhere. Note the inconsistent spacing
# ("Trial 1" vs "Trial2") is the file's, and is deliberately preserved.
TRIALS = ["Trial 1 Hands only", "Trial 1 HOI", "Trial2 Hands only", "Trial2 HOI"]

# Everything this script produces (per-trial animation HTML, the summary
# plot, the before/after statistics) lands here. "cascade_only" = the
# marker-quality cascade's own output, with no MANO fitting involved.
OUT_DIR = REPO_ROOT / "results" / "cascade_only"


def load_trial_filename_map(path: Path) -> dict[tuple[str, str], str]:
    """Read trial_filename_map.csv: participant,filename,trial_key.

    An explicit, hand-editable override for find_trial_files' keyword
    guess — e.g. a filename the "1"/"2" + "hoi"/"hand"/"only"/"static"
    heuristic gets wrong. Blank trial_key means "not a trial at all"
    (excluded — a recording that isn't a real trial or a usable static
    reference, e.g. an aborted take).
    """
    import csv as csv_mod

    mapping: dict[tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv_mod.DictReader(f):
            participant = (row.get("participant") or "").strip()
            filename = (row.get("filename") or "").strip()
            trial_key = (row.get("trial_key") or "").strip()
            if participant and filename:
                mapping[(participant, filename.lower())] = trial_key
    return mapping


def _classify_trial_name(name: str) -> str | None:
    """Guess a canonical trial key ("Trial1_handsonly", ..., or "static")
    from a filename stem or a manual_frames.csv trial label — trial number
    "1"/"2" + "hoi"/"hand"/"only" in the text, or "static"/"staic" (a
    recurring typo) for a calibration recording. Returns None if nothing
    recognisable is present."""
    low = name.lower()
    if "static" in low or "staic" in low:
        return "static"
    if "1" in low or "2" in low:
        trial_num = "1" if "1" in low else "2"
        if "hoi" in low:
            kind = "hoi"
        elif "hand" in low or "only" in low:
            kind = "handsonly"
        else:
            return None
        return f"Trial{trial_num}_{kind}"
    return None


def find_trial_files(participant: str, participant_dir: Path,
                      overrides: dict[tuple[str, str], str] | None = None) -> dict[str, Path]:
    """Map canonical trial keys ("Trial1_handsonly", ..., and "static" for a
    calibration/static recording if present) to actual CSV files in a
    participant's directory.

    Filenames aren't consistent across participants (typos like
    "Trail1_HOI", underscore variants like "Trial1_hands_only", mixed
    case). ``overrides`` (from trial_filename_map.csv) is consulted first
    for an explicit assignment; any file not listed there falls back to
    ``_classify_trial_name``'s keyword guess. Newer batches use filenames
    with no recognisable keywords at all — add explicit rows to
    trial_filename_map.csv for those rather than relying on the guess.
    When two files classify to the same key, the first (filename-sorted)
    wins and the rest are dropped with a warning.
    """
    overrides = overrides or {}
    found: dict[str, Path] = {}
    for csv_path in sorted(participant_dir.glob("*.csv")):
        key = overrides.get((participant, csv_path.name.lower()))
        if key is None:
            key = _classify_trial_name(csv_path.stem)
            if key is None:
                continue
        elif key == "":
            continue  # explicitly marked "not a trial" in the map
        if key in found:
            print(f"[warn] {participant_dir}: both {found[key].name} and {csv_path.name} "
                  f"match {key!r} — keeping {found[key].name}")
            continue
        found[key] = csv_path
    return found


# The manual_frames.csv readers live in label_marker_quality so
# relabel_with_mano.py uses exactly the same trial resolution; aliased here
# so the cells below read as before.
_participant_sort_key = lmq.participant_sort_key
_slug = lmq.slug
_sniff_delimiter = lmq.sniff_delimiter
load_manual_trial_sessions = lmq.load_trial_sessions
find_session_file = lmq.find_session_file


def _resolve_participants(requested, manual_sessions):
    return lmq.resolve_participants(requested, manual_sessions, source=REF_CSV.name)


def auto_reference(markers, min_frames=300):
    """Fallback when a trial has no row in manual_frames.csv yet: the
    longest run(s) of frames where every marker is present. Unverified --
    prefer adding a real reference range to manual_frames.csv instead."""
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
    idx = []
    for s, e in runs:
        idx.extend(range(s, e))
        if len(idx) >= min_frames:
            break
    return np.array(sorted(idx), dtype=int)


# %% Before/after statistics - definitions
# "Before" is the raw Vicon export as loaded: a marker sample counts if
# its XYZ is finite (i.e. Nexus labelled it in that frame). "After" is
# what survives the cascade - the samples it positively confirms as
# correct. The gap is what screening costs you, and is the number to
# look at before feeding a trial into a MANO fit. Defined up here so the
# load cell below can fill stats_by_trial as it goes.
import csv as _csv

STATUS_MISSING, STATUS_INCORRECT, STATUS_CORRECT = 0, 1, 2

_STAT_FIELDS = [
    "participant", "session", "csv_file", "n_frames", "n_markers", "n_samples",
    "ref_frames",
    # before
    "before_pct_present", "before_frames_full", "before_pct_frames_full",
    # after
    "after_pct_correct", "after_frames_full", "after_pct_frames_full",
    "after_pct_incorrect", "after_pct_missing",
    # before -> after
    "pct_present_rejected", "delta_pct_samples", "delta_pct_frames_full",
]


def trial_stats(participant, session, csv_file, markers, status, n_ref):
    """One row of before/after numbers for a single trial."""
    present = np.isfinite(markers).all(axis=-1)      # (T, M) raw availability
    correct = status == STATUS_CORRECT
    n_frames, n_markers = present.shape
    n_samples = present.size

    before_pct = present.sum() / n_samples * 100
    after_pct = correct.sum() / n_samples * 100
    before_full = int(present.all(axis=1).sum())     # frames with every marker labelled
    after_full = int(correct.all(axis=1).sum())      # frames the cascade fully vouches for
    n_present = int(present.sum())

    return {
        "participant": participant, "session": session, "csv_file": csv_file,
        "n_frames": n_frames, "n_markers": n_markers, "n_samples": n_samples,
        "ref_frames": n_ref,
        "before_pct_present": round(before_pct, 2),
        "before_frames_full": before_full,
        "before_pct_frames_full": round(before_full / n_frames * 100, 2),
        "after_pct_correct": round(after_pct, 2),
        "after_frames_full": after_full,
        "after_pct_frames_full": round(after_full / n_frames * 100, 2),
        "after_pct_incorrect": round((status == STATUS_INCORRECT).sum() / n_samples * 100, 2),
        "after_pct_missing": round((status == STATUS_MISSING).sum() / n_samples * 100, 2),
        # Of the samples Vicon *did* label, how many the cascade refuses to
        # vouch for - the headline "cost of screening" number.
        "pct_present_rejected": (round((n_present - int(correct.sum())) / n_present * 100, 2)
                                 if n_present else 0.0),
        "delta_pct_samples": round(after_pct - before_pct, 2),
        "delta_pct_frames_full": round((after_full - before_full) / n_frames * 100, 2),
    }


# %% Load + label every trial
# Each trial's result is kept in results[(participant, trial)] as
# (markers, labels, bones, status, pct_correct). Set a breakpoint inside
# label_marker_quality.label_quality_cascade to step through the cascade
# stage by stage for a specific trial.
results = {}
stats_by_trial = {}
trial_map = load_trial_filename_map(TRIAL_MAP_CSV)
manual_sessions = load_manual_trial_sessions(REF_CSV)
# Resolved once here; the summary/animation cells below use this rather
# than PARTICIPANTS, so ["all"] works when running cell by cell too.
participants = _resolve_participants(PARTICIPANTS, manual_sessions)

for participant in participants:
    participant_dir = DATA_ROOT / participant / participant
    # manual_frames.csv's CSV-name column is the only source for trial
    # slots — its cell + ".csv" is the filename. find_trial_files' keyword
    # guess is used *only* to locate the extra "static" bone-length
    # reference, which manual_frames.csv doesn't track; deliberately not as
    # a fallback for a trial whose CSV name doesn't resolve, since a guess
    # there would silently paper over the typo instead of surfacing it.
    keyword_files = find_trial_files(participant, participant_dir, trial_map)
    sessions = manual_sessions.get(participant, {})
    trial_files: dict[str, Path] = {}
    if "static" in keyword_files:
        trial_files["static"] = keyword_files["static"]
    for session, csv_name in sessions.items():
        session_path = find_session_file(participant_dir, csv_name)
        if session_path is None:
            print(f"[warn] {participant}/{session!r}: manual_frames.csv CSV name "
                  f"{csv_name!r} -> {csv_name}.csv not found in {participant_dir} "
                  f"— fix the typo in manual_frames.csv or rename the file")
            continue
        trial_files[session] = session_path

    static_path = trial_files.get("static")
    static_markers_raw, static_labels = (None, None)
    if static_path is not None:
        static_markers_raw, static_labels = load_csv(str(static_path))
        print(f"{participant}: using {static_path.name} ({static_markers_raw.shape[0]} frames) "
              f"as an extra bone-length reference")

    for trial in TRIALS:
        path = trial_files.get(trial)
        if path is None:
            print(f"[skip] {participant}/{trial}: no matching CSV in {participant_dir}")
            continue

        markers, labels = load_csv(str(path))

        static_markers = None
        if static_markers_raw is not None:
            static_markers = lmq.align_markers_to_labels(static_markers_raw, static_labels, labels)

        try:
            ref_frames = lmq.load_ref_ranges_csv(str(REF_CSV), participant, trial)
            ref_source = "manual_frames.csv"
        except ValueError:
            ref_frames = auto_reference(markers)
            ref_source = "AUTO (unverified — add a row to manual_frames.csv)"
        ref_frames = ref_frames[(ref_frames >= 0) & (ref_frames < markers.shape[0])]

        status, bones = lmq.label_quality_cascade(markers, labels, ref_frames, static_markers=static_markers)
        pct_correct = (status == 2).sum(axis=1) / status.shape[1] * 100

        results[(participant, trial)] = (markers, labels, bones, status, pct_correct)
        stats_by_trial[(participant, trial)] = trial_stats(
            participant, trial, path.name, markers, status, len(ref_frames))
        print(f"{participant}/{trial}: {markers.shape[0]} frames, "
              f"ref={ref_source} ({len(ref_frames)} frames), "
              f"mean correct {pct_correct.mean():.1f}% "
              f"(min {pct_correct.min():.0f}%, max {pct_correct.max():.0f}%)")

# %% Summary: % correct over time, all trials
fig, axes = plt.subplots(len(participants), len(TRIALS), figsize=(4 * len(TRIALS), 3 * len(participants)),
                          sharey=True, squeeze=False)
for row, participant in enumerate(participants):
    for col, trial in enumerate(TRIALS):
        ax = axes[row, col]
        key = (participant, trial)
        if key not in results:
            ax.axis("off")
            continue
        _, _, _, _, pct_correct = results[key]
        ax.plot(pct_correct, color="#2980b9", linewidth=0.7)
        ax.set_title(f"{participant}/{trial}\nmean {pct_correct.mean():.0f}%", fontsize=9)
        ax.set_ylim(0, 100)
        if col == 0:
            ax.set_ylabel("% correct")
fig.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary_png = OUT_DIR / "quality_summary.png"
fig.savefig(summary_png, dpi=120)
print(f"Saved summary plot to {summary_png}")
plt.show()  # blocks in a plain terminal; close the window to continue

# %% Before/after statistics -> results/cascade_only/*.csv
OUT_DIR.mkdir(parents=True, exist_ok=True)
stats_rows = [stats_by_trial[k] for k in results]

if stats_rows:
    # Pooled totals, weighted by sample/frame counts rather than a mean of
    # means - trials differ in length by an order of magnitude.
    tot_samples = sum(r["n_samples"] for r in stats_rows)
    tot_frames = sum(r["n_frames"] for r in stats_rows)
    overall = {f: "" for f in _STAT_FIELDS}
    overall.update({
        "participant": "ALL", "session": f"{len(stats_rows)} trial(s)", "csv_file": "",
        "n_frames": tot_frames, "n_markers": "", "n_samples": tot_samples,
        "ref_frames": sum(r["ref_frames"] for r in stats_rows),
        "before_pct_present": round(
            sum(r["before_pct_present"] * r["n_samples"] for r in stats_rows) / tot_samples, 2),
        "after_pct_correct": round(
            sum(r["after_pct_correct"] * r["n_samples"] for r in stats_rows) / tot_samples, 2),
        "before_frames_full": sum(r["before_frames_full"] for r in stats_rows),
        "after_frames_full": sum(r["after_frames_full"] for r in stats_rows),
    })
    overall["before_pct_frames_full"] = round(overall["before_frames_full"] / tot_frames * 100, 2)
    overall["after_pct_frames_full"] = round(overall["after_frames_full"] / tot_frames * 100, 2)
    overall["delta_pct_samples"] = round(
        overall["after_pct_correct"] - overall["before_pct_present"], 2)
    overall["delta_pct_frames_full"] = round(
        overall["after_pct_frames_full"] - overall["before_pct_frames_full"], 2)

    stats_csv = OUT_DIR / "before_after_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_STAT_FIELDS)
        w.writeheader()
        w.writerows(stats_rows)
        w.writerow(overall)
    print(f"Saved before/after statistics to {stats_csv}")

    # Per-marker view, pooled over every trial: which markers the cascade
    # rejects most is usually where a label swap or a bad plate lives.
    marker_totals: dict[str, list[int]] = {}
    for (participant, trial), (markers, labels, _, status, _) in results.items():
        present = np.isfinite(markers).all(axis=-1)
        for m, label in enumerate(labels):
            # Vicon labels carry a subject prefix ("P14:Ring1"); strip it so
            # the same anatomical marker pools across participants instead of
            # producing one row per person.
            base = label.split(":")[-1]
            acc = marker_totals.setdefault(base, [0, 0, 0])  # samples, present, correct
            acc[0] += present.shape[0]
            acc[1] += int(present[:, m].sum())
            acc[2] += int((status[:, m] == STATUS_CORRECT).sum())
    marker_csv = OUT_DIR / "before_after_by_marker.csv"
    with marker_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["marker", "n_samples", "before_pct_present", "after_pct_correct",
                    "pct_present_rejected"])
        for label, (n, n_present, n_correct) in sorted(
                marker_totals.items(),
                key=lambda kv: (kv[1][1] - kv[1][2]) / kv[1][1] if kv[1][1] else 0,
                reverse=True):
            w.writerow([
                label, n,
                round(n_present / n * 100, 2),
                round(n_correct / n * 100, 2),
                round((n_present - n_correct) / n_present * 100, 2) if n_present else 0.0,
            ])
    print(f"Saved per-marker breakdown to {marker_csv}")

    # Paired before/after bars, one pair per trial.
    labels_x = [f"{r['participant']}\n{r['session']}" for r in stats_rows]
    xs = np.arange(len(stats_rows))
    fig_ba, ax_ba = plt.subplots(figsize=(max(6, 0.7 * len(stats_rows) + 2), 4.5))
    ax_ba.bar(xs - 0.2, [r["before_pct_present"] for r in stats_rows], 0.4,
              label="before (labelled by Vicon)", color="#95a5a6")
    ax_ba.bar(xs + 0.2, [r["after_pct_correct"] for r in stats_rows], 0.4,
              label="after (confirmed by cascade)", color="#2980b9")
    ax_ba.set_xticks(xs)
    ax_ba.set_xticklabels(labels_x, rotation=90, fontsize=7)
    ax_ba.set_ylabel("% of marker samples")
    ax_ba.set_ylim(0, 100)
    ax_ba.set_title("Marker samples before vs after the quality cascade")
    ax_ba.legend(fontsize=8)
    fig_ba.tight_layout()
    ba_png = OUT_DIR / "before_after.png"
    fig_ba.savefig(ba_png, dpi=120)
    print(f"Saved before/after plot to {ba_png}")
    print(f"OVERALL: {overall['before_pct_present']}% of samples labelled -> "
          f"{overall['after_pct_correct']}% confirmed correct "
          f"({overall['delta_pct_samples']} pp); fully-clean frames "
          f"{overall['before_pct_frames_full']}% -> {overall['after_pct_frames_full']}%")
else:
    print("No trials loaded - nothing to summarise.")


# %% Animation setup (helpers)
N_OUT = 1000

# Original marker colors, by finger.
FINGER_PALETTE = {
    "thumb": "#e74c3c", "index": "#27ae60", "middle": "#2980b9",
    "ring": "#e67e22", "pinky": "#8e44ad", "palm": "#444444",
    "wrist": "#444444", "forearm": "#95a5a6",
}
STATUS_OUTLINE = {0: "rgba(0,0,0,0)", 1: "#e74c3c", 2: "#2ecc71"}  # missing, incorrect, correct


def marker_color(label):
    low = label.lower()
    for key, color in FINGER_PALETTE.items():
        if key in low:
            return color
    return "#000000"


def marker_digit(label):
    """Trailing digit(s) of a marker label ("Thumb1" -> "1"), for a compact
    on-plot annotation next to each marker dot."""
    m = re.search(r"(\d+)$", label.strip())
    return m.group(1) if m else ""


def marker_trace_data(markers, status, t):
    fm = markers[t]
    present = np.isfinite(fm).all(axis=-1)
    x = np.where(present, fm[:, 0], np.nan)
    y = np.where(present, fm[:, 1], np.nan)
    z = np.where(present, fm[:, 2], np.nan)
    outline = [STATUS_OUTLINE[int(s)] for s in status[t]]
    return x, y, z, outline


def bone_lines(frame_markers, bones):
    """(x, y, z) with None gaps so all bone segments render as one trace."""
    xs, ys, zs = [], [], []
    for i, j, _ in bones:
        if np.isfinite(frame_markers[i]).all() and np.isfinite(frame_markers[j]).all():
            xs += [frame_markers[i, 0], frame_markers[j, 0], None]
            ys += [frame_markers[i, 1], frame_markers[j, 1], None]
            zs += [frame_markers[i, 2], frame_markers[j, 2], None]
    return xs, ys, zs


# %% Animate one trial (interactive 3D) — helpers
# Plotly's slider already lets you scrub to a frame, but there's no
# single-frame step or numeric "go to frame" control — add both via a
# post-render script that drives the same Plotly.animate() call the slider
# uses, keyed by the frame's `name` (== str(k) into frame_idx).
_STEP_CONTROLS_JS = """
(function() {
  var gd = document.getElementById('%(div_id)s');
  var nFrames = %(n_frames)d;
  var cur = 0;
  function goToFrame(k) {
    cur = ((k %% nFrames) + nFrames) %% nFrames;
    Plotly.animate(gd, [String(cur)],
      {frame: {duration: 0, redraw: true}, transition: {duration: 0}, mode: 'immediate'});
  }
  gd.on('plotly_animatingframe', function(e) { cur = parseInt(e.name, 10); });

  var bar = document.createElement('div');
  bar.style = 'margin-top:8px;display:flex;gap:8px;align-items:center;font-family:sans-serif;';
  bar.innerHTML =
    '<button id="prevFrameBtn">&#9198; Prev</button>' +
    '<button id="nextFrameBtn">Next &#9197;</button>' +
    '<input id="gotoFrameInput" type="number" min="1" max="' + nFrames +
    '" placeholder="frame # (1-' + nFrames + ')" style="width:130px">' +
    '<button id="gotoFrameBtn">Go</button>';
  gd.parentNode.insertBefore(bar, gd.nextSibling);

  document.getElementById('prevFrameBtn').onclick = function() { goToFrame(cur - 1); };
  document.getElementById('nextFrameBtn').onclick = function() { goToFrame(cur + 1); };
  function doGoto() {
    var v = parseInt(document.getElementById('gotoFrameInput').value, 10);
    if (!isNaN(v)) goToFrame(v - 1);
  }
  document.getElementById('gotoFrameBtn').onclick = doGoto;
  document.getElementById('gotoFrameInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') doGoto();
  });
  document.addEventListener('keydown', function(e) {
    if (document.activeElement && document.activeElement.id === 'gotoFrameInput') return;
    if (e.key === 'ArrowRight') goToFrame(cur + 1);
    else if (e.key === 'ArrowLeft') goToFrame(cur - 1);
  });
})();
"""

def build_animation_figure(participant, trial, markers, labels, bones, status, pct_correct):
    """Build the interactive 3D Plotly animation figure for one trial."""
    frame_idx = np.unique(np.linspace(0, markers.shape[0] - 1, N_OUT).astype(int))
    center = np.nanmean(markers, axis=(0, 1))
    half = 200  # mm
    fill_colors = [marker_color(l) for l in labels]
    digit_text = [marker_digit(l) for l in labels]
    n_markers = markers.shape[1]

    def frame_title(t):
        n_present = int(np.isfinite(markers[t]).all(axis=-1).sum())
        return (f"{participant}/{trial}  frame {t}  correct {pct_correct[t]:.0f}%  "
                f"markers {n_present}/{n_markers}")

    x0, y0, z0, outline0 = marker_trace_data(markers, status, frame_idx[0])
    bx0, by0, bz0 = bone_lines(markers[frame_idx[0]], bones)

    # Plotly's Scatter3d marker outline (`line=`) renders as a hairline in
    # WebGL regardless of `width` — draw the status ring as a separate,
    # larger marker layer behind the fill layer instead, which is reliably
    # visible.
    ring_trace = go.Scatter3d(
        x=x0, y=y0, z=z0, mode="markers",
        marker=dict(size=14, color=outline0), hoverinfo="skip", name="status",
    )
    marker_trace = go.Scatter3d(
        x=x0, y=y0, z=z0, mode="markers+text",
        marker=dict(size=6, color=fill_colors),
        text=digit_text, textposition="top center",
        textfont=dict(size=10, color="#000000"),
        hovertext=labels, hoverinfo="text", name="markers",
    )
    bone_trace = go.Scatter3d(
        x=bx0, y=by0, z=bz0, mode="lines",
        line=dict(color="lightgray", width=2), hoverinfo="skip", name="bones",
    )

    # Legend proxy traces: no real data, just a swatch + label per finger
    # group and per quality status, since Scatter3d doesn't expose a
    # discrete-color legend the way px does.
    finger_legend_traces = [
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=6, color=color), name=finger,
            legendgroup="finger", legendgrouptitle=dict(text="Finger"),
        )
        for finger, color in FINGER_PALETTE.items()
    ]
    status_legend_traces = [
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=color), name=label,
            legendgroup="status", legendgrouptitle=dict(text="Status"),
        )
        for label, color in [("correct", STATUS_OUTLINE[2]), ("incorrect", STATUS_OUTLINE[1])]
    ]

    frames = []
    for k, t in enumerate(frame_idx):
        x, y, z, outline = marker_trace_data(markers, status, t)
        bx, by, bz = bone_lines(markers[t], bones)
        frames.append(go.Frame(
            name=str(k),
            data=[
                go.Scatter3d(x=x, y=y, z=z, marker=dict(color=outline)),
                go.Scatter3d(x=x, y=y, z=z, marker=dict(color=fill_colors)),
                go.Scatter3d(x=bx, y=by, z=bz),
            ],
            layout=go.Layout(title=frame_title(t)),
        ))

    fig = go.Figure(
        data=[ring_trace, marker_trace, bone_trace, *finger_legend_traces, *status_legend_traces],
        frames=frames,
        layout=go.Layout(
            title=frame_title(frame_idx[0]),
            width=950, height=800,
            legend=dict(x=1.02, y=1),
            scene=dict(
                xaxis=dict(range=[center[0] - half, center[0] + half], title="X (mm)"),
                yaxis=dict(range=[center[1] - half, center[1] + half], title="Y (mm)"),
                zaxis=dict(range=[center[2] - half, center[2] + half], title="Z (mm)"),
                aspectmode="cube",
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[
                        None, {"frame": {"duration": 60, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[
                        [None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                currentvalue=dict(prefix="frame: "),
                steps=[
                    dict(method="animate", label=str(t),
                         args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                    for k, t in enumerate(frame_idx)
                ],
            )],
        ),
    )
    return fig, len(frame_idx)


def save_animation_html(fig, n_frames, out_path):
    _div_id = "animfig"
    fig.write_html(
        str(out_path), include_plotlyjs=True, div_id=_div_id,
        post_script=_STEP_CONTROLS_JS % {"div_id": _div_id, "n_frames": n_frames},
    )
    print(f"Saved interactive animation to {out_path}")


# %% Animate every trial (interactive 3D) — one HTML file per (participant, trial)
OUT_DIR.mkdir(parents=True, exist_ok=True)
animation_paths = []
for (participant, trial), (markers, labels, bones, status, pct_correct) in results.items():
    fig, n_frames = build_animation_figure(participant, trial, markers, labels, bones, status, pct_correct)
    anim_html_path = OUT_DIR / f"animation_{participant}_{_slug(trial)}.html"
    save_animation_html(fig, n_frames, anim_html_path)
    animation_paths.append(anim_html_path)

if animation_paths:
    webbrowser.open(animation_paths[0].as_uri())
