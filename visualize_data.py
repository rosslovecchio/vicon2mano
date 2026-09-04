"""Debug-friendly script version of visualize_data.ipynb.

Same logic as the notebook, but as a plain script split into ``# %%`` cells
(VS Code's Python extension runs/debugs each one individually — "Run Cell" /
"Debug Cell" appear above each ``# %%`` marker) so you can set breakpoints
and step through with the regular debugger instead of a notebook kernel.

Run the whole thing with F5 (or `python visualize_data.py`), or run/debug
cell by cell from the top.
"""

# %% Setup
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
PARTICIPANTS = ["P7",]# "P8", "P9", "P10", "P11"]
TRIALS = ["Trial1_handsonly", "Trial1_hoi", "Trial2_handsonly", "Trial2_hoi"]

OUT_DIR = REPO_ROOT / "scripts" / "debug_out"


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


def find_trial_files(participant: str, participant_dir: Path,
                      overrides: dict[tuple[str, str], str] | None = None) -> dict[str, Path]:
    """Map canonical trial keys ("Trial1_handsonly", ..., and "static" for a
    calibration/static recording if present) to actual CSV files in a
    participant's directory.

    Filenames aren't consistent across participants (typos like
    "Trail1_HOI", underscore variants like "Trial1_hands_only", mixed
    case). ``overrides`` (from trial_filename_map.csv) is consulted first
    for an explicit assignment; any file not listed there falls back to a
    keyword guess (trial number "1"/"2" + "hoi"/"hand"/"only" in the name,
    or "static" in the name for a calibration recording).
    """
    overrides = overrides or {}
    found: dict[str, Path] = {}
    for csv_path in sorted(participant_dir.glob("*.csv")):
        key = overrides.get((participant, csv_path.name.lower()))
        if key is None:
            # Not in the map at all (a new/unseen file) — fall back to guessing.
            name = csv_path.stem.lower()
            if "static" in name:
                key = "static"
            elif "1" in name or "2" in name:
                trial_num = "1" if "1" in name else "2"
                if "hoi" in name:
                    kind = "hoi"
                elif "hand" in name or "only" in name:
                    kind = "handsonly"
                else:
                    continue
                key = f"Trial{trial_num}_{kind}"
            else:
                continue
        elif key == "":
            continue  # explicitly marked "not a trial" in the map
        if key in found:
            print(f"[warn] {participant_dir}: both {found[key].name} and {csv_path.name} "
                  f"match {key!r} — keeping {found[key].name}")
            continue
        found[key] = csv_path
    return found


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


# %% Load + label every trial
# Each trial's result is kept in results[(participant, trial)] as
# (markers, labels, bones, status, pct_correct). Set a breakpoint inside
# label_marker_quality.label_quality_cascade to step through the cascade
# stage by stage for a specific trial.
results = {}
trial_map = load_trial_filename_map(TRIAL_MAP_CSV)

for participant in PARTICIPANTS:
    participant_dir = DATA_ROOT / participant / participant
    trial_files = find_trial_files(participant, participant_dir, trial_map)

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
        print(f"{participant}/{trial}: {markers.shape[0]} frames, "
              f"ref={ref_source} ({len(ref_frames)} frames), "
              f"mean correct {pct_correct.mean():.1f}% "
              f"(min {pct_correct.min():.0f}%, max {pct_correct.max():.0f}%)")

# %% Summary: % correct over time, all trials
fig, axes = plt.subplots(len(PARTICIPANTS), len(TRIALS), figsize=(4 * len(TRIALS), 3 * len(PARTICIPANTS)),
                          sharey=True, squeeze=False)
for row, participant in enumerate(PARTICIPANTS):
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

# %% Animation setup (helpers)
ANIMATE_TRIAL = ("P7", "Trial1_hoi")
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


# %% Animate one trial (interactive 3D)
markers, labels, bones, status, pct_correct = results[ANIMATE_TRIAL]
participant, trial = ANIMATE_TRIAL

frame_idx = np.unique(np.linspace(0, markers.shape[0] - 1, N_OUT).astype(int))
center = np.nanmean(markers, axis=(0, 1))
half = 200  # mm
fill_colors = [marker_color(l) for l in labels]


def frame_title(t):
    return f"{participant}/{trial}  frame {t}  correct {pct_correct[t]:.0f}%"


x0, y0, z0, outline0 = marker_trace_data(markers, status, frame_idx[0])
bx0, by0, bz0 = bone_lines(markers[frame_idx[0]], bones)

# Plotly's Scatter3d marker outline (`line=`) renders as a hairline in
# WebGL regardless of `width` — draw the status ring as a separate, larger
# marker layer behind the fill layer instead, which is reliably visible.
ring_trace = go.Scatter3d(
    x=x0, y=y0, z=z0, mode="markers",
    marker=dict(size=14, color=outline0), hoverinfo="skip", name="status",
)
marker_trace = go.Scatter3d(
    x=x0, y=y0, z=z0, mode="markers",
    marker=dict(size=6, color=fill_colors),
    text=labels, hoverinfo="text", name="markers",
)
bone_trace = go.Scatter3d(
    x=bx0, y=by0, z=bz0, mode="lines",
    line=dict(color="lightgray", width=2), hoverinfo="skip", name="bones",
)

# Legend proxy traces: no real data, just a swatch + label per finger group
# and per quality status, since Scatter3d doesn't expose a discrete-color
# legend the way px does.
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

anim_html_path = OUT_DIR / f"animation_{participant}_{trial}.html"
_div_id = "animfig"
fig.write_html(
    str(anim_html_path), include_plotlyjs=True, div_id=_div_id,
    post_script=_STEP_CONTROLS_JS % {"div_id": _div_id, "n_frames": len(frame_idx)},
)
print(f"Saved interactive animation to {anim_html_path}")
webbrowser.open(anim_html_path.as_uri())
