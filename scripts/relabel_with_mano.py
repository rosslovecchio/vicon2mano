#!/usr/bin/env python3
"""Relabel incorrectly-labelled markers with a subject-specific MANO model.

Pipeline, for one (participant, trial):

1. Run the quality cascade (``label_marker_quality.label_quality_cascade``)
   to get a per-frame, per-marker correct/incorrect/missing verdict.
2. Calibrate on frames the cascade trusts: fit MANO to estimate the
   subject's shape (``betas``), then each marker's constant offset from its
   joint in that joint's own bone-segment frame.
3. For every frame at or above ``--min-correct-pct``, refit the pose using
   **only the markers the cascade marked correct** (the rest are NaN'd out,
   and ``MANOFitter`` masks non-finite targets out of its loss), predict
   where each marker should be, and reassign the flagged markers' positions
   to the labels they actually fit, via a small assignment problem.
4. Write an interactive animation where relabelled markers are ringed blue,
   correct ones stay green, and still-incorrect ones stay red.

Scope: only reassigns labels among markers that are **present**. A missing
marker is never invented — gap filling is a separate problem with a
different error budget.

Result on P7/Trial1_handsonly (2026-09-07)
-----------------------------------------
All 8254 frames at >=50% cascade-correct, 22398 of 46011 flagged
marker-instances reassigned. Re-running the cascade on the result:

    correct before 73.9%  ->  after 86.2%   (+22375 marker-instances)
    frames improved 3102, worsened 15, unchanged 5137
    of the 22398 markers moved, 19212 (85.8%) became cascade-CORRECT
    markers that had been CORRECT and got moved: 0

Confirmed independently of the MANO model (bone lengths are marker-to-marker
distances against canonical values from the cascade's own reference frames,
so they cannot be gamed by fitting to the model):

    bones touching a relabelled marker: 15.64mm -> 6.59mm error, 92.9% better
    whole frame, all bones:              6.74mm -> 3.38mm error, 92.9% better

**Repair every qualifying frame, not a sampled subset.** An earlier run
repaired only the ~175 frames the animation samples, which sit ~48 frames
apart. Each repaired frame was then an island among unrepaired neighbours, so
a correctly relabelled marker appeared to teleport ~30-40mm between adjacent
frames, tripped the cascade's 12.5mm/frame speed check, and got re-flagged:
0 of 494 moved markers came back CORRECT even though their bone-length error
had dropped from 8.50mm to 1.12mm in the frame checked by hand. The
relabelling was right and the verification was wrong. Contiguous coverage is
what makes the cascade a valid oracle here.

Usage
-----
python scripts/relabel_with_mano.py --participant P7 --trial Trial1_handsonly
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.loader import load_csv                      # noqa: E402
from vicon2mano import mano_relabel as mr                   # noqa: E402
import label_marker_quality as lmq                          # noqa: E402

DATA_ROOT = Path(r"D:\ExperimentsJune25")
REF_CSV = DATA_ROOT / "manual_frames.csv"
TRIAL_MAP_CSV = DATA_ROOT / "trial_filename_map.csv"
MANO_DIR = REPO_ROOT.parent / "clean_kinematics" / "mano_v1_2" / "models"
OUT_DIR = REPO_ROOT / "scripts" / "debug_out"

# status codes: cascade uses 0/1/2; 3 is added here for "relabelled by MANO"
MISSING, INCORRECT, CORRECT, RELABELLED = 0, 1, 2, 3

FINGER_PALETTE = {
    "thumb": "#e74c3c", "index": "#27ae60", "middle": "#2980b9",
    "ring": "#e67e22", "pinky": "#8e44ad", "palm": "#444444",
    "forearm": "#95a5a6",
}
STATUS_OUTLINE = {
    MISSING: "rgba(0,0,0,0)",
    INCORRECT: "#e74c3c",     # red   — still wrong
    CORRECT: "#2ecc71",       # green — cascade-verified, untouched
    RELABELLED: "#2e86ff",    # blue  — reassigned by the MANO model
}


def marker_color(label: str) -> str:
    low = label.lower()
    for key, color in FINGER_PALETTE.items():
        if key in low:
            return color
    return "#000000"


def marker_digit(label: str) -> str:
    for ch in reversed(label):
        if ch.isdigit():
            return ch
    return ""


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


def find_trial_csv(participant: str, trial: str) -> tuple[Path | None, Path | None]:
    """Return (trial csv, static csv) for a participant."""
    pdir = DATA_ROOT / participant / participant
    overrides = load_trial_map(TRIAL_MAP_CSV)
    trial_path = static_path = None
    for c in sorted(pdir.glob("*.csv")):
        key = overrides.get((participant, c.name.lower()))
        if key == trial:
            trial_path = c
        elif key == "static":
            static_path = c
    return trial_path, static_path


def auto_reference(markers: np.ndarray, need: int = 300) -> np.ndarray:
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


def relabel_trial(participant: str, trial: str, *, min_correct_pct: float,
                   n_out: int, min_correct_calib: int, max_dist_mm: float,
                   side: str = "right"):
    trial_path, static_path = find_trial_csv(participant, trial)
    if trial_path is None:
        raise SystemExit(f"No CSV mapped to {participant}/{trial}")
    print(f"Loading {trial_path}")
    markers, labels = load_csv(str(trial_path))

    static_markers = None
    if static_path is not None:
        sm, sl = load_csv(str(static_path))
        static_markers = lmq.align_markers_to_labels(sm, sl, labels)
        print(f"  static reference: {static_path.name} ({sm.shape[0]} frames)")

    try:
        ref = lmq.load_ref_ranges_csv(str(REF_CSV), participant, trial)
    except ValueError:
        ref = auto_reference(markers)
        print("  reference frames: AUTO (no manual_frames.csv row)")
    ref = ref[ref < markers.shape[0]]

    status, bones = lmq.label_quality_cascade(
        markers, labels, ref, static_markers=static_markers)
    pct = (status == CORRECT).sum(axis=1) / status.shape[1] * 100
    print(f"  cascade: {(status == CORRECT).mean():.1%} correct overall; "
          f"{(pct >= min_correct_pct).sum()} frames >= {min_correct_pct:.0f}%")

    m2j = mr.marker_joint_map(labels, side=side)
    mapped = sorted(m2j)
    print(f"  {len(m2j)} markers map to MANO joints")

    # ---- calibration: subject shape + per-marker segment-frame offsets ----
    calib = mr.select_fit_frames(status, mapped,
                                 min_correct=min_correct_calib, max_frames=300)
    if calib.size < 20:
        raise SystemExit(
            f"Only {calib.size} calibration frames at min_correct="
            f"{min_correct_calib}; too few to calibrate this trial.")
    n_bins = len(np.unique((calib / markers.shape[0] * 30).astype(int)))
    print(f"  calibration frames: {calib.size} across {n_bins} time bins")

    t0 = time.time()
    res_c = mr.fit_frames(markers, labels, calib, mano_dir=str(MANO_DIR), side=side)
    offsets, spreads = mr.calibrate_marker_offsets(
        markers, calib, res_c.joints, m2j, status=status)
    print(f"  shape+offset calibration: {time.time() - t0:.0f}s, "
          f"betas={np.round(res_c.betas, 3)}")
    print(f"  offsets for {len(offsets)}/{len(m2j)} markers, "
          f"mean spread {np.mean(list(spreads.values())):.1f} mm")

    # ---- frames to repair ----
    # Repair EVERY qualifying frame, not just the ones the animation samples.
    # Repairing a scattered subset leaves each fixed frame an island among
    # unfixed neighbours, so a relabelled marker looks like it teleports
    # ~30-40mm between adjacent frames and the cascade's speed check
    # (12.5mm/frame) re-flags it — which makes the cascade useless as a
    # verification oracle even when the relabelling is geometrically right.
    frame_idx = np.unique(np.linspace(0, markers.shape[0] - 1, n_out).astype(int))
    targets = np.flatnonzero(pct >= min_correct_pct)
    print(f"  repairing all {targets.size} frames >= {min_correct_pct:.0f}% correct "
          f"(animation will show {frame_idx.size} sampled frames)")
    if targets.size == 0:
        raise SystemExit("No frames meet the correctness threshold.")

    # Fit pose on the trusted markers only: NaN out everything the cascade
    # did not verify, so a mislabelled marker cannot bend the fit toward
    # itself and then "confirm" its own wrong label.
    trusted = markers.copy()
    trusted[status != CORRECT] = np.nan

    t0 = time.time()
    res_t = mr.fit_frames(trusted, labels, targets, mano_dir=str(MANO_DIR),
                          side=side, betas=res_c.betas)
    print(f"  pose fit on {targets.size} frames: {time.time() - t0:.0f}s")

    pred = mr.predict_marker_positions(res_t.joints, offsets, m2j, len(labels))

    # ---- reassign flagged markers ----
    out_status = status.copy().astype(np.int8)
    relabelled = markers.copy()
    n_moves = 0
    moves_by_pair: dict[tuple[str, str], int] = {}
    for k, t in enumerate(targets):
        flagged = status[t] == INCORRECT
        if not flagged.any():
            continue
        mapping = mr.relabel_frame(
            markers[t] * 1e-3, pred[k], flagged, max_dist_m=max_dist_mm / 1000.0)
        if not mapping:
            continue
        mr.apply_relabel(relabelled, t, mapping)
        for slot, src in mapping.items():
            out_status[t, slot] = RELABELLED
            key = (labels[slot], labels[src])
            moves_by_pair[key] = moves_by_pair.get(key, 0) + 1
        n_moves += len(mapping)

    n_flagged = int((status[targets] == INCORRECT).sum())
    print(f"\n  relabelled {n_moves} marker-instances "
          f"of {n_flagged} flagged ({n_moves / max(1, n_flagged):.1%})")
    if moves_by_pair:
        print("  most common reassignments (label <- position taken from):")
        for (dst, src), c in sorted(moves_by_pair.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {dst:24s} <- {src:24s} {c:5d} frames")

    return dict(markers=markers, relabelled=relabelled, labels=labels, bones=bones,
                status=status, out_status=out_status, pct=pct,
                frame_idx=frame_idx, targets=targets, res_t=res_t, pred=pred)


# ---- animation -------------------------------------------------------------

_STEP_JS = """
var gd = document.getElementById('%(div_id)s');
var n = %(n_frames)d, cur = 0;
function go(i){ cur = Math.max(0, Math.min(n-1, i));
  Plotly.animate(gd, [String(cur)],
    {mode:'immediate', frame:{duration:0, redraw:true}, transition:{duration:0}}); }
document.addEventListener('keydown', function(e){
  if(e.key === 'ArrowRight'){ go(cur+1); e.preventDefault(); }
  if(e.key === 'ArrowLeft'){ go(cur-1); e.preventDefault(); }
});
gd.on('plotly_animatingframe', function(ev){
  if(ev && ev.name !== undefined){ cur = parseInt(ev.name); }});
"""


def build_figure(d, participant, trial, min_correct_pct):
    markers, labels = d["relabelled"], d["labels"]
    bones, out_status, pct = d["bones"], d["out_status"], d["pct"]
    frame_idx = d["frame_idx"]
    repaired = set(int(t) for t in d["targets"])

    center = np.nanmean(markers, axis=(0, 1))
    half = 200
    fill_colors = [marker_color(l) for l in labels]
    digit_text = [marker_digit(l) for l in labels]
    n_markers = markers.shape[1]

    def title(t):
        n_rel = int((out_status[t] == RELABELLED).sum())
        tag = "" if t in repaired else "   (below threshold — untouched)"
        return (f"{participant}/{trial}  frame {t}  "
                f"cascade-correct {pct[t]:.0f}%  relabelled {n_rel}{tag}")

    def trace_data(t):
        fm = markers[t]
        ok = np.isfinite(fm).all(axis=-1)
        x = np.where(ok, fm[:, 0], np.nan)
        y = np.where(ok, fm[:, 1], np.nan)
        z = np.where(ok, fm[:, 2], np.nan)
        ring = [STATUS_OUTLINE[int(s)] for s in out_status[t]]
        return x, y, z, ring

    def bone_xyz(t):
        fm = markers[t]
        xs, ys, zs = [], [], []
        for i, j, _ in bones:
            if np.isfinite(fm[i]).all() and np.isfinite(fm[j]).all():
                xs += [fm[i, 0], fm[j, 0], None]
                ys += [fm[i, 1], fm[j, 1], None]
                zs += [fm[i, 2], fm[j, 2], None]
        return xs, ys, zs

    x0, y0, z0, ring0 = trace_data(frame_idx[0])
    bx0, by0, bz0 = bone_xyz(frame_idx[0])

    ring_tr = go.Scatter3d(x=x0, y=y0, z=z0, mode="markers",
                           marker=dict(size=14, color=ring0),
                           hoverinfo="skip", name="status")
    mk_tr = go.Scatter3d(x=x0, y=y0, z=z0, mode="markers+text",
                         marker=dict(size=6, color=fill_colors),
                         text=digit_text, textposition="top center",
                         textfont=dict(size=10, color="#000000"),
                         hovertext=labels, hoverinfo="text", name="markers")
    bone_tr = go.Scatter3d(x=bx0, y=by0, z=bz0, mode="lines",
                           line=dict(color="lightgray", width=2),
                           hoverinfo="skip", name="bones")

    finger_legend = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                     marker=dict(size=6, color=c), name=f,
                     legendgroup="finger", legendgrouptitle=dict(text="Finger"))
        for f, c in FINGER_PALETTE.items()
    ]
    status_legend = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                     marker=dict(size=10, color=c), name=n,
                     legendgroup="status", legendgrouptitle=dict(text="Status"))
        for n, c in [("correct (cascade)", STATUS_OUTLINE[CORRECT]),
                     ("relabelled (MANO)", STATUS_OUTLINE[RELABELLED]),
                     ("still incorrect", STATUS_OUTLINE[INCORRECT])]
    ]

    frames = []
    for k, t in enumerate(frame_idx):
        x, y, z, ring = trace_data(t)
        bx, by, bz = bone_xyz(t)
        frames.append(go.Frame(
            name=str(k),
            data=[go.Scatter3d(x=x, y=y, z=z, marker=dict(color=ring)),
                  go.Scatter3d(x=x, y=y, z=z, marker=dict(color=fill_colors)),
                  go.Scatter3d(x=bx, y=by, z=bz)],
            layout=go.Layout(title=title(t))))

    fig = go.Figure(
        data=[ring_tr, mk_tr, bone_tr, *finger_legend, *status_legend],
        frames=frames,
        layout=go.Layout(
            title=title(frame_idx[0]),
            width=950, height=800, legend=dict(x=1.02, y=1),
            scene=dict(
                xaxis=dict(range=[center[0]-half, center[0]+half], title="X (mm)"),
                yaxis=dict(range=[center[1]-half, center[1]+half], title="Y (mm)"),
                zaxis=dict(range=[center[2]-half, center[2]+half], title="Z (mm)"),
                aspectmode="cube"),
            updatemenus=[dict(type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate", args=[None, {
                    "frame": {"duration": 60, "redraw": True},
                    "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate", args=[[None], {
                    "frame": {"duration": 0}, "mode": "immediate"}])])],
            sliders=[dict(currentvalue=dict(prefix="frame: "), steps=[
                dict(method="animate", label=str(t), args=[[str(k)], {
                    "frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                for k, t in enumerate(frame_idx)])],
        ),
    )
    return fig, len(frame_idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="P7")
    ap.add_argument("--trial", default="Trial1_handsonly")
    ap.add_argument("--min-correct-pct", type=float, default=50.0,
                    help="only repair frames at least this %% cascade-correct")
    ap.add_argument("--n-out", type=int, default=1000,
                    help="frames sampled for the animation (and repaired)")
    ap.add_argument("--min-correct-calib", type=int, default=10,
                    help="markers that must be correct for a calibration frame")
    ap.add_argument("--max-dist-mm", type=float, default=30.0,
                    help="max prediction-to-marker distance for a reassignment")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    args = ap.parse_args(argv)

    d = relabel_trial(args.participant, args.trial,
                      min_correct_pct=args.min_correct_pct, n_out=args.n_out,
                      min_correct_calib=args.min_correct_calib,
                      max_dist_mm=args.max_dist_mm, side=args.side)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, n_frames = build_figure(d, args.participant, args.trial, args.min_correct_pct)
    out = OUT_DIR / f"relabelled_{args.participant}_{args.trial}.html"
    fig.write_html(str(out), include_plotlyjs=True, div_id="animfig",
                   post_script=_STEP_JS % {"div_id": "animfig", "n_frames": n_frames})
    print(f"\nSaved {out}")
    return out


if __name__ == "__main__":
    main()
