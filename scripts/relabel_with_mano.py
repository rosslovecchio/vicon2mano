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
import traceback
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


# Static recordings that are usable for MANO calibration but NOT as the
# cascade's bone-length reference. P9's static was taken after a Forearm
# marker fell off and was reattached, so its forearm-plate geometry no longer
# matches the trials — folding it into the cascade reference collapsed 3 of
# P9's 4 trials to 0% correct, and it is deliberately left unmapped in
# trial_filename_map.csv for that reason. MANO calibration only ever touches
# the 16 *hand* markers (MANO has no forearm), so the forearm disturbance is
# irrelevant there and the file is still a valid shape/offset source.
MANO_ONLY_STATIC = {
    "P9": "static_after_forerm1_fall.csv",
}


def find_trial_csv(participant: str, trial: str) -> tuple[Path | None, Path | None, Path | None]:
    """Return (trial csv, cascade-static csv, mano-calibration-static csv).

    The two statics are usually the same file; they differ only where a
    static recording is trustworthy for the hand but not for the forearm
    (see MANO_ONLY_STATIC).
    """
    pdir = DATA_ROOT / participant / participant
    overrides = load_trial_map(TRIAL_MAP_CSV)
    trial_path = static_path = None
    for c in sorted(pdir.glob("*.csv")):
        key = overrides.get((participant, c.name.lower()))
        if key == trial:
            trial_path = c
        elif key == "static":
            static_path = c

    mano_static = static_path
    extra = MANO_ONLY_STATIC.get(participant)
    if extra is not None:
        cand = pdir / extra
        if cand.exists():
            mano_static = cand
    return trial_path, static_path, mano_static


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
                   min_trusted: int = 6, side: str = "right",
                   skip_unusable: bool = True):
    """Fit, predict, then assign at one gate value. See _prepare_trial."""
    ctx = _prepare_trial(participant, trial, min_correct_pct=min_correct_pct,
                         n_out=n_out, min_correct_calib=min_correct_calib,
                         side=side, skip_unusable=skip_unusable)
    return _assign_and_verify(ctx, max_dist_mm=max_dist_mm, min_trusted=min_trusted)


def _prepare_trial(participant: str, trial: str, *, min_correct_pct: float,
                    n_out: int, min_correct_calib: int, side: str = "right",
                    skip_unusable: bool = True):
    """Everything up to and including the model's marker-position prediction.

    Split out from the assignment so a gate sweep can reuse a single (slow)
    fit across many --max-dist-mm values.
    """
    trial_path, static_path, mano_static_path = find_trial_csv(participant, trial)
    if trial_path is None:
        raise SystemExit(f"No CSV mapped to {participant}/{trial}")
    print(f"Loading {trial_path}")
    markers, labels = load_csv(str(trial_path))

    # Cascade bone reference — only a static the map endorses.
    static_markers = None
    if static_path is not None:
        sm, sl = load_csv(str(static_path))
        static_markers = lmq.align_markers_to_labels(sm, sl, labels)
        print(f"  cascade static reference: {static_path.name} ({sm.shape[0]} frames)")

    # MANO calibration source — may be a static the cascade rejects.
    calib_static = static_markers
    if mano_static_path is not None and mano_static_path != static_path:
        sm2, sl2 = load_csv(str(mano_static_path))
        calib_static = lmq.align_markers_to_labels(sm2, sl2, labels)
        print(f"  MANO-only static (not used as cascade reference): "
              f"{mano_static_path.name} ({sm2.shape[0]} frames)")

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
    # Preferred source is the trial itself, since that guarantees the marker
    # placement matches. Trials the cascade rates poorly often have almost no
    # usable frames though (P10, P14: ~3), so fall back to the participant's
    # static recording — betas and skin offsets are properties of the subject
    # and the marker placement, not of the trial. A static trial is one pose,
    # so it cannot *validate* pose-invariance, but the segment-frame offset is
    # constructed to be pose-invariant and that assumption is validated
    # separately on the participants that do have varied data.
    calib_src, calib_labels = markers, labels
    calib_status = status
    calib = mr.select_fit_frames(status, mapped,
                                 min_correct=min_correct_calib, max_frames=300)
    source = "trial"
    if calib.size < 20:
        if calib_static is None:
            raise SystemExit(
                f"Only {calib.size} calibration frames in the trial and no static "
                f"recording for {participant}; cannot calibrate.")
        present = np.isfinite(calib_static[:, mapped]).all(axis=(1, 2))
        idx = np.flatnonzero(present)
        if idx.size < 20:
            raise SystemExit(
                f"Only {calib.size} trial frames and {idx.size} complete static "
                f"frames for {participant}; cannot calibrate.")
        if idx.size > 150:
            idx = idx[np.linspace(0, idx.size - 1, 150).astype(int)]
        calib_src, calib_status, calib, source = calib_static, None, idx, "static"
        print(f"  only {mr.select_fit_frames(status, mapped, min_correct=min_correct_calib).size}"
              f" usable trial frames -> calibrating from the static recording")

    if source == "trial":
        n_bins = len(np.unique((calib / markers.shape[0] * 30).astype(int)))
        print(f"  calibration frames: {calib.size} across {n_bins} time bins (trial)")
    else:
        print(f"  calibration frames: {calib.size} (static recording, single pose)")

    t0 = time.time()
    res_c = mr.fit_frames(calib_src, calib_labels, calib,
                          mano_dir=str(MANO_DIR), side=side)
    offsets, spreads = mr.calibrate_marker_offsets(
        calib_src, calib, res_c.joints, m2j, status=calib_status)
    print(f"  shape+offset calibration: {time.time() - t0:.0f}s, "
          f"betas={np.round(res_c.betas, 3)}")
    if not offsets:
        raise SystemExit("No marker offsets could be calibrated.")
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

    # Cheap reliability pre-check on a small sample before committing to the
    # full fit. A trial whose model cannot separate neighbouring markers
    # produces near-arbitrary reassignments (P10: ratio 1.75, only 18% of
    # moves verify), and the full fit costs minutes to hours, so it is worth
    # a few seconds to find out first.
    probe = targets[np.linspace(0, targets.size - 1,
                                min(200, targets.size)).astype(int)]
    probe = np.unique(probe)
    t0 = time.time()
    res_p = mr.fit_frames(trusted, labels, probe, mano_dir=str(MANO_DIR),
                          side=side, betas=res_c.betas)
    pred_p = mr.predict_marker_positions(res_p.joints, offsets, m2j, len(labels))
    rel_probe = mr.model_reliability(markers, status, pred_p, probe, m2j)
    print(f"  reliability probe ({probe.size} frames, {time.time()-t0:.0f}s): "
          f"p95 error {rel_probe['p95_err_mm']:.1f} mm vs spacing "
          f"{rel_probe['spacing_mm']:.1f} mm -> ratio {rel_probe['ratio']:.2f} "
          f"({rel_probe['verdict'].upper()})")
    if skip_unusable and rel_probe["verdict"] == "unusable":
        raise SystemExit(
            f"model unusable for this trial (ratio {rel_probe['ratio']:.2f}); "
            f"skipping — pass --no-skip-unusable to run it anyway")

    t0 = time.time()
    res_t = mr.fit_frames(trusted, labels, targets, mano_dir=str(MANO_DIR),
                          side=side, betas=res_c.betas)
    print(f"  pose fit on {targets.size} frames: {time.time() - t0:.0f}s")

    pred = mr.predict_marker_positions(res_t.joints, offsets, m2j, len(labels))

    return dict(participant=participant, trial=trial, markers=markers, labels=labels,
                bones=bones, status=status, pct=pct, m2j=m2j, offsets=offsets,
                spreads=spreads, frame_idx=frame_idx, targets=targets,
                res_t=res_t, pred=pred, ref=ref, static_markers=static_markers)


def _assign_and_verify(ctx, *, max_dist_mm, min_trusted: int = 6,
                        verbose: bool = True):
    participant, trial = ctx["participant"], ctx["trial"]
    markers, labels, bones = ctx["markers"], ctx["labels"], ctx["bones"]
    status, pct, targets = ctx["status"], ctx["pct"], ctx["targets"]
    pred, frame_idx = ctx["pred"], ctx["frame_idx"]
    ref, static_markers = ctx["ref"], ctx["static_markers"]

    rel = ctx.get("reliability")
    if rel is None:
        rel = mr.model_reliability(markers, status, pred, targets, ctx["m2j"])
        ctx["reliability"] = rel
    if verbose:
        print(f"  model reliability: p95 error {rel['p95_err_mm']:.1f} mm vs "
              f"marker spacing {rel['spacing_mm']:.1f} mm -> ratio "
              f"{rel['ratio']:.2f} ({rel['verdict'].upper()})")
        if rel["verdict"] != "good":
            print(f"  WARNING: at this ratio the model cannot reliably tell "
                  f"neighbouring markers apart; treat the reassignments as "
                  f"unverified (P10, ratio 1.75, had only 18% of moves verify "
                  f"against 86% on P7 at ratio 0.40)")

    # ---- reassign flagged markers ----
    out_status = status.copy().astype(np.int8)
    relabelled = markers.copy()
    n_moves = 0
    n_skipped_untrusted = 0
    moves_by_pair: dict[tuple[str, str], int] = {}
    for k, t in enumerate(targets):
        flagged = status[t] == INCORRECT
        if not flagged.any():
            continue
        # The pose was fitted from this frame's CORRECT markers alone. With
        # too few of them the fit is under-constrained, so its predictions
        # are not evidence about anything and reassigning from them would be
        # inventing structure. Matters most when --min-correct-pct is low.
        if int((status[t] == CORRECT).sum()) < min_trusted:
            n_skipped_untrusted += 1
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

    # ---- re-run the cascade on the repaired positions ----
    # This is the verification *and* what the animation reports, so "correct"
    # in the title means correct after repair rather than before it.
    status_after, _ = lmq.label_quality_cascade(
        relabelled, labels, ref, static_markers=static_markers)
    pct_after = (status_after == CORRECT).sum(axis=1) / status_after.shape[1] * 100

    before = int((status[targets] == CORRECT).sum())
    after = int((status_after[targets] == CORRECT).sum())
    tot = status[targets].size
    moved = out_status == RELABELLED
    now_ok = int((status_after[moved] == CORRECT).sum())
    print(f"\n  cascade on repaired frames: {before/tot:.1%} -> {after/tot:.1%} correct "
          f"({after - before:+d} marker-instances)")
    print(f"  of {n_moves} relabelled markers, {now_ok} ({now_ok/max(1,n_moves):.1%}) "
          f"now verify as CORRECT")

    # Ring colours come from the post-repair verdict, except that anything we
    # moved stays blue so the repair itself remains visible.
    ring_status = status_after.copy().astype(np.int8)
    ring_status[moved] = RELABELLED

    return dict(markers=markers, relabelled=relabelled, labels=labels, bones=bones,
                status=status, out_status=ring_status, pct=pct, pct_after=pct_after,
                frame_idx=frame_idx, targets=targets, res_t=ctx["res_t"], pred=pred,
                summary=dict(before=before, after=after, total=tot, moved=n_moves,
                             now_ok=now_ok, flagged=n_flagged,
                             ratio=rel["ratio"], verdict=rel["verdict"],
                             p95_err_mm=rel["p95_err_mm"], spacing_mm=rel["spacing_mm"]))


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
    pct_after = d["pct_after"]
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
                f"correct {pct_after[t]:.0f}% (was {pct[t]:.0f}%)  "
                f"relabelled {n_rel}{tag}")

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
                    help="max prediction-to-marker distance for a reassignment. "
                         "30mm is the sweep optimum; tighter is uniformly worse "
                         "(P7: 78.3%% correct at 8mm vs 86.2%% at 30mm) because "
                         "the assignment is solved jointly per frame, not as "
                         "independent pairwise choices")
    ap.add_argument("--min-trusted", type=int, default=6,
                    help="skip relabelling in frames with fewer correct markers "
                         "than this (the pose fit would be under-constrained)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip trials whose output HTML already exists (resume)")
    ap.add_argument("--io-retries", type=int, default=3,
                    help="retries for transient I/O errors reading the data drive")
    ap.add_argument("--io-retry-wait", type=float, default=30.0,
                    help="seconds to wait between I/O retries")
    ap.add_argument("--no-skip-unusable", action="store_true",
                    help="run a trial even when the reliability probe says the "
                         "model cannot separate neighbouring markers")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--all", action="store_true",
                    help="run every (participant, trial) in trial_filename_map.csv")
    args = ap.parse_args(argv)

    if args.all:
        return run_all(args)

    out = run_one(args.participant, args.trial, args)
    return out


def run_one(participant, trial, args):
    d = relabel_trial(participant, trial,
                      min_correct_pct=args.min_correct_pct, n_out=args.n_out,
                      min_correct_calib=args.min_correct_calib,
                      max_dist_mm=args.max_dist_mm,
                      min_trusted=args.min_trusted, side=args.side,
                      skip_unusable=not args.no_skip_unusable)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, n_frames = build_figure(d, participant, trial, args.min_correct_pct)
    out = OUT_DIR / f"relabelled_{participant}_{trial}.html"
    fig.write_html(str(out), include_plotlyjs=True, div_id="animfig",
                   post_script=_STEP_JS % {"div_id": "animfig", "n_frames": n_frames})
    print(f"\nSaved {out}")
    return out, d.get("summary")


TRIAL_KEYS = ["Trial1_handsonly", "Trial1_hoi", "Trial2_handsonly", "Trial2_hoi"]


def run_all(args):
    """Relabel every (participant, trial) listed in trial_filename_map.csv."""
    overrides = load_trial_map(TRIAL_MAP_CSV)
    participants = sorted({p for (p, _fn) in overrides},
                          key=lambda s: (len(s), s))   # P7, P8, ..., P10, ...
    rows = []
    t_start = time.time()
    for participant in participants:
        have = {overrides[(p, fn)] for (p, fn) in overrides if p == participant}
        for trial in TRIAL_KEYS:
            if trial not in have:
                continue
            print("\n" + "=" * 78)
            print(f"### {participant} / {trial}")
            print("=" * 78)
            if args.skip_existing and (OUT_DIR / f"relabelled_{participant}_{trial}.html").exists():
                print("  already done, skipping (--skip-existing)")
                continue
            # The data lives on an external drive that intermittently drops
            # out under sustained load; a single blip once killed 30 trials
            # in a row because every one failed instantly on OSError. Retry
            # I/O errors rather than burning the rest of the batch.
            for attempt in range(1, args.io_retries + 2):
                try:
                    _out, summary = run_one(participant, trial, args)
                    if summary:
                        rows.append((participant, trial, summary, None))
                    break
                except SystemExit as exc:
                    print(f"  SKIPPED: {exc}")
                    rows.append((participant, trial, None, str(exc)))
                    break
                except OSError as exc:
                    if attempt <= args.io_retries:
                        print(f"  I/O error ({exc}); retry {attempt}/{args.io_retries} "
                              f"in {args.io_retry_wait}s")
                        time.sleep(args.io_retry_wait)
                        continue
                    traceback.print_exc()
                    print(f"  FAILED after {attempt} attempts: {type(exc).__name__}: {exc}")
                    rows.append((participant, trial, None, f"{type(exc).__name__}: {exc}"))
                    break
                except Exception as exc:                # keep the batch going
                    traceback.print_exc()
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    rows.append((participant, trial, None, f"{type(exc).__name__}: {exc}"))
                    break

    print("\n" + "=" * 78)
    print(f"SUMMARY — {len(rows)} trials in {(time.time()-t_start)/60:.0f} min")
    print("=" * 78)
    print(f"{'trial':28s} {'before':>8s} {'after':>8s} {'moved':>8s} {'fixed':>8s} "
          f"{'ratio':>6s} {'verdict':>9s}")
    csv_path = OUT_DIR / "relabel_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv_mod.writer(fh)
        w.writerow(["participant", "trial", "pct_before", "pct_after",
                    "flagged", "moved", "now_correct",
                    "p95_err_mm", "spacing_mm", "ratio", "verdict", "note"])
        for participant, trial, s, note in rows:
            name = f"{participant}/{trial}"
            if s is None:
                print(f"{name:28s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s} "
                      f"{'--':>6s} {'--':>9s}  {note}")
                w.writerow([participant, trial, "", "", "", "", "", "", "", "", "", note])
                continue
            b, a = s["before"] / s["total"], s["after"] / s["total"]
            fixed = s["now_ok"] / max(1, s["moved"])
            print(f"{name:28s} {b:7.1%} {a:7.1%} {s['moved']:8d} {fixed:7.1%} "
                  f"{s['ratio']:6.2f} {s['verdict']:>9s}")
            w.writerow([participant, trial, f"{b:.4f}", f"{a:.4f}",
                        s["flagged"], s["moved"], s["now_ok"],
                        f"{s['p95_err_mm']:.1f}", f"{s['spacing_mm']:.1f}",
                        f"{s['ratio']:.3f}", s["verdict"], ""])
    print(f"\nWrote {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
