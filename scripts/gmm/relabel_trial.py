#!/usr/bin/env python3
"""Whole-trial strategyGMM relabelling + animation, with no cascade.

Runs ``vicon2mano/gmm_labeler.py`` over an entire recording and writes an
interactive Plotly animation in the same style as
``results/mano`` (finger-coloured markers with digits,
skeleton lines, per-marker status rings, Play/Pause + frame slider).

**Nothing here depends on the quality cascade.** Everything the method
needs it derives from the marker geometry itself:

1. **Anchor plate, learned then located.** ``learn_anchor_triangle`` takes
   the *modal* pairwise distances of the Palm1/Palm2/Palm3 markers -- the
   rigid plate's true geometry is the one value that repeats, so the mode
   finds it even when a large share of frames are mislabelled (the median
   does not: it lands between the right and wrong configurations).
   ``locate_anchor_triangle`` then searches each frame's whole point cloud
   for a triple matching that triangle, so a frame whose palm labels were
   swapped still yields a usable frame -- the anchors are *repaired*, not
   merely rejected. On P7/Trial1_handsonly this lifts usable coverage from
   20% of frames (trusting the Palm1/2/3 columns) to 72.8%, of which
   25,447 frames needed repaired anchors.
2. **Two-pass training.** Pass 1 fits each finger marker's GMM on every
   anchor-valid frame. Since some of those frames are mislabelled, pass 2
   refits using only the frames pass 1 found self-consistent (no
   reassignment proposed), which sharpens the priors -- EM-style
   bootstrapping, no external labels required.
3. **Model-free verification.** Bone lengths (marker-to-marker distances
   along the finger chains) have their own modal reference values, exactly
   like the anchor triangle. Comparing bone-length error before vs after
   relabelling scores the repair without consulting the model that
   proposed it, and without the cascade.

Usage
-----
python scripts/relabel_trial_gmm.py --participant P7 --trial "Trial 1 Hands only"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv                # noqa: E402
from vicon2mano.strategies.gmm import labeler as gl               # noqa: E402
from vicon2mano.core import dataset as rwm
from vicon2mano.core import viz
from vicon2mano.core.bones import (                         # noqa: E402
    modal_bone_lengths, consensus_bone_lengths, bone_error,
)

OUT_DIR = REPO_ROOT / "results" / "gmm"

ANCHOR_NAMES = ("Palm1", "Palm2", "Palm3")
FINGER_NAMES = ["Thumb1", "Thumb2", "Thumb3",
                "Index1", "Index2", "Index3",
                "Middle1", "Middle2", "Middle3",
                "Ring1", "Ring2", "Ring3",
                "Pinky1", "Pinky2", "Pinky3"]

# Same palette/encoding as scripts/relabel_with_mano.py, so the two tools'
# animations read the same way.
FINGER_PALETTE = viz.FINGER_PALETTE
UNCHANGED, REASSIGNED, NO_FRAME = 0, 1, 2
STATUS_OUTLINE = {
    UNCHANGED:  viz.RING_KEPT,      # green — strategyGMM agrees with the label
    REASSIGNED: viz.RING_MOVED_OK,  # blue  — strategyGMM moved this marker
    NO_FRAME:   viz.RING_NONE,      # no ring — no anchor frame / marker absent
}


def _finger_chains(labels: list[str]) -> list[tuple[int, int]]:
    """Skeleton lines: consecutive joints within each finger, plus each
    finger's base back to the palm centre marker."""
    by_name = {}
    for i, l in enumerate(labels):
        by_name[l.split(":")[-1]] = i
    bones = []
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        chain = [f"{finger}{k}" for k in (1, 2, 3)]
        for a, b in zip(chain, chain[1:]):
            if a in by_name and b in by_name:
                bones.append((by_name[a], by_name[b]))
        if "Palm2" in by_name and chain[0] in by_name:
            bones.append((by_name["Palm2"], by_name[chain[0]]))
    for a, b in (("Palm1", "Palm2"), ("Palm2", "Palm3"), ("Palm1", "Palm3")):
        if a in by_name and b in by_name:
            bones.append((by_name[a], by_name[b]))
    return bones


def load_prior_gmms(prior_csv: Path, target_by_name: dict[str, int]) -> dict[int, gl.GMMParams]:
    """Marker GMMs from a separately labelled trial assumed fully correct.

    Unlike the in-trial bootstrap (``consensus_bone_lengths`` + repaired
    anchors), a ground-truth trial needs no self-consistency filtering --
    every finite frame is trustworthy by construction, and the anchor frame
    comes straight from the labelled Palm1/2/3 columns (``rigid_frames``),
    not the repair search in ``rigid_frames_from_triangle``. Returned GMMs
    are keyed by marker index in ``target_by_name`` (the trial being
    relabelled), not the prior file's own indexing, since the two trials'
    marker columns are not guaranteed to be in the same order.
    """
    print(f"Loading prior trial {prior_csv}")
    prior_markers, prior_labels = load_csv(str(prior_csv))
    prior_by_name = {l.split(":")[-1]: i for i, l in enumerate(prior_labels)}
    missing_anchor = [n for n in ANCHOR_NAMES if n not in prior_by_name]
    if missing_anchor:
        raise SystemExit(f"Prior CSV missing anchor marker(s) {missing_anchor}")
    prior_anchor_idxs = tuple(prior_by_name[n] for n in ANCHOR_NAMES)
    R_p, o_p = gl.rigid_frames(prior_markers, *prior_anchor_idxs)
    local_p = gl.to_local(prior_markers, R_p, o_p)
    ref_p = np.flatnonzero(np.isfinite(local_p[:, prior_anchor_idxs]).all(axis=(1, 2)))
    prior_target_idxs = [prior_by_name[n] for n in FINGER_NAMES if n in prior_by_name]
    gmm_p = gl.fit_marker_gmms(local_p, prior_target_idxs, ref_p, n_components=3)

    prior_gmms = {target_by_name[name]: gmm_p[prior_by_name[name]]
                  for name in FINGER_NAMES
                  if name in prior_by_name and prior_by_name[name] in gmm_p
                  and name in target_by_name}
    print(f"  prior trained on {len(ref_p)} ground-truth frames, "
          f"{len(prior_gmms)}/{len(FINGER_NAMES)} markers modelled")
    return prior_gmms


def run(participant: str, trial: str, *, n_out: int, n_hypotheses: int,
        tol_mm: float, passes: int, process_var: float, obs_var: float,
        out_dir: Path, min_persist: float = 0.8,
        max_residual_mm: float = 8.0, prior_csv: Path | None = None,
        bone_veto: bool = True) -> None:
    trial_path, *_ = rwm.find_trial_csv(participant, trial)
    if trial_path is None:
        raise SystemExit(f"No CSV mapped to {participant}/{trial}")
    print(f"Loading {trial_path}")
    markers, labels = load_csv(str(trial_path))
    T, N, _ = markers.shape
    print(f"  {T} frames, {N} markers")

    by_name = {l.split(":")[-1]: i for i, l in enumerate(labels)}
    anchor_idxs = tuple(by_name[n] for n in ANCHOR_NAMES)
    target_idxs = [by_name[n] for n in FINGER_NAMES if n in by_name]

    # ---- 1. anchor plate: learn its geometry, then find it every frame ----
    t0 = time.time()
    segments = gl.segment_by_jumps(markers, anchor_idxs)
    ref_tri = gl.learn_reference_geometry(markers, anchor_idxs, segments)
    triangle = gl.solve_segment_anchors(markers, segments, ref_tri, anchor_idxs,
                                        tol_mm=tol_mm, min_persist=min_persist)
    found = triangle[:, 0] >= 0
    as_labelled = found & (triangle == np.array(anchor_idxs)).all(axis=1)
    print(f"  {len(segments)} regimes; reference geometry "
          f"{np.round(ref_tri, 2)}")
    print(f"  anchor solved on {found.mean():.1%} of frames "
          f"({found.sum()}), {int((found & ~as_labelled).sum())} with repaired "
          f"anchors, {len(segments) - len(set(map(tuple, triangle[found])))} "
          f"regimes refused  [{time.time() - t0:.0f}s]")

    R, origin = gl.rigid_frames_from_triangle(markers, triangle)
    local = gl.to_local(markers, R, origin)     # NaN where no anchor frame

    # ---- 2. reference geometry + training frames, both by consensus ----
    #
    # Training on "every anchor-valid frame" does not work on a recording
    # this corrupted: the majority of those frames are themselves
    # mislabelled, so the GMM learns the wrong prior and then "corrects"
    # every frame toward it (measured: 34688/34688 anchor-valid frames
    # proposed a change, and the bone-length veto had to throw out 79% of
    # them). Consensus inliers are frames whose every bone length agrees
    # with the largest mutually-consistent set -- geometrically verified
    # clean, with no cascade and no manual labels.
    bones = _finger_chains(labels)
    anchor_set = set(anchor_idxs)
    anchor_bone_idx = [k for k, (i, j) in enumerate(bones)
                       if i in anchor_set and j in anchor_set]
    ref_len, inliers = consensus_bone_lengths(markers, bones)
    print(f"  consensus reference from {len(inliers)} self-consistent frames "
          f"({len(inliers) / T:.1%})")

    train = np.intersect1d(inliers, np.flatnonzero(found))
    if train.size < 100:
        raise SystemExit(f"Only {train.size} frames are both anchor-valid and "
                          f"geometrically self-consistent; cannot train.")
    init_frame = int(train[0])

    prior_gmms = load_prior_gmms(prior_csv, by_name) if prior_csv is not None else None

    mapping = None
    for p in range(1, passes + 1):
        if p == 1 and prior_gmms is not None:
            gmms = prior_gmms
            kept = [m for m in target_idxs if m in gmms]
            print(f"  pass 1: using external ground-truth prior, "
                  f"{len(kept)}/{len(target_idxs)} markers modelled")
        else:
            sample = train if len(train) <= 4000 else train[
                np.linspace(0, len(train) - 1, 4000).astype(int)]
            gmms = gl.fit_marker_gmms(local, target_idxs, sample, n_components=3)
            kept = [m for m in target_idxs if m in gmms]
            print(f"  pass {p}: training on {len(sample)} frames, "
                  f"{len(kept)}/{len(target_idxs)} markers modelled")
        t0 = time.time()
        mapping = gl.relabel_local(local, gmms, kept, n_hypotheses=n_hypotheses,
                                   process_var=process_var, obs_var=obs_var,
                                   init_frame=init_frame)
        identity = np.arange(len(kept))
        unchanged = (mapping == identity[None, :]) | (mapping < 0)
        moved_frames = ~unchanged.all(axis=1)
        print(f"    {int(moved_frames.sum())} frames with >=1 reassignment "
              f"({moved_frames.mean():.1%})  [{time.time() - t0 / 1:.0f}s]")
        if p < passes:
            # Refit on frames that are consensus-clean *after* this pass's
            # relabelling -- the repair should enlarge the self-consistent
            # set, which in turn sharpens the priors.
            provisional = markers.copy()
            kept_arr_p = np.array(kept)
            for t in np.flatnonzero(found):
                snap = markers[t]
                for slot, src in enumerate(mapping[t]):
                    if src >= 0:
                        provisional[t, kept_arr_p[slot]] = snap[kept_arr_p[src]]
            err_p = bone_error(provisional, bones, ref_len)
            clean = np.isfinite(err_p).all(axis=1) & (np.nanmax(err_p, axis=1) <= 4.0)
            train = np.intersect1d(np.flatnonzero(clean), np.flatnonzero(found))
            print(f"    consensus-clean after pass {p}: {train.size} frames")
            if train.size < 100:
                print("    too few; keeping this pass's model")
                break
            init_frame = int(train[0])

    # ---- 3. apply the mapping, gated on a model-free check ----
    #
    # A proposal is accepted only if it does not make the frame's bone
    # lengths worse. Bone lengths are measured against their own modal
    # reference (same rigidity argument as the anchor triangle), so this
    # never consults the GMM that made the proposal, nor the cascade --
    # it is an independent veto, mirroring the accept-only-if-verification-
    # agrees contract in relabel_with_mano.py.
    #
    # It earns its keep immediately: without it, P7 frame 1427 proposes a
    # Thumb1<->Thumb2 swap that sends the touched bones from 0.9mm to
    # 34.3mm of error (they are adjacent joints ~34mm apart, so swapping
    # them is exactly one bone-length wrong).
    relabelled = markers.copy()
    status = np.full((T, N), NO_FRAME, dtype=np.int8)
    kept_arr = np.array(kept)
    n_rejected = 0
    for t in range(T):
        if not found[t]:
            continue
        snapshot = markers[t]
        cand = snapshot.copy()
        cand_status = np.full(N, NO_FRAME, dtype=np.int8)
        for slot, src in enumerate(mapping[t]):
            m = kept_arr[slot]
            if src < 0:
                continue
            s = kept_arr[src]
            cand[m] = snapshot[s]
            cand_status[m] = UNCHANGED if s == m else REASSIGNED
        if not as_labelled[t]:
            for pos, a in enumerate(anchor_idxs):
                cand[a] = snapshot[triangle[t, pos]]
                cand_status[a] = REASSIGNED
        else:
            for a in anchor_idxs:
                cand_status[a] = UNCHANGED

        if bone_veto and (cand_status == REASSIGNED).any():
            eb = bone_error(snapshot[None], bones, ref_len)[0]
            ea = bone_error(cand[None], bones, ref_len)[0]
            ok = np.isfinite(eb) & np.isfinite(ea)
            # The anchor's own bones are forced to match the reference by
            # the repair itself, so scoring them is circular -- it hands
            # every anchor repair a free ~67mm "improvement" and lets bad
            # frames through (P7 frame 38315: 529->478 ACCEPT with them,
            # 461->476 REJECT without). Judge on the bones the repair did
            # not get to choose.
            ok[anchor_bone_idx] = False
            worse = ok.any() and np.nansum(ea[ok]) > np.nansum(eb[ok]) + 1e-9
            # ...and a frame still this far from anatomically valid has not
            # been "repaired" in any useful sense, however much it improved.
            still_bad = ok.any() and np.nanmean(ea[ok]) > max_residual_mm
            if worse or still_bad:
                n_rejected += 1
                relabelled[t] = snapshot
                status[t] = np.where(cand_status == NO_FRAME, NO_FRAME, UNCHANGED)
                continue
        relabelled[t] = cand
        status[t] = cand_status

    n_moved = int((status == REASSIGNED).sum())
    print(f"  {n_moved} marker-instances reassigned "
          f"({n_moved / max(1, int(found.sum()) * len(kept)):.1%} of usable slots); "
          f"{n_rejected} frames rejected by the bone-length veto")

    # ---- 4. model-free verification: bone lengths ----
    err_before = bone_error(markers, bones, ref_len)
    err_after = bone_error(relabelled, bones, ref_len)
    usable = found[:, None] & np.isfinite(err_before) & np.isfinite(err_after)
    mb, ma = np.nanmean(err_before[usable]), np.nanmean(err_after[usable])
    better = float((err_after[usable] < err_before[usable] - 1e-9).mean())
    print(f"  bone-length error (all bones, anchor-valid frames): "
          f"{mb:.2f}mm -> {ma:.2f}mm  ({better:.1%} of bone-instances improved)")

    # bones touching a reassigned marker — where the repair actually acted
    touch = np.zeros_like(err_before, dtype=bool)
    for k, (i, j) in enumerate(bones):
        touch[:, k] = (status[:, i] == REASSIGNED) | (status[:, j] == REASSIGNED)
    sel = usable & touch
    if sel.any():
        print(f"  bones touching a reassigned marker: "
              f"{np.nanmean(err_before[sel]):.2f}mm -> {np.nanmean(err_after[sel]):.2f}mm "
              f"({float((err_after[sel] < err_before[sel] - 1e-9).mean()):.1%} improved)")

    # ---- 5. animation ----
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{participant}_{trial.replace(' ', '_')}"
    frame_idx = np.unique(np.linspace(0, T - 1, n_out).astype(int))
    fig = build_figure(relabelled, markers, labels, bones, status, found, as_labelled,
                       frame_idx, participant, trial, err_before, err_after)
    out_html = out_dir / f"gmm_relabelled_{slug}.html"
    fig.write_html(str(out_html), include_plotlyjs=True, div_id="animfig",
                   auto_play=False)
    print(f"Saved {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")

    out_npz = out_dir / f"gmm_relabelled_{slug}.npz"
    np.savez_compressed(out_npz, relabelled=relabelled, status=status,
                        mapping=mapping, kept=kept_arr, triangle=triangle,
                        labels=np.array(labels), bones=np.array(bones),
                        ref_len=ref_len)
    print(f"Saved {out_npz}")

    summary = out_dir / f"gmm_relabelled_{slug}.txt"
    with summary.open("w") as f:
        f.write(f"strategyGMM whole-trial relabelling (no cascade)\n")
        f.write(f"{participant} / {trial}  --  {T} frames, {N} markers\n\n")
        f.write(f"anchor triangle (modal, mm): {np.round(ref_tri, 2).tolist()}\n")
        f.write(f"anchor frame located: {found.sum()} frames ({found.mean():.1%})\n")
        f.write(f"  with repaired anchors: {int((found & ~as_labelled).sum())}\n")
        f.write(f"markers modelled: {len(kept)}\n")
        f.write(f"marker-instances reassigned: {n_moved}\n\n")
        f.write(f"bone-length error, all bones:  {mb:.2f}mm -> {ma:.2f}mm "
              f"({better:.1%} improved)\n")
        if sel.any():
            f.write(f"bone-length error, bones touching a reassigned marker: "
                  f"{np.nanmean(err_before[sel]):.2f}mm -> "
                  f"{np.nanmean(err_after[sel]):.2f}mm\n")
    print(f"Saved {summary}")


def _bone_finger(i: int, j: int, labels: list[str]) -> str:
    """Which finger a bone belongs to, for colouring -- e.g. the
    Palm2-Thumb1 base bone is coloured as "thumb" (the finger it connects
    to), not "palm", by checking both endpoints and preferring a named
    finger over the palm/forearm fallback."""
    for idx in (i, j):
        low = labels[idx].lower()
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            if finger in low:
                return finger
    return "palm"


def build_figure(corrected, original, labels, bones, status, found, as_labelled,
                  frame_idx, participant, trial, err_before, err_after):
    """Plotly animation matching results/mano style.

    ``corrected`` (this trial's repaired positions -- ``relabelled`` in
    ``run()``) drives the marker dots and the bold/solid skeleton;
    ``original`` (the raw, unrepaired positions) drives the thin/dashed
    skeleton toggled by the "original bones" button. Marker colour/label
    text are fixed per column (``labels``) -- since a repair only ever
    swaps *positions* between columns, not the column identities
    themselves, a column's colour already reflects its corrected identity.
    """
    fill_colors = [viz.marker_color(l) for l in labels]
    digit_text = [viz.marker_digit(l) for l in labels]
    center = np.nanmean(corrected, axis=(0, 1))
    half = 200

    bone_groups: dict[str, list[tuple[int, int]]] = {}
    for i, j in bones:
        bone_groups.setdefault(_bone_finger(i, j, labels), []).append((i, j))
    group_names = list(bone_groups)

    def title(t):
        # Report the OUTCOME, not the proposal. `as_labelled` says the
        # anchor search preferred a different triple, but that repair is
        # only real if the bone-length veto accepted it -- otherwise the
        # frame is byte-identical to the input. Reporting the proposal
        # made vetoed frames read as "(anchors repaired)  reassigned 0"
        # with identical before/after error, which is a contradiction
        # (seen on frame 13410).
        n_moved = int((status[t] == REASSIGNED).sum())
        if not found[t]:
            tag = "   (no anchor frame — untouched)"
        elif n_moved == 0 and not as_labelled[t]:
            tag = "   (repair proposed but VETOED — unchanged)"
        elif n_moved == 0:
            tag = "   (no change proposed)"
        elif not as_labelled[t]:
            tag = "   (anchors repaired)"
        else:
            tag = ""
        eb = np.nanmean(err_before[t]); ea = np.nanmean(err_after[t])
        bone = "" if not np.isfinite(eb) else \
            f"  bone err {ea:.1f}mm (was {eb:.1f}mm)"
        return (f"{participant}/{trial}  frame {t}  "
                f"reassigned {n_moved}{bone}{tag}")

    def trace_data(t):
        fm = corrected[t]
        ok = np.isfinite(fm).all(axis=-1)
        x = np.where(ok, fm[:, 0], np.nan)
        y = np.where(ok, fm[:, 1], np.nan)
        z = np.where(ok, fm[:, 2], np.nan)
        ring = [STATUS_OUTLINE[int(s)] if ok[i] else STATUS_OUTLINE[NO_FRAME]
                for i, s in enumerate(status[t])]
        return x, y, z, ring

    def group_bone_xyz(fm, group):
        xs, ys, zs = [], [], []
        for i, j in group:
            if np.isfinite(fm[i]).all() and np.isfinite(fm[j]).all():
                xs += [fm[i, 0], fm[j, 0], None]
                ys += [fm[i, 1], fm[j, 1], None]
                zs += [fm[i, 2], fm[j, 2], None]
        return xs, ys, zs

    t0 = int(frame_idx[0])
    x0, y0, z0, ring0 = trace_data(t0)

    def marker_xyz(fm):
        ok = np.isfinite(fm).all(axis=-1)
        x = np.where(ok, fm[:, 0], np.nan)
        y = np.where(ok, fm[:, 1], np.nan)
        z = np.where(ok, fm[:, 2], np.nan)
        return x, y, z

    ring_tr = go.Scatter3d(x=x0, y=y0, z=z0, mode="markers",
                           marker=dict(size=14, color=ring0),
                           hoverinfo="skip", name="status")
    mk_tr = go.Scatter3d(x=x0, y=y0, z=z0, mode="markers+text",
                         marker=dict(size=6, color=fill_colors),
                         text=digit_text, textposition="top center",
                         textfont=dict(size=10, color="#000000"),
                         hovertext=labels, hoverinfo="text", name="markers")
    ox0, oy0, oz0 = marker_xyz(original[t0])
    orig_mk_tr = go.Scatter3d(x=ox0, y=oy0, z=oz0, mode="markers",
                              marker=dict(size=5, color=fill_colors, opacity=0.35),
                              hovertext=[f"{l} (raw)" for l in labels], hoverinfo="text",
                              name="markers (raw)")

    # Two skeletons: the repaired one (bold, solid, one trace per finger so
    # each bone can carry its own finger colour) and the raw/unrepaired one
    # (thin, dashed, same colours) -- overlaid so a repair's effect on the
    # geometry is visible directly, not just inferred from the status ring.
    corrected_trs, original_trs = [], []
    for name in group_names:
        color = FINGER_PALETTE.get(name, "#444444")
        cx, cy, cz = group_bone_xyz(corrected[t0], bone_groups[name])
        corrected_trs.append(go.Scatter3d(
            x=cx, y=cy, z=cz, mode="lines",
            line=dict(color=color, width=6), hoverinfo="skip",
            name=f"{name} (corrected)", legendgroup="bones-corrected",
            legendgrouptitle=dict(text="Bones") if name == group_names[0] else None,
            showlegend=False))
        ox, oy, oz = group_bone_xyz(original[t0], bone_groups[name])
        original_trs.append(go.Scatter3d(
            x=ox, y=oy, z=oz, mode="lines",
            line=dict(color=color, width=2, dash="dash"), hoverinfo="skip",
            name=f"{name} (original)", legendgroup="bones-original",
            visible=True, showlegend=False))

    finger_legend = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                     marker=dict(size=6, color=c), name=f,
                     legendgroup="finger", legendgrouptitle=dict(text="Finger"))
        for f, c in FINGER_PALETTE.items()
    ]
    status_legend = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                     marker=dict(size=10, color=c), name=n,
                     legendgroup="status", legendgrouptitle=dict(text="strategyGMM"))
        for n, c in [("label kept", STATUS_OUTLINE[UNCHANGED]),
                     ("reassigned", STATUS_OUTLINE[REASSIGNED])]
    ]
    style_legend = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                     line=dict(color="#444444", width=6), name="corrected (repaired)",
                     legendgroup="style", legendgrouptitle=dict(text="Bone style")),
        go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                     line=dict(color="#444444", width=2, dash="dash"), name="original (raw)",
                     legendgroup="style"),
    ]

    # Data trace order: ring, corrected markers, original markers, then
    # each finger's corrected bone then original bone traces. Indices below
    # must track this order -- they drive which traces the two checkboxes
    # (corrected / original) toggle together.
    n_corrected = len(corrected_trs)
    corrected_idx = [1] + list(range(3, 3 + n_corrected))
    original_idx = [2] + list(range(3 + n_corrected, 3 + 2 * n_corrected))

    frames = []
    for k, t in enumerate(frame_idx):
        t = int(t)
        x, y, z, ring = trace_data(t)
        ox, oy, oz = marker_xyz(original[t])
        frame_data = [go.Scatter3d(x=x, y=y, z=z, marker=dict(color=ring)),
                      go.Scatter3d(x=x, y=y, z=z, marker=dict(color=fill_colors)),
                      go.Scatter3d(x=ox, y=oy, z=oz, marker=dict(color=fill_colors))]
        for name in group_names:
            cx, cy, cz = group_bone_xyz(corrected[t], bone_groups[name])
            frame_data.append(go.Scatter3d(x=cx, y=cy, z=cz))
        for name in group_names:
            ox2, oy2, oz2 = group_bone_xyz(original[t], bone_groups[name])
            frame_data.append(go.Scatter3d(x=ox2, y=oy2, z=oz2))
        frames.append(go.Frame(name=str(k), data=frame_data,
                               layout=go.Layout(title=title(t))))

    # Plotly's updatemenus have no native checkbox widget -- each of these
    # two independent Show/Hide button pairs acts as one, toggling the
    # markers+bones of that set (corrected or original) together via
    # "restyle", orthogonal to the Play/Pause animation controls. Stacked
    # in their own row (not side-by-side) so a wide 2-button menu can never
    # visually overlap its neighbour and silently steal its clicks.
    def checkbox(label: str, idx: list[int], y: float) -> dict:
        return dict(type="buttons", direction="right", showactive=True,
                    x=0.0, y=y, pad=dict(t=4, b=4),
                    buttons=[
                        dict(label=f"Show {label}", method="restyle",
                             args=[{"visible": True}, idx]),
                        dict(label=f"Hide {label}", method="restyle",
                             args=[{"visible": False}, idx]),
                    ])

    return go.Figure(
        data=[ring_tr, mk_tr, orig_mk_tr, *corrected_trs, *original_trs,
             *finger_legend, *status_legend, *style_legend],
        frames=frames,
        layout=go.Layout(
            title=title(t0),
            width=950, height=800, legend=dict(x=1.02, y=1),
            scene=dict(
                xaxis=dict(range=[center[0]-half, center[0]+half], title="X (mm)"),
                yaxis=dict(range=[center[1]-half, center[1]+half], title="Y (mm)"),
                zaxis=dict(range=[center[2]-half, center[2]+half], title="Z (mm)"),
                aspectmode="cube"),
            margin=dict(t=160),
            updatemenus=[
                dict(type="buttons", showactive=False, x=0.0, y=1.24, buttons=[
                    dict(label="Play", method="animate", args=[None, {
                        "frame": {"duration": 60, "redraw": True},
                        "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {
                        "frame": {"duration": 0}, "mode": "immediate"}])]),
                checkbox("corrected", corrected_idx, 1.14),
                checkbox("original", original_idx, 1.04),
            ],
            sliders=[dict(currentvalue=dict(prefix="frame: "), steps=[
                dict(method="animate", label=str(int(t)), args=[[str(k)], {
                    "frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                for k, t in enumerate(frame_idx)])],
        ),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="P7")
    ap.add_argument("--trial", default="Trial 1 Hands only",
                    help="manual_frames.csv Session name")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--n-out", type=int, default=200,
                    help="frames sampled into the animation")
    ap.add_argument("--n-hypotheses", type=int, default=5)
    ap.add_argument("--tol-mm", type=float, default=2.0,
                    help="anchor-triangle match tolerance. The plate is not "
                         "perfectly rigid (mount flex, marker wobble): the "
                         "best-achievable triangle error on P7 is ~1.2mm per "
                         "distance at the median and ~2.4mm at p90, so 2.5mm "
                         "rejected 27%% of frames for no good reason. 4.5mm "
                         "takes coverage 73.4%% -> 97.5%% without changing the "
                         "trainable-clean-frame count; the bone-length veto is "
                         "the backstop against a loose match.")
    ap.add_argument("--max-residual-mm", type=float, default=8.0,
                    help="reject a frame's repair if the result is still "
                         "further than this from anatomically valid, however "
                         "much it improved")
    ap.add_argument("--min-persist", type=float, default=0.8,
                    help="fraction of a regime's frames a candidate anchor "
                         "triple must match before it is accepted; below this "
                         "the regime is refused rather than guessed at")
    ap.add_argument("--passes", type=int, default=2,
                    help="training passes (2 = bootstrap refit on "
                         "self-consistent frames)")
    ap.add_argument("--process-var", type=float, default=0.5)
    ap.add_argument("--obs-var", type=float, default=1.0)
    ap.add_argument("--prior-csv", default=None,
                    help="path to a separately labelled trial, assumed fully "
                         "correct (e.g. a manually-labelled-and-filled export), "
                         "whose marker GMMs seed pass 1 instead of this trial's "
                         "own consensus-bootstrapped frames. Later passes (if "
                         "--passes > 1) still refit on this trial's own "
                         "consensus-clean frames as usual.")
    ap.add_argument("--no-bone-veto", action="store_true",
                    help="accept every proposed reassignment unconditionally, "
                         "skipping step 3's independent bone-length check. "
                         "Without the veto, a wrong proposal (e.g. a "
                         "Thumb1<->Thumb2 swap) reaches the output uncaught -- "
                         "use only to see the model's raw output, e.g. when "
                         "judging an external --prior-csv on its own merits.")
    args = ap.parse_args(argv)

    run(args.participant, args.trial, n_out=args.n_out,
        n_hypotheses=args.n_hypotheses, tol_mm=args.tol_mm, passes=args.passes,
        process_var=args.process_var, obs_var=args.obs_var,
        out_dir=Path(args.out_dir), min_persist=args.min_persist,
        max_residual_mm=args.max_residual_mm,
        prior_csv=Path(args.prior_csv) if args.prior_csv else None,
        bone_veto=not args.no_bone_veto)


if __name__ == "__main__":
    main()
