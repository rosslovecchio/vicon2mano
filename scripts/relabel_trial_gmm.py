#!/usr/bin/env python3
"""Whole-trial strategyGMM relabelling + animation, with no cascade.

Runs ``vicon2mano/gmm_labeler.py`` over an entire recording and writes an
interactive Plotly animation in the same style as
``results/mano_relabelling_pass1`` (finger-coloured markers with digits,
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

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.loader import load_csv                # noqa: E402
from vicon2mano import gmm_labeler as gl               # noqa: E402
import relabel_with_mano as rwm                        # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "gmm_relabelling"

ANCHOR_NAMES = ("Palm1", "Palm2", "Palm3")
FINGER_NAMES = ["Thumb1", "Thumb2", "Thumb3",
                "Index1", "Index2", "Index3",
                "Middle1", "Middle2", "Middle3",
                "Ring1", "Ring2", "Ring3",
                "Pinky1", "Pinky2", "Pinky3"]

# Same palette/encoding as scripts/relabel_with_mano.py, so the two tools'
# animations read the same way.
FINGER_PALETTE = rwm.FINGER_PALETTE
UNCHANGED, REASSIGNED, NO_FRAME = 0, 1, 2
STATUS_OUTLINE = {
    UNCHANGED:  "#2ecc71",          # green — strategyGMM agrees with the label
    REASSIGNED: "#2e86ff",          # blue  — strategyGMM moved this marker
    NO_FRAME:   "rgba(0,0,0,0)",    # no ring — no anchor frame / marker absent
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


def modal_bone_lengths(markers: np.ndarray, bones: list[tuple[int, int]],
                        *, bin_mm: float = 0.5, refine_mm: float = 2.0) -> np.ndarray:
    """Reference length per bone from each bone's *marginal* mode.

    Works only where a bone is correctly labelled in most frames. Kept for
    comparison and diagnostics; :func:`consensus_bone_lengths` is what the
    pipeline uses, because that assumption fails badly on real data (see
    its docstring).
    """
    out = np.full(len(bones), np.nan)
    for k, (i, j) in enumerate(bones):
        v = np.linalg.norm(markers[:, i] - markers[:, j], axis=1)
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        hist, edges = np.histogram(v, bins=np.arange(0, v.max() + bin_mm, bin_mm))
        mode = edges[int(np.argmax(hist))] + bin_mm / 2
        near = v[np.abs(v - mode) < refine_mm]
        out[k] = near.mean() if near.size else mode
    return out


def consensus_bone_lengths(
    markers: np.ndarray, bones: list[tuple[int, int]], *,
    tol_mm: float = 4.0, n_hypotheses: int = 400, n_compare: int = 3000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference lengths from the largest mutually-consistent set of frames.

    RANSAC over frames: every candidate frame's whole bone-length vector is
    one hypothesis, scored by how many other frames agree with it on *every*
    bone at once; the winner's inliers are then averaged.

    Why not the per-bone mode (:func:`modal_bone_lengths`): that assumes each
    bone is labelled correctly in most frames, and on real data it is not.
    On P7/Trial1_handsonly, `Ring1` is mislabelled in the *majority* of
    frames, so its marginal mode locks onto the wrong configuration --
    Palm2-Ring1 reads 34mm, which is shorter than Palm2-Pinky1 (54mm) and
    anatomically impossible. Consensus fixes it (50mm) because wrong
    configurations do not agree with *each other*: each different swap
    produces a different distance vector, so only correctly-labelled frames
    pile up into one large mutually-consistent set.

    Returns ``(ref_lengths, inlier_frames)``. The inliers are frames whose
    every bone matches the reference -- i.e. frames that are geometrically
    self-consistent, and so the natural training set for the GMMs, with no
    external quality verdict involved.
    """
    i_idx = np.array([b[0] for b in bones])
    j_idx = np.array([b[1] for b in bones])
    D = np.linalg.norm(markers[:, i_idx] - markers[:, j_idx], axis=2)   # (T, B)
    full = np.flatnonzero(np.isfinite(D).all(axis=1))
    if full.size == 0:
        return np.full(len(bones), np.nan), full

    rng = np.random.default_rng(seed)
    hyp = full if full.size <= n_hypotheses else full[
        rng.choice(full.size, size=n_hypotheses, replace=False)]
    comp = full if full.size <= n_compare else full[
        np.linspace(0, full.size - 1, n_compare).astype(int)]
    Dc = D[comp]

    best_score, best_h = -1, int(hyp[0])
    for h in hyp:
        score = int((np.abs(Dc - D[h]) <= tol_mm).all(axis=1).sum())
        if score > best_score:
            best_score, best_h = score, int(h)

    inliers = full[(np.abs(D[full] - D[best_h]) <= tol_mm).all(axis=1)]
    ref = D[inliers].mean(axis=0) if inliers.size else D[best_h]
    return ref, inliers


def bone_error(markers: np.ndarray, bones: list[tuple[int, int]],
               ref: np.ndarray) -> np.ndarray:
    """(T, n_bones) absolute deviation from each bone's reference length."""
    err = np.full((markers.shape[0], len(bones)), np.nan)
    for k, (i, j) in enumerate(bones):
        if not np.isfinite(ref[k]):
            continue
        err[:, k] = np.abs(np.linalg.norm(markers[:, i] - markers[:, j], axis=1) - ref[k])
    return err


def run(participant: str, trial: str, *, n_out: int, n_hypotheses: int,
        tol_mm: float, passes: int, process_var: float, obs_var: float,
        out_dir: Path) -> None:
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
    ref_tri = gl.learn_anchor_triangle(markers, anchor_idxs)
    print(f"  anchor triangle (modal, mm): {np.round(ref_tri, 2)}")
    t0 = time.time()
    triangle = gl.locate_anchor_triangle(markers, ref_tri, anchor_idxs, tol_mm=tol_mm)
    found = triangle[:, 0] >= 0
    as_labelled = found & (triangle == np.array(anchor_idxs)).all(axis=1)
    print(f"  anchor frame located on {found.mean():.1%} of frames "
          f"({found.sum()}), {int((found & ~as_labelled).sum())} with repaired "
          f"anchors  [{time.time() - t0:.0f}s]")

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
    ref_len, inliers = consensus_bone_lengths(markers, bones)
    print(f"  consensus reference from {len(inliers)} self-consistent frames "
          f"({len(inliers) / T:.1%})")

    train = np.intersect1d(inliers, np.flatnonzero(found))
    if train.size < 100:
        raise SystemExit(f"Only {train.size} frames are both anchor-valid and "
                          f"geometrically self-consistent; cannot train.")
    init_frame = int(train[0])
    mapping = None
    for p in range(1, passes + 1):
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

        if (cand_status == REASSIGNED).any():
            eb = bone_error(snapshot[None], bones, ref_len)[0]
            ea = bone_error(cand[None], bones, ref_len)[0]
            ok = np.isfinite(eb) & np.isfinite(ea)
            if ok.any() and np.nansum(ea[ok]) > np.nansum(eb[ok]) + 1e-9:
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
    fig = build_figure(relabelled, labels, bones, status, found, as_labelled,
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


def build_figure(markers, labels, bones, status, found, as_labelled,
                  frame_idx, participant, trial, err_before, err_after):
    """Plotly animation matching results/mano_relabelling_pass1's style."""
    fill_colors = [rwm.marker_color(l) for l in labels]
    digit_text = [rwm.marker_digit(l) for l in labels]
    center = np.nanmean(markers, axis=(0, 1))
    half = 200

    def title(t):
        n_moved = int((status[t] == REASSIGNED).sum())
        if not found[t]:
            tag = "   (no anchor frame — untouched)"
        elif as_labelled[t]:
            tag = ""
        else:
            tag = "   (anchors repaired)"
        eb = np.nanmean(err_before[t]); ea = np.nanmean(err_after[t])
        bone = "" if not np.isfinite(eb) else \
            f"  bone err {ea:.1f}mm (was {eb:.1f}mm)"
        return (f"{participant}/{trial}  frame {t}  "
                f"reassigned {n_moved}{bone}{tag}")

    def trace_data(t):
        fm = markers[t]
        ok = np.isfinite(fm).all(axis=-1)
        x = np.where(ok, fm[:, 0], np.nan)
        y = np.where(ok, fm[:, 1], np.nan)
        z = np.where(ok, fm[:, 2], np.nan)
        ring = [STATUS_OUTLINE[int(s)] if ok[i] else STATUS_OUTLINE[NO_FRAME]
                for i, s in enumerate(status[t])]
        return x, y, z, ring

    def bone_xyz(t):
        fm = markers[t]
        xs, ys, zs = [], [], []
        for i, j in bones:
            if np.isfinite(fm[i]).all() and np.isfinite(fm[j]).all():
                xs += [fm[i, 0], fm[j, 0], None]
                ys += [fm[i, 1], fm[j, 1], None]
                zs += [fm[i, 2], fm[j, 2], None]
        return xs, ys, zs

    t0 = int(frame_idx[0])
    x0, y0, z0, ring0 = trace_data(t0)
    bx0, by0, bz0 = bone_xyz(t0)

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
                     legendgroup="status", legendgrouptitle=dict(text="strategyGMM"))
        for n, c in [("label kept", STATUS_OUTLINE[UNCHANGED]),
                     ("reassigned", STATUS_OUTLINE[REASSIGNED])]
    ]

    frames = []
    for k, t in enumerate(frame_idx):
        t = int(t)
        x, y, z, ring = trace_data(t)
        bx, by, bz = bone_xyz(t)
        frames.append(go.Frame(
            name=str(k),
            data=[go.Scatter3d(x=x, y=y, z=z, marker=dict(color=ring)),
                  go.Scatter3d(x=x, y=y, z=z, marker=dict(color=fill_colors)),
                  go.Scatter3d(x=bx, y=by, z=bz)],
            layout=go.Layout(title=title(t))))

    return go.Figure(
        data=[ring_tr, mk_tr, bone_tr, *finger_legend, *status_legend],
        frames=frames,
        layout=go.Layout(
            title=title(t0),
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
    ap.add_argument("--tol-mm", type=float, default=2.5,
                    help="anchor-triangle match tolerance")
    ap.add_argument("--passes", type=int, default=2,
                    help="training passes (2 = bootstrap refit on "
                         "self-consistent frames)")
    ap.add_argument("--process-var", type=float, default=0.5)
    ap.add_argument("--obs-var", type=float, default=1.0)
    args = ap.parse_args(argv)

    run(args.participant, args.trial, n_out=args.n_out,
        n_hypotheses=args.n_hypotheses, tol_mm=args.tol_mm, passes=args.passes,
        process_var=args.process_var, obs_var=args.obs_var,
        out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
