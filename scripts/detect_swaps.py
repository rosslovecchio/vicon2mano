#!/usr/bin/env python3
"""Detect (and optionally fix) marker-label swaps in "clean" recordings.

Even recordings that pass triage as "clean" sometimes carry small labelling
errors — two markers whose names are swapped (adjacent fingers, or two markers
on one finger). The markers are all present and correctly *named for the set*,
just attached to the wrong physical points, so presence-based triage misses
them. Left undetected they corrupt the joint angles.

Two complementary geometric signals (validated on injected swaps):

1. **Descriptor swap-test (primary).** A consensus per-label distance
   descriptor is built from many clean recordings (median over recordings ->
   robust to the occasional swapped one). For each anatomically plausible
   pair (i, j) we test whether assigning marker i to label j and j to label i
   fits the consensus better than the current labelling, by more than a
   margin. Catches both cross-finger and same-finger swaps (~88% recall).
2. **Bone-length confirmation (precision).** Whether swapping also brings the
   affected within-finger bone lengths closer to canonical. This is decisive
   for cross-finger swaps but blind to same-finger ones, so it is used to
   RANK confidence, not as a veto:
       high   = descriptor + bone-length agree
       medium = descriptor only (review before trusting)

With ``--fix`` only high-confidence swaps are corrected by default (labels
exchanged; **positions never change**); ``--fix-medium`` includes the rest.

Usage
-----
python scripts/detect_swaps.py --h5 data/all_trajectories_synced.h5 \
    --session-type Hands_only_Left --side left \
    [--validate] [--margin 0.25] [--fix] [--out-dir vicon2mano/eval/swaps]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicon2mano.correspondence import _distance_descriptor, _normalise_label
from vicon2mano.loader import load_h5_trajectory

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# --- data access --------------------------------------------------------------

def side_markers(markers, labels, side):
    """Subset to one hand-side (or all, for side-less single-hand left)."""
    if side in ("left", "right"):
        idx = [i for i, l in enumerate(labels) if side in l.lower()]
    else:
        idx = list(range(len(labels)))
    return markers[:, idx], [labels[i] for i in idx]


def list_recordings(h5, session_type):
    import h5py
    with h5py.File(h5, "r") as f:
        return [f"{px}/{s}" for px in sorted(f.keys())
                for s in sorted(f[px].keys()) if session_type in s]


# --- canonical model ----------------------------------------------------------

def build_consensus(h5, recordings, side, stride, ref_labels=None):
    """Median per-label distance descriptor + canonical bone lengths."""
    sigs, bones, used = [], [], []
    labels_ref = ref_labels
    for rec in recordings:
        px, sess = rec.split("/")
        try:
            m, l, _ = load_h5_trajectory(h5, px, sess)
        except Exception:
            continue
        mk, ll = side_markers(m, l, side)
        if labels_ref is None:
            labels_ref = ll
        if ll != labels_ref:
            continue
        if np.isfinite(mk).all(axis=-1).mean() < 0.9:
            continue
        sigs.append(_distance_descriptor(mk, stride=stride))
        bones.append(_bone_lengths(mk, labels_ref))
        used.append(rec)
    canon = np.median(np.stack(sigs), axis=0)
    bone_canon = np.nanmedian(np.stack(bones), axis=0)
    return labels_ref, canon, bone_canon, used


def _finger_chain(labels, finger):
    out = []
    for k in (1, 2, 3):
        hit = None
        for i, l in enumerate(labels):
            if _normalise_label(l) == f"{finger.lower()}{k}":
                hit = i
                break
        out.append(hit)
    return out


def _bone_lengths(markers, labels):
    """Mean within-finger bone lengths [thumb12, thumb23, index12, ...]."""
    vals = []
    for fg in FINGERS:
        c = _finger_chain(labels, fg)
        for a, b in ((c[0], c[1]), (c[1], c[2])):
            if a is None or b is None:
                vals.append(np.nan)
            else:
                d = np.linalg.norm(markers[:, a] - markers[:, b], axis=-1)
                d = d[np.isfinite(d)]
                vals.append(float(d.mean()) if len(d) else np.nan)
    return np.array(vals)


def plausible_pairs(canon, k=4):
    """Anatomically plausible confusions: each label's k nearest by descriptor."""
    cc = np.linalg.norm(canon[:, None, :] - canon[None, :, :], axis=-1)
    pairs = set()
    for i in range(len(canon)):
        for j in np.argsort(cc[i])[1:k + 1]:
            pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


# --- detection ----------------------------------------------------------------

def _swapped(markers, i, j):
    sw = markers.copy()
    sw[:, [i, j]] = sw[:, [j, i]]
    return sw


def _bone_error(markers, labels, bone_canon):
    b = _bone_lengths(markers, labels)
    return float(np.nansum(np.abs(b - bone_canon)))


def _resolve_conflicts(flags):
    """A marker can be in at most one accepted swap — keep the strongest gain."""
    flags = sorted(flags, key=lambda f: -f[2])
    used, kept = set(), []
    for f in flags:
        i, j = f[0], f[1]
        if i in used or j in used:
            continue
        used |= {i, j}
        kept.append(f)
    return kept


def detect_swaps(markers, labels, canon, bone_canon, pairs, *, stride, margin):
    """Return [(i, j, gain, confidence)] swaps.

    Descriptor gain > margin flags a candidate (primary signal). Confidence is
    'high' when swapping also reduces bone-length error vs canonical (the
    independent confirmation), else 'medium'."""
    sig = _distance_descriptor(markers, stride=stride)
    flags = []
    for i, j in pairs:
        cur = np.linalg.norm(sig[i] - canon[i]) + np.linalg.norm(sig[j] - canon[j])
        swp = np.linalg.norm(sig[i] - canon[j]) + np.linalg.norm(sig[j] - canon[i])
        gain = cur - swp
        if gain <= margin:
            continue
        bone_confirms = _bone_error(markers, labels, bone_canon) > \
            _bone_error(_swapped(markers, i, j), labels, bone_canon)
        flags.append((i, j, float(gain), "high" if bone_confirms else "medium"))
    return _resolve_conflicts(flags)


# --- validation harness -------------------------------------------------------

def validate(h5, recordings, side, stride, margin, canon, bone_canon, labels_ref):
    """Inject one known swap per held-out recording; report recall + precision."""
    pairs = plausible_pairs(canon)
    rng = np.random.default_rng(0)
    tp = fn = trials = asis = 0
    high = 0
    for rec in recordings:
        px, sess = rec.split("/")
        m, l, _ = load_h5_trajectory(h5, px, sess)
        mk, ll = side_markers(m, l, side)
        if ll != labels_ref or np.isfinite(mk).all(axis=-1).mean() < 0.9:
            continue
        base = detect_swaps(mk, ll, canon, bone_canon, pairs, stride=stride, margin=margin)
        base_pairs = {(a, b) for a, b, *_ in base}
        asis += len(base)
        i, j = pairs[rng.integers(len(pairs))]
        got = detect_swaps(_swapped(mk, i, j), ll, canon, bone_canon, pairs,
                           stride=stride, margin=margin)
        hit = next((f for f in got if (f[0], f[1]) == (i, j)), None)
        trials += 1
        if hit:
            tp += 1
            if hit[3] == "high":
                high += 1
        else:
            fn += 1
    print(f"\n[validate] {trials} injected-swap trials  "
          f"recall={tp/max(1,trials):.0%} (TP={tp} FN={fn}; {high} high-confidence)  "
          f"detections on as-is recordings={asis}")


# --- main ---------------------------------------------------------------------

def _write_relabeled_csv(out_dir, rec, markers, labels):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from relabel_markers import write_wide_csv
    name = rec.replace("/", "__") + "_relabeled.csv"
    write_wide_csv(out_dir / name, markers, labels)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--session-type", required=True)
    ap.add_argument("--side", default="left", choices=["left", "right", "all"])
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--n-consensus", type=int, default=15)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fix", action="store_true",
                    help="write corrected recordings for high-confidence swaps")
    ap.add_argument("--fix-medium", action="store_true",
                    help="also apply medium-confidence (descriptor-only) swaps")
    ap.add_argument("--out-dir", default="vicon2mano/eval/swaps")
    args = ap.parse_args()

    recs = list_recordings(args.h5, args.session_type)
    print(f"[detect_swaps] {len(recs)} '{args.session_type}' recordings; "
          f"building consensus from up to {args.n_consensus}")
    labels_ref, canon, bone_canon, used = build_consensus(
        args.h5, recs[:args.n_consensus], args.side, args.stride)
    pairs = plausible_pairs(canon)
    print(f"[detect_swaps] consensus from {len(used)} recordings, "
          f"{len(labels_ref)} markers, {len(pairs)} plausible pairs")

    if args.validate:
        holdout = [r for r in recs if r not in used][:12]
        validate(args.h5, holdout, args.side, args.stride, args.margin,
                 canon, bone_canon, labels_ref)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pair_counts = [], {}
    n_scanned = n_high = n_med = 0
    for rec in recs:
        px, sess = rec.split("/")
        try:
            m, l, _ = load_h5_trajectory(args.h5, px, sess)
            mk, ll = side_markers(m, l, args.side)
        except Exception:
            continue
        if ll != labels_ref or np.isfinite(mk).all(axis=-1).mean() < 0.9:
            continue
        n_scanned += 1
        flags = detect_swaps(mk, ll, canon, bone_canon, pairs,
                             stride=args.stride, margin=args.margin)
        fixed_labels = list(l)
        applied = False
        for i, j, g, conf in flags:
            key = tuple(sorted((ll[i], ll[j])))
            pair_counts[key] = pair_counts.get(key, 0) + 1
            n_high += conf == "high"
            n_med += conf == "medium"
            rows.append({"recording": rec, "label_a": ll[i], "label_b": ll[j],
                         "confidence": conf, "descriptor_gain": round(g, 3)})
            if args.fix and (conf == "high" or args.fix_medium):
                gi, gj = l.index(ll[i]), l.index(ll[j])
                fixed_labels[gi], fixed_labels[gj] = fixed_labels[gj], fixed_labels[gi]
                applied = True
        if args.fix and applied:
            _write_relabeled_csv(out_dir, rec, m, fixed_labels)

    with open(out_dir / "detected_swaps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording", "label_a", "label_b",
                                          "confidence", "descriptor_gain"])
        w.writeheader()
        w.writerows(rows)

    recs_hit = len({r["recording"] for r in rows})
    print(f"\n[detect_swaps] scanned {n_scanned} recordings")
    print(f"  recordings with >=1 detected swap: {recs_hit} "
          f"({recs_hit / max(1, n_scanned):.0%})")
    print(f"  swaps: {len(rows)}  ({n_high} high-confidence, {n_med} medium)")
    if pair_counts:
        print("  by pair type:")
        for k, c in sorted(pair_counts.items(), key=lambda x: -x[1]):
            print(f"    {c:3d}  {k[0]} <-> {k[1]}")
    print(f"  -> {out_dir / 'detected_swaps.csv'}")


if __name__ == "__main__":
    main()
