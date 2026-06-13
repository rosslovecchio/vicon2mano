#!/usr/bin/env python3
"""Detect (and optionally fix) marker-label swaps in "clean" recordings.

Even recordings that pass triage as "clean" sometimes carry small labelling
errors — two markers whose names are swapped (adjacent fingers, or two markers
on one finger). The markers are all present and correctly *named for the set*,
just attached to the wrong physical points, so presence-based triage misses
them. Left undetected they corrupt the joint angles.

Detection focuses on **finger markers** (forearm markers are excluded: they
sit near-symmetrically on a rigid plate, are ambiguous to swap, and do not
affect finger joint angles). Three geometric/kinematic signals:

1. **Descriptor swap-test (primary).** Consensus per-label distance
   descriptor (median over many clean recordings -> robust to the occasional
   swapped one); for each plausible pair, does exchanging the two labels fit
   the consensus better than the current labelling by > margin. Catches both
   cross- and same-finger swaps (~88% recall).
2. **Bone-length confirmation.** Does the swap bring within-finger bone
   lengths closer to canonical. Decisive for cross-finger swaps.
3. **Motion confirmation.** Independent (kinematic) check. Cross-finger: a
   marker's speed time-series should correlate with its labelled finger's
   other markers; if it instead matches the swap partner's finger, the swap
   is confirmed. Same-finger: motion amplitude grows distally (DIP > PIP >
   MCP), so a violated ordering that the swap repairs confirms it.

Confidence: **high** if bone-length OR motion confirms the descriptor flag;
**medium** if descriptor only (review before trusting). With ``--fix`` only
high-confidence swaps are applied (positions never change); ``--fix-medium``
includes the rest.

Usage
-----
python scripts/detect_swaps.py --h5 data/all_trajectories_synced.h5 \
    --session-type Hands_only_Left --side left \
    [--validate] [--margin 0.25] [--fix] [--out-dir vicon2mano/eval/swaps]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicon2mano.correspondence import _distance_descriptor, _normalise_label
from vicon2mano.loader import load_h5_trajectory

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# --- data access --------------------------------------------------------------

def side_markers(markers, labels, side):
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


def _finger_of(label):
    """Finger name for a label, or None (forearm / palm / wrist)."""
    n = _normalise_label(label)
    for fg in FINGERS:
        if n.startswith(fg.lower()):
            return fg
    return None


def _rank_in_finger(label):
    """Trailing index 1/2/3 (MCP/PIP/DIP) within a finger, else None."""
    m = re.search(r"(\d+)$", _normalise_label(label))
    return int(m.group(1)) if m else None


def _is_finger_marker(label):
    return _finger_of(label) is not None


# --- canonical model ----------------------------------------------------------

def build_consensus(h5, recordings, side, stride, ref_labels=None):
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
        if ll != labels_ref or np.isfinite(mk).all(axis=-1).mean() < 0.9:
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


def plausible_pairs(canon, labels, k=4):
    """k nearest labels by descriptor, restricted to finger markers."""
    cc = np.linalg.norm(canon[:, None, :] - canon[None, :, :], axis=-1)
    finger = [i for i, l in enumerate(labels) if _is_finger_marker(l)]
    fset = set(finger)
    pairs = set()
    for i in finger:
        order = [j for j in np.argsort(cc[i]) if int(j) in fset and int(j) != i]
        for j in order[:k]:
            pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


# --- signals ------------------------------------------------------------------

def _swapped(markers, i, j):
    sw = markers.copy()
    sw[:, [i, j]] = sw[:, [j, i]]
    return sw


def _bone_error(markers, labels, bone_canon):
    return float(np.nansum(np.abs(_bone_lengths(markers, labels) - bone_canon)))


def _speeds(markers, stride=10):
    """Per-marker speed time-series (T', N), NaN where undefined."""
    s = markers[::stride]
    v = np.linalg.norm(np.diff(s, axis=0), axis=-1)     # (T'-1, N)
    return v


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _motion_confirms(markers, labels, i, j, speeds):
    """Independent kinematic confirmation that i,j are swapped."""
    fi, fj = _finger_of(labels[i]), _finger_of(labels[j])
    if fi is None or fj is None:
        return False
    if fi != fj:
        # cross-finger: speed should match own finger's other markers
        def mate(idx, fg, excl):
            ids = [k for k, l in enumerate(labels)
                   if _finger_of(l) == fg and k != excl]
            if not ids:
                return None
            return np.nanmean(speeds[:, ids], axis=1)
        mi, mj = mate(i, fi, i), mate(j, fj, j)
        if mi is None or mj is None:
            return False
        cur = _corr(speeds[:, i], mi) + _corr(speeds[:, j], mj)
        swp = _corr(speeds[:, i], mj) + _corr(speeds[:, j], mi)
        return swp > cur + 0.05
    # same-finger: distal markers move more (bigger arc); ordering must hold
    ri, rj = _rank_in_finger(labels[i]), _rank_in_finger(labels[j])
    if ri is None or rj is None:
        return False
    amp_i = np.nanmean(speeds[:, i])
    amp_j = np.nanmean(speeds[:, j])
    # expected: higher rank (more distal) -> larger amplitude
    # current labelling violated, swap repairs it
    expected_now = (amp_i < amp_j) == (ri < rj)
    return not expected_now and abs(amp_i - amp_j) > 0.05 * max(amp_i, amp_j, 1e-9)


def _resolve_conflicts(flags):
    flags = sorted(flags, key=lambda f: -f[2])
    used, kept = set(), []
    for f in flags:
        if f[0] in used or f[1] in used:
            continue
        used |= {f[0], f[1]}
        kept.append(f)
    return kept


def detect_swaps(markers, labels, canon, bone_canon, pairs, *, stride, margin):
    """Return [(i, j, gain, confidence)] swaps; confidence high if bone OR
    motion confirms the descriptor flag, else medium."""
    sig = _distance_descriptor(markers, stride=stride)
    speeds = _speeds(markers)
    flags = []
    for i, j in pairs:
        cur = np.linalg.norm(sig[i] - canon[i]) + np.linalg.norm(sig[j] - canon[j])
        swp = np.linalg.norm(sig[i] - canon[j]) + np.linalg.norm(sig[j] - canon[i])
        gain = cur - swp
        if gain <= margin:
            continue
        bone_ok = _bone_error(markers, labels, bone_canon) > \
            _bone_error(_swapped(markers, i, j), labels, bone_canon)
        motion_ok = _motion_confirms(markers, labels, i, j, speeds)
        conf = "high" if (bone_ok or motion_ok) else "medium"
        flags.append((i, j, float(gain), conf))
    return _resolve_conflicts(flags)


# --- validation harness -------------------------------------------------------

def validate(h5, recordings, side, stride, margin, canon, bone_canon, labels_ref):
    pairs = plausible_pairs(canon, labels_ref)
    rng = np.random.default_rng(0)
    tp = fn = trials = asis = high = 0
    for rec in recordings:
        px, sess = rec.split("/")
        m, l, _ = load_h5_trajectory(h5, px, sess)
        mk, ll = side_markers(m, l, side)
        if ll != labels_ref or np.isfinite(mk).all(axis=-1).mean() < 0.9:
            continue
        asis += len(detect_swaps(mk, ll, canon, bone_canon, pairs,
                                 stride=stride, margin=margin))
        i, j = pairs[rng.integers(len(pairs))]
        got = detect_swaps(_swapped(mk, i, j), ll, canon, bone_canon, pairs,
                           stride=stride, margin=margin)
        hit = next((f for f in got if (f[0], f[1]) == (i, j)), None)
        trials += 1
        if hit:
            tp += 1
            high += hit[3] == "high"
        else:
            fn += 1
    print(f"\n[validate] {trials} injected-swap trials  recall={tp/max(1,trials):.0%} "
          f"(TP={tp} FN={fn}; {high} reached high-confidence)  "
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
          f"consensus from up to {args.n_consensus}")
    labels_ref, canon, bone_canon, used = build_consensus(
        args.h5, recs[:args.n_consensus], args.side, args.stride)
    pairs = plausible_pairs(canon, labels_ref)
    print(f"[detect_swaps] consensus from {len(used)} recordings, "
          f"{len(labels_ref)} markers, {len(pairs)} plausible finger pairs")

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
        fixed_labels, applied = list(l), False
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
