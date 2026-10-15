"""Real Vicon frames as unlabelled clouds, with ground-truth identities.

The labeller never sees Vicon's labels: a frame's cloud is simply every raw
point present, whatever column it sits in. Ground truth comes from a trusted
labelled export: a raw point's identity is the label whose trusted position
coincides with it (<1 mm); a raw point matching nothing is a ghost.
"""

from __future__ import annotations

import numpy as np


def raw_identities(raw: np.ndarray, trusted: np.ndarray, tol_mm: float = 1.0) -> np.ndarray:
    """(T, C) identity of every raw column per frame: label index into the
    trusted columns, -1 ghost / unmatched, -2 absent."""
    T, C, _ = raw.shape
    ident = np.full((T, C), -2)
    for c in range(C):
        d = np.linalg.norm(trusted - raw[:, c][:, None], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        i = d.argmin(1)
        hit = d[np.arange(T), i] < tol_mm
        pres = np.isfinite(raw[:, c]).all(-1)
        ident[:, c] = np.where(pres, np.where(hit, i, -1), -2)
    return ident


def clouds(raw: np.ndarray, ident: np.ndarray, frames: np.ndarray, n_points: int):
    """Padded, median-centred clouds for ``frames``.

    Returns pts (F, n_points, 3) mm, valid (F, n_points), label (F, n_points)
    (-1 ghost, -2 pad) and col (F, n_points): the raw column each point came
    from (-1 pad), so labels can be mapped back onto the recording.
    """
    F = len(frames)
    pts = np.zeros((F, n_points, 3), np.float32)
    valid = np.zeros((F, n_points), bool)
    label = np.full((F, n_points), -2)
    col = np.full((F, n_points), -1)
    for k, t in enumerate(frames):
        c = np.flatnonzero(ident[t] > -2)[:n_points]
        X = raw[t, c]
        pts[k, :len(c)] = X - np.median(X, 0)
        valid[k, :len(c)] = True
        label[k, :len(c)] = ident[t, c]
        col[k, :len(c)] = c
    return pts, valid, label, col


def pieces(raw: np.ndarray, jump_mm: float = 5.0) -> np.ndarray:
    """(T, C) piece id per raw slot (-1 absent): each column is cut wherever
    the marker disappears or moves more than ``jump_mm`` in one frame.

    On P10/Trial2 99.9% of the 7,879 pieces carry a single true identity, so
    a piece is the unit a label should be decided for (SOMA's tracklets).
    """
    T, C, _ = raw.shape
    pid = np.full((T, C), -1)
    nxt = 0
    for c in range(C):
        p = raw[:, c]
        pres = np.isfinite(p).all(-1)
        step = np.r_[np.inf, np.linalg.norm(np.diff(p, axis=0), axis=-1)]
        start = pres & (~np.r_[False, pres[:-1]] | (step > jump_mm))
        seg = np.cumsum(start) - 1 + nxt
        pid[pres, c] = seg[pres]
        nxt += int(start.sum())
    return pid


def vote_pieces(frame_label: np.ndarray, frame_conf: np.ndarray, pid: np.ndarray,
                n_labels: int) -> np.ndarray:
    """Give every piece the label with the largest summed confidence over its
    frames, then resolve clashes: where two slots in a frame share a label,
    the piece with the larger vote share keeps it and the other is unlabelled
    (-1) in that frame. frame_label/conf: (T, C) per-frame labels (-1 none)."""
    P = pid.max() + 1
    votes = np.zeros((P, n_labels + 1))
    m = pid >= 0
    np.add.at(votes, (pid[m], np.where(frame_label[m] >= 0, frame_label[m], n_labels)), frame_conf[m])
    win = votes.argmax(1)
    share = votes.max(1) / np.maximum(votes.sum(1), 1e-12)
    out = np.where(m, win[np.maximum(pid, 0)], -1)
    out[out == n_labels] = -1
    sh = np.where(m, share[np.maximum(pid, 0)], 0.0)
    for t in np.flatnonzero([(len(r[r >= 0]) != len(np.unique(r[r >= 0]))) for r in out]):
        for L in np.unique(out[t][out[t] >= 0]):
            cs = np.flatnonzero(out[t] == L)
            if len(cs) > 1:
                keep = cs[sh[t, cs].argmax()]
                out[t, cs[cs != keep]] = -1
    return out


def relabelled_markers(raw: np.ndarray, slot_label: np.ndarray, n_labels: int) -> np.ndarray:
    """(T, n_labels, 3): each raw point moved to the column of its assigned
    label. Unlabelled points (-1) are dropped; positions are never altered."""
    T, C, _ = raw.shape
    out = np.full((T, n_labels, 3), np.nan, dtype=np.float64)
    t, c = np.nonzero(slot_label >= 0)
    out[t, slot_label[t, c]] = raw[t, c]
    return out


def write_wide_csv(path, markers: np.ndarray, labels: list[str]) -> None:
    """(T, N, 3) mm -> single-header wide Nexus CSV that
    ``vicon2mano.core.loader.load_csv`` reads back; blank cells for NaN."""
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["_Frame", "_Sub Frame"] + [h for l in labels for h in (f"{l}_X", "_Y", "_Z")])
        for t in range(markers.shape[0]):
            v = markers[t].reshape(-1)
            w.writerow([t + 1, 0] + ["" if not np.isfinite(x) else f"{x:.6f}" for x in v])
