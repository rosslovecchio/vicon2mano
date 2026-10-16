"""Marker library: where each marker sits on the MANO surface, plus real
per-frame deviations from that spot.

Built once from a fitted trial with trusted labels (P10/Trial2: the calibrated
surface fit + the manually labelled CSV). For every marker:

* an **attachment** -- one mesh face and a barycentric point on it, chosen as
  the face the real marker most often sits closest to;
* a per-frame **residual** -- the real marker's offset from that point,
  expressed in the face's own local frame (edge, in-plane normal, surface
  normal). That frame rotates and deforms with the mesh, so the residual
  carries skin standoff, sliding with flexion and placement error exactly as
  they were measured, and re-applying it on a different hand shape or pose
  transfers them realistically. Forearm markers are not on MANO at all; they
  get the same treatment against the nearest wrist face, so their residual
  encodes the real wrist angle of that frame.

Re-applying a frame's residuals to that same frame's fitted pose reproduces
the real markers exactly -- the round-trip the tests check.

Residuals are NaN wherever the manual position is not a real observation (it
was gap-filled in the manual file) or the frame's fit is poor, so the
generator only ever copies measured geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MANO hand_pose blocks (axis-angle, 3 per joint) in MANO's joint order.
FINGER_BLOCKS = {"Index": (0, 9), "Middle": (9, 18), "Pinky": (18, 27),
                 "Ring": (27, 36), "Thumb": (36, 45)}


def marker_group(name: str) -> str:
    """Which pose source a marker follows: its finger, else the hand root."""
    for f in FINGER_BLOCKS:
        if name.startswith(f):
            return f
    return "Root"


@dataclass
class MarkerLibrary:
    labels: list[str]          # (M,)
    face_idx: np.ndarray       # (M,) int  -- attachment face per marker
    bary: np.ndarray           # (M, 3)    -- barycentric point on that face
    hand_pose: np.ndarray      # (N, 45)   -- library poses, axis-angle
    betas: np.ndarray          # (10,)     -- subject shape of the source fit
    residual: np.ndarray       # (N, M, 3) mm, face-local; NaN = not a real observation
    frames: np.ndarray         # (N,) source frame numbers
    miss_rate: np.ndarray      # (M,) fraction of raw frames each marker is absent
    source: str = ""

    def subset(self, labels: list[str]) -> "MarkerLibrary":
        """The same library restricted to ``labels`` (in that order) -- e.g. a
        reduced marker set such as P1/P2's, where Forearm3, Forearm4 and Palm3
        were never placed. Poses and every kept marker's geometry are
        unchanged."""
        idx = [self.labels.index(l) for l in labels]
        return MarkerLibrary(labels=list(labels), face_idx=self.face_idx[idx], bary=self.bary[idx],
                             hand_pose=self.hand_pose, betas=self.betas,
                             residual=self.residual[:, idx], frames=self.frames,
                             miss_rate=self.miss_rate[idx], source=f"{self.source} | subset {len(idx)}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=np.array(self.labels), face_idx=self.face_idx,
                            bary=self.bary, hand_pose=self.hand_pose, betas=self.betas,
                            residual=self.residual, frames=self.frames,
                            miss_rate=self.miss_rate, source=np.array(self.source))

    @classmethod
    def load(cls, path: str | Path) -> "MarkerLibrary":
        z = np.load(path)
        return cls(labels=[str(s) for s in z["labels"]], face_idx=z["face_idx"],
                   bary=z["bary"], hand_pose=z["hand_pose"], betas=z["betas"],
                   residual=z["residual"], frames=z["frames"],
                   miss_rate=z["miss_rate"], source=str(z["source"]))


# ---------------------------------------------------------------- geometry

def face_frames(V: np.ndarray, F: np.ndarray, face_idx: np.ndarray, bary: np.ndarray):
    """Attachment points and local frames for every marker.

    V (..., 778, 3), returns P (..., M, 3) and R (..., M, 3, 3) whose columns
    are (edge, in-plane normal, surface normal), so world = P + R @ local.
    Works for numpy and torch alike.
    """
    xp = _xp(V)
    tri = V[..., F[face_idx], :]                                   # (..., M, 3, 3)
    P = (tri * _as(bary, V)[..., None]).sum(-2)
    e1 = tri[..., 1, :] - tri[..., 0, :]
    n = _cross(xp, e1, tri[..., 2, :] - tri[..., 0, :])
    e1 = e1 / _norm(xp, e1)
    n = n / _norm(xp, n)
    e2 = _cross(xp, n, e1)
    R = xp.stack([e1, e2, n], -1)
    return P, R


def _xp(a):
    if isinstance(a, np.ndarray):
        return np
    import torch
    return torch


def _as(b, like):
    if isinstance(like, np.ndarray):
        return b
    import torch
    return torch.as_tensor(b, dtype=like.dtype, device=like.device)


def _cross(xp, a, b):
    return np.cross(a, b) if xp is np else xp.cross(a, b, dim=-1)


def _norm(xp, a):
    if xp is np:
        return np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
    return a.norm(dim=-1, keepdim=True) + 1e-12


def closest_point_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Closest point on each triangle to p. p (3,), tri (K, 3, 3) -> (K, 3).
    Ericson, Real-Time Collision Detection 5.1.5, vectorised over K."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    out = np.empty_like(a)
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom; w = vc / denom
    out[:] = a + ab * v[:, None] + ac * w[:, None]                 # interior
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[m] = a[m] + ab[m] * (d1[m] / (d1[m] - d3[m]))[:, None]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[m] = a[m] + ac[m] * (d2[m] / (d2[m] - d6[m]))[:, None]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        out[m] = b[m] + (c[m] - b[m]) * t[m][:, None]
    m = (d1 <= 0) & (d2 <= 0); out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3); out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6); out[m] = c[m]
    return out


def nearest_face(p: np.ndarray, V: np.ndarray, F: np.ndarray, vert_faces: list[np.ndarray],
                 k_vert: int = 12) -> int:
    """Index of the mesh face closest to point p (search near the k nearest vertices)."""
    nv = np.argsort(np.linalg.norm(V - p, axis=1))[:k_vert]
    cand = np.unique(np.concatenate([vert_faces[v] for v in nv]))
    q = closest_point_on_triangles(p, V[F[cand]])
    return int(cand[np.linalg.norm(q - p, axis=1).argmin()])


# ---------------------------------------------------------------- building

def build_library(
    model,                         # smplx MANO layer, use_pca=False, flat_hand_mean=True
    fit: dict,                     # global_orient (T,3), hand_pose (T,45), transl (T,3) m, betas (10,)
    markers_mm: np.ndarray,        # (T, M, 3) trusted labelled positions
    real_mask: np.ndarray,         # (T, M) bool -- position is a real observation (not a fill)
    labels: list[str],
    raw_present: np.ndarray,       # (T_raw, M) bool, for occlusion statistics
    *,
    frames: np.ndarray,            # candidate library frames
    vote_stride: int = 10,
    max_fit_err_mm: float = 30.0,
    source: str = "",
    batch: int = 2000,
) -> MarkerLibrary:
    """See module docstring. ``max_fit_err_mm``: a marker whose fitted
    attachment point is further than this from the real marker in a frame
    gets a NaN residual there (bad fit -- don't copy it)."""
    import torch
    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
    F = np.asarray(model.faces, dtype=np.int64)
    betas = np.asarray(fit["betas"], np.float32)

    def verts(idx):
        out = []
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            with torch.no_grad():
                o = model(global_orient=torch.tensor(fit["global_orient"][j], device=dev),
                          hand_pose=torch.tensor(fit["hand_pose"][j], device=dev),
                          transl=torch.tensor(fit["transl"][j], device=dev),
                          betas=torch.tensor(betas, device=dev)[None].expand(len(j), -1))
            out.append(o.vertices.cpu().numpy() * 1000.0)
        return np.concatenate(out)

    # 1. attachment face per marker: majority over a subsample of frames
    vf = [[] for _ in range(model.v_template.shape[0])]
    for fi, (a, b, c) in enumerate(F):
        vf[a].append(fi); vf[b].append(fi); vf[c].append(fi)
    vf = [np.array(x) for x in vf]
    sub = frames[::vote_stride]
    Vs = verts(sub)
    M = len(labels)
    face_idx = np.zeros(M, np.int64)
    bary = np.zeros((M, 3))
    for m in range(M):
        votes, pts = {}, []
        for k, t in enumerate(sub):
            if not real_mask[t, m]:
                continue
            f = nearest_face(markers_mm[t, m], Vs[k], F, vf)
            votes[f] = votes.get(f, 0) + 1
        if not votes:
            raise ValueError(f"marker {labels[m]} never observed in the library frames")
        # The nearest face drifts between frames (a fingertip marker votes
        # over 4+ small faces). The raw majority face plus a median bary can
        # land where the marker never is -- Pinky3 came out 89mm off. So try
        # the top voted faces and keep the one whose fixed attachment point
        # stays closest to the marker, by median distance over all frames.
        ks = [k for k, t in enumerate(sub) if real_mask[t, m]]
        best = (np.inf, None, None)
        for f in sorted(votes, key=votes.get, reverse=True)[:8]:
            pts = []
            for k in ks:
                tri = Vs[k][F[f]]
                q = closest_point_on_triangles(markers_mm[sub[k], m], tri[None])[0]
                T_ = np.stack([tri[1] - tri[0], tri[2] - tri[0]], 1)
                uv = np.linalg.lstsq(T_, q - tri[0], rcond=None)[0]
                pts.append([1 - uv.sum(), uv[0], uv[1]])
            b = np.median(pts, 0); b /= b.sum()
            Pk = np.einsum("kij,i->kj", Vs[ks][:, F[f]], b)
            err = np.median(np.linalg.norm(markers_mm[sub[ks], m] - Pk, axis=-1))
            if err < best[0]:
                best = (err, f, b)
        face_idx[m], bary[m] = best[1], best[2]

    # 2. per-frame residuals in each attachment face's local frame
    V = verts(frames)
    P, R = face_frames(V, F, face_idx, bary)                       # (N,M,3), (N,M,3,3)
    X = markers_mm[frames]
    res = np.einsum("nmji,nmj->nmi", R, X - P)                      # R^T (x - P)
    ok = real_mask[frames] & np.isfinite(X).all(-1)
    # residual magnitude beyond the marker's own typical spread = bad fit there
    mag = np.linalg.norm(res, axis=-1)
    for m in range(M):
        v = mag[ok[:, m], m]
        lim = max(max_fit_err_mm, np.median(v) + 4 * 1.4826 * np.median(np.abs(v - np.median(v))))
        ok[:, m] &= mag[:, m] <= lim
    res[~ok] = np.nan

    return MarkerLibrary(labels=list(labels), face_idx=face_idx, bary=bary.astype(np.float32),
                         hand_pose=fit["hand_pose"][frames].astype(np.float32), betas=betas,
                         residual=res.astype(np.float32), frames=np.asarray(frames),
                         miss_rate=1.0 - raw_present.mean(0), source=source)
