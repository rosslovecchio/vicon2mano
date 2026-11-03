"""Camera -> Vicon extrinsics for the RealSense cameras (docs/camera_calibration.md).

Correspondences come from MediaPipe Hands: its 21 landmarks are in the repo's 21-joint
order, so they pair one-to-one with the fitted MANO ``joints21`` (Vicon frame). A pose
from those is then refined on the reflective marker dots, re-picked inside a shrinking
gate around each projected Vicon marker. Dots alone do not work: the EMG sleeve gives
~100 bright false detections per frame.

Pose convention: ``p = (rotvec, t)`` maps Vicon -> camera, ``x_cam = R x_vicon + t`` (metres).
Intrinsics are a dict ``fx, fy, ppx, ppy, model, coeffs`` as librealsense reports them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def detect_dots(rgb: np.ndarray, th: float = 40, size: int = 9, amin: int = 2, amax: int = 80,
                sat_max: float = 60) -> np.ndarray:
    """(n, 2) pixel centres of small near-white spots brighter than their surroundings (top-hat filter)."""
    g = rgb.astype(np.float32).mean(2)
    top = g - ndi.grey_opening(g, size=(size, size))
    m = (top > th) & (rgb.max(2).astype(np.float32) - rgb.min(2) < sat_max)
    lab, n = ndi.label(m)
    if not n:
        return np.zeros((0, 2))
    idx = np.arange(1, n + 1)
    area = ndi.sum(m, lab, idx)
    cen = np.array(ndi.center_of_mass(top * m, lab, idx))[:, ::-1]
    return cen[(area >= amin) & (area <= amax)]


def depth_at(depth_m: np.ndarray, uv: np.ndarray, r: int = 2) -> np.ndarray:
    """Median valid depth (m) in a (2r+1)^2 patch around each pixel; NaN where fewer than 3 are valid."""
    out = np.full(len(uv), np.nan)
    for k, (x, y) in enumerate(np.round(uv).astype(int)):
        p = depth_m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        p = p[p > 0]
        if len(p) >= 3:
            out[k] = np.median(p)
    return out


def deproject(uv: np.ndarray, z: np.ndarray, intr: dict) -> np.ndarray:
    """Pixels + depth (m) -> camera-frame points, as rs2_deproject_pixel_to_point (inverse Brown-Conrady)."""
    x, y = (uv[:, 0] - intr["ppx"]) / intr["fx"], (uv[:, 1] - intr["ppy"]) / intr["fy"]
    if "inverse_brown_conrady" in intr["model"]:
        k1, k2, p1, p2, k3 = intr["coeffs"]
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return np.c_[x * z, y * z, z]


def project(p: np.ndarray, X: np.ndarray, intr: dict) -> np.ndarray:
    """Vicon-frame points (..., 3) -> pixels (..., 2): the inverse of ``deproject`` (fixed-point undistortion)."""
    C = X @ Rot.from_rotvec(p[:3]).as_matrix().T + p[3:]
    x, y = C[..., 0] / C[..., 2], C[..., 1] / C[..., 2]
    if "inverse_brown_conrady" in intr["model"] and np.any(intr["coeffs"]):
        k1, k2, p1, p2, k3 = intr["coeffs"]
        xd, yd = x.copy(), y.copy()
        for _ in range(8):
            r2 = x * x + y * y
            f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            x, y = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / f, (yd - 2 * p2 * x * y - p1 * (r2 + 2 * y * y)) / f
    return np.stack([x * intr["fx"] + intr["ppx"], y * intr["fy"] + intr["ppy"]], -1)


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) minimising |R A_i + t - B_i| over rows."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cb - R @ ca


def solve_2d(p: np.ndarray, X: np.ndarray, uv: np.ndarray, intr: dict, f_scale: float = 5.0) -> np.ndarray:
    """Pose minimising robust (soft-L1) reprojection error of points X onto pixels uv, from p."""
    return least_squares(lambda q: (project(q, X, intr) - uv).ravel(), p, loss="soft_l1", f_scale=f_scale).x


def pose_from_landmarks(lm: np.ndarray, depth_m: np.ndarray, J: np.ndarray, intr: dict) -> tuple[np.ndarray, float] | None:
    """(pose, median landmark px) of one frame: landmarks lifted by depth -> Kabsch onto joints J, then 2-D refined."""
    P = deproject(lm, depth_at(depth_m, lm), intr)
    ok = np.isfinite(P).all(1)
    if ok.sum() < 6:
        return None
    for _ in range(3):                               # drop landmarks > 3 cm off the rigid fit
        R, t = kabsch(J[ok], P[ok])
        keep = ok & (np.linalg.norm(J @ R.T + t - P, axis=1) < 0.03)
        if keep.sum() < 6:
            break
        ok = keep
    p = solve_2d(np.r_[Rot.from_matrix(R).as_rotvec(), t], J[ok], lm[ok], intr)
    return p, float(np.median(np.linalg.norm(project(p, J[ok], intr) - lm[ok], axis=1)))


def refine_on_dots(p: np.ndarray, frames: list[tuple[np.ndarray, np.ndarray]], intr: dict,
                   gates=(15, 10, 8, 6, 6), min_pairs: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Refine a pose on marker dots: each round pairs every projected Vicon marker with its nearest dot inside
    the gate and re-solves. frames = [(markers (N,3) m, dots (n,2) px)], pooled. Returns (pose, final residuals px)."""
    trees = [(V, cKDTree(d)) for V, d in frames if len(V) and len(d)]
    res = np.array([])
    for gate in gates:
        X, U = [], []
        for V, tree in trees:
            d, j = tree.query(project(p, V, intr))
            m = d < gate
            X.append(V[m]); U.append(tree.data[j[m]])
        X, U = np.concatenate(X) if X else np.zeros((0, 3)), np.concatenate(U) if U else np.zeros((0, 2))
        if len(X) < min_pairs:
            break
        p = solve_2d(p, X, U, intr, f_scale=2.0)
        res = np.linalg.norm(project(p, X, intr) - U, axis=1)
    return p, res


def is_moving(grey: np.ndarray, frac: float = 0.05) -> bool:
    """True if the view shifts: more than frac of the frames are displaced > 1 px from the temporal median
    image (phase correlation). P16/Trial2/HOI: head-mounted cam1 0.22, tripod cam2 0.00. Per-frame hand
    poses cannot decide this: one hand pins the camera centre only to ~10 cm."""
    ref = np.median(grey, axis=0)
    F = np.conj(np.fft.fft2(ref - ref.mean()))
    h, w = ref.shape
    moved = 0
    for g in grey:
        A = np.fft.fft2(g - g.mean()) * F
        y, x = np.unravel_index(np.argmax(np.fft.ifft2(A / (np.abs(A) + 1e-9)).real), A.shape)
        moved += np.hypot(x if x < w // 2 else x - w, y if y < h // 2 else y - h) > 1
    return moved / len(grey) > frac


def to_matrix(p: np.ndarray) -> np.ndarray:
    """4x4 T_vicon_from_cam (metres) of a Vicon -> camera pose vector."""
    R = Rot.from_rotvec(p[:3]).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ p[3:]
    return T


def pooled_landmark_pose(p: np.ndarray, hands: list[list[np.ndarray]], J: list[np.ndarray], intr: dict,
                         gates=(80, 40, 20, 12, 12)) -> tuple[np.ndarray, np.ndarray]:
    """Static camera: refine one pose on every frame's landmarks. Per frame the detected hand closest to the
    prediction is used, and landmarks further than 2x the gate are dropped. Returns (pose, residuals px)."""
    res = np.array([])
    for gate in gates:
        X, U = [], []
        for hs, Jk in zip(hands, J):
            if not hs:
                continue
            pred = project(p, Jk, intr)
            err = [np.median(np.linalg.norm(h - pred, axis=1)) for h in hs]
            h = hs[int(np.argmin(err))]
            keep = np.linalg.norm(h - pred, axis=1) < 2 * gate
            if min(err) <= 2 * gate and keep.sum() >= 6:
                X.append(Jk[keep]); U.append(h[keep])
        if not X:
            break
        X, U = np.concatenate(X), np.concatenate(U)
        p = solve_2d(p, X, U, intr)
        res = np.linalg.norm(project(p, X, intr) - U, axis=1)
    return p, res
