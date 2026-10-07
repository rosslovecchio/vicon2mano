"""Fit MANO parameters to Vicon 3D marker positions.

Optimisation objective (per frame)
-----------------------------------
  L = w_joint * sum_j ||J_j(θ,β) - m_{a(j)}||²
    + w_pose  * ||θ - θ_prior||²
    + w_shape * ||β||²
    + w_temp  * ||θ_t - θ_{t-1}||²   (temporal smoothness, sequence mode)

where J_j(θ,β) are the MANO joint positions predicted by the model and
m_{a(j)} is the Vicon marker assigned to joint j.

References
----------
- Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies
  Together", SIGGRAPH Asia 2017  (MANO)
- Bogo et al., "Keep it SMPL: Automatic Estimation of 3D Human Pose and
  Shape from a Single Image", ECCV 2016  (SMPLify optimisation recipe)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import inspect
    import numpy as _np
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # chumpy (a smplx dependency) uses removed numpy type aliases
    _np_compat = {"int": int, "float": float, "bool": bool, "complex": complex,
                  "object": object, "str": str, "unicode": str}
    for _attr, _builtin in _np_compat.items():
        if not hasattr(_np, _attr):
            setattr(_np, _attr, _builtin)
    import smplx
    _SMPLX_AVAILABLE = True
except ImportError:
    _SMPLX_AVAILABLE = False

from vicon2mano.core.correspondence import MANO_JOINT_NAMES, get_assignment

# ---------------------------------------------------------------------------
# smplx MANO output → repo 21-joint convention
#
# smplx returns 16 joints in MANO's native kinematic-tree order:
#   0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
# The repo convention (MANO_JOINT_NAMES) is wrist, then thumb→pinky with a
# fingertip after each finger.  Fingertips are not skeleton joints in MANO;
# they are taken from fixed mesh vertices.
# ---------------------------------------------------------------------------

_SMPLX_JOINT_ORDER = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
_REPO_JOINT_SLOTS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
_TIP_VERTEX_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky
_REPO_TIP_SLOTS = [4, 8, 12, 16, 20]


def mano_output_to_joints21(out) -> "torch.Tensor":
    """Map an smplx MANO forward output to (B, 21, 3) in repo joint order."""
    j21 = out.joints.new_zeros((out.joints.shape[0], 21, 3))
    j21[:, _REPO_JOINT_SLOTS] = out.joints[:, _SMPLX_JOINT_ORDER]
    j21[:, _REPO_TIP_SLOTS] = out.vertices[:, _TIP_VERTEX_IDS]
    return j21


@dataclass
class FitConfig:
    mano_model_path: str = "../clean_kinematics/mano_v1_2/models"  # directory with MANO_RIGHT.pkl
    hand_side: str = "right"             # "right" | "left"

    # optimisation
    n_iters_shape: int = 100             # stage 1: shape only
    n_iters_pose: int = 300              # stage 2: pose + shape
    lr: float = 3e-3
    # Adam caps per-step movement at the learning rate, so each parameter
    # group needs a rate matched to how far it must travel: PCA pose
    # coefficients reach ±2 for real grasps, global orientation needs
    # fractions of a radian after Kabsch init, transl only millimetres.
    # A single small lr leaves the pose stuck near the flat mean pose.
    lr_pose: float = 5e-2
    lr_orient: float = 1e-2

    # loss weights — the joint term is squared *metres* (~1e-4–1e-3 at a
    # good fit) while pose PCA coefficients reach ±3 for real grasps, so the
    # pose/temporal weights must be small enough that articulating costs
    # less than the joint error it removes (w_pose=1e-3 made the flat hand
    # the global optimum).
    w_joint: float = 1.0
    w_pose: float = 1e-5
    w_shape: float = 1e-2
    w_temp: float = 5e-3                 # temporal smoothness (param space)
    # Joint-space *acceleration* penalty: gauge-invariant and axis-angle-
    # wraparound-safe, and constant-velocity motion costs zero.  Weight is
    # sized so even a vigorous 100 ms flexion (~4 mm/frame², cost ~3e-5)
    # stays well under the data term (~3e-4) while solver teleports
    # (~1e-1 m/frame², cost ~0.02) are crushed — at 20.0 fast gestures were
    # visibly damped (articulation corr 0.98 → 0.84).
    w_accel: float = 2.0

    # outlier rescue: frames whose mean joint acceleration exceeds this are
    # re-initialised from clean neighbours and locally re-optimised
    outlier_accel_m: float = 5e-3        # 5 mm/frame²
    n_iters_rescue: int = 200

    # unit conversion: multiply markers to reach metres (MANO is in metres)
    marker_scale: float = 1e-3          # Vicon → mm → m

    use_pca: bool = True
    n_pca_comps: int = 6                # PCA components for hand pose

    # long sequences: shape stage subsamples to at most this many frames;
    # the pose stage optimises in independent chunks of this length
    # (full-batch optimisation of a whole recording would exhaust memory)
    shape_max_frames: int = 2000
    chunk_frames: int = 2000


@dataclass
class FitResult:
    """Per-frame MANO parameters."""
    global_orient: np.ndarray  # (T, 3)   axis-angle
    hand_pose: np.ndarray      # (T, 45)  axis-angle or (T, n_pca) if use_pca
    transl: np.ndarray         # (T, 3)   global translation in metres
    betas: np.ndarray          # (10,)    shape (shared across frames)
    joints: np.ndarray         # (T, 21, 3) predicted joints in metres
    vertices: np.ndarray       # (T, 778, 3) mesh vertices
    assignment: np.ndarray     # (21,) or (T, 21) marker→joint index map


class MANOFitter:
    """Fit MANO to a sequence of Vicon 3D marker frames.

    Usage::

        fitter = MANOFitter(config)
        result = fitter.fit(markers, marker_labels)
        # markers: (T, 22, 3) in mm
    """

    def __init__(self, config: FitConfig | None = None, *, labeler=None):
        self.cfg = config or FitConfig()
        self.labeler = labeler  # Optional[DeepLabeler]; None → Hungarian fallback
        if not _SMPLX_AVAILABLE:
            raise ImportError(
                "smplx is required: pip install smplx\n"
                "Download MANO model weights from https://mano.is.tue.mpg.de/"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        markers: np.ndarray,              # (T, N, 3) mm
        marker_labels: list[str] | None = None,
        *,
        verbose: bool = True,
    ) -> FitResult:
        """Fit MANO to a sequence of marker frames.

        Args:
            markers:       (T, N, 3) Vicon marker positions in millimetres.
            marker_labels: optional list of N label strings for seeded
                           correspondence initialisation.
            verbose:       print optimisation progress.
        """
        T, N, _ = markers.shape
        markers_m = markers.astype(np.float32) * self.cfg.marker_scale  # → metres

        # ---------- stage 0: coarse correspondences ----------
        # zero-pose joints are identical for every frame — forward once
        with torch.no_grad():
            init_joints = self._forward_np(
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, self._pose_dim()), dtype=np.float32),
                np.zeros(10, dtype=np.float32),
            )  # (1, 21, 3)
        init_joints = np.broadcast_to(init_joints, (T, 21, 3))

        assign = get_assignment(
            markers_m,
            init_joints,
            labeler=self.labeler,
            per_frame=False,
            marker_labels=marker_labels,
            side=self.cfg.hand_side,
        )

        if verbose:
            matched = (assign >= 0).sum()
            print(f"[vicon2mano] {matched}/21 joints matched to markers")

        # ---------- stage 1: optimise shape only (mean pose) --------------
        stride = max(1, T // self.cfg.shape_max_frames)
        betas = self._optimise_shape(markers_m[::stride], assign, verbose=verbose)

        # ---------- stage 2: optimise pose per frame, in chunks -----------
        chunk = self.cfg.chunk_frames
        gos, hps, trs = [], [], []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            if verbose and T > chunk:
                print(f"[vicon2mano] pose stage: frames {start}–{end} of {T}")
            go, hp, tr = self._optimise_pose(
                markers_m[start:end], assign, betas, verbose=verbose
            )
            gos.append(go)
            hps.append(hp)
            trs.append(tr)
        global_orient = np.concatenate(gos, axis=0)
        hand_pose = np.concatenate(hps, axis=0)
        transl = np.concatenate(trs, axis=0)

        # ---------- stage 3: rescue outlier frames ------------------------
        global_orient, hand_pose, transl = self._refine_outliers(
            markers_m, assign, global_orient, hand_pose, transl, betas,
            verbose=verbose,
        )

        # ---------- collect final outputs ---------------------------------
        joints_np, verts_np = self._collect_outputs(
            global_orient, hand_pose, transl, betas
        )

        return FitResult(
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
            betas=betas,
            joints=joints_np,
            vertices=verts_np,
            assignment=assign,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        model_path = Path(self.cfg.mano_model_path)
        # smplx.create accepts either a .pkl file or a parent dir containing
        # a mano/ subdirectory.  If the user points directly at the directory
        # holding MANO_RIGHT.pkl / MANO_LEFT.pkl, resolve to the file.
        if model_path.is_dir() and not (model_path / "mano").exists():
            side_str = "RIGHT" if self.cfg.hand_side == "right" else "LEFT"
            pkl = model_path / f"MANO_{side_str}.pkl"
            if not pkl.exists():
                raise FileNotFoundError(f"MANO weights not found at {pkl}")
            model_path = pkl
        return smplx.create(
            str(model_path),
            model_type="mano",
            is_rhand=(self.cfg.hand_side == "right"),
            use_pca=self.cfg.use_pca,
            num_pca_comps=self.cfg.n_pca_comps,
            flat_hand_mean=True,
            batch_size=1,
        ).to(self.device)

    def _forward_np(
        self,
        global_orient: np.ndarray,  # (T, 3)
        hand_pose: np.ndarray,       # (T, 45) or (T, n_pca)
        betas: np.ndarray,           # (10,)
    ) -> np.ndarray:
        T = global_orient.shape[0]
        results = []
        bs = 32  # process in mini-batches to avoid OOM on long sequences
        for start in range(0, T, bs):
            sl = slice(start, min(start + bs, T))
            go = torch.tensor(global_orient[sl], dtype=torch.float32, device=self.device)
            hp = torch.tensor(hand_pose[sl], dtype=torch.float32, device=self.device)
            out = self._model(
                global_orient=go,
                hand_pose=hp,
                betas=torch.tensor(betas, dtype=torch.float32, device=self.device)
                      .unsqueeze(0).expand(go.shape[0], -1),
            )
            results.append(mano_output_to_joints21(out).detach().cpu().numpy())
        return np.concatenate(results, axis=0)

    def _optimise_shape(
        self,
        markers_m: np.ndarray,  # (T, N, 3)
        assign: np.ndarray,     # (21,)
        *,
        verbose: bool,
    ) -> np.ndarray:
        cfg = self.cfg
        betas = nn.Parameter(torch.zeros(10, device=self.device))

        T = markers_m.shape[0]
        target = self._assigned_targets(markers_m, assign)  # (T, 21, 3) or NaN
        target_t = torch.tensor(target, device=self.device)
        mask = torch.isfinite(target_t).all(-1)  # (T, 21)

        # MANO is rooted at the origin; Vicon markers live in world
        # coordinates, so a per-frame translation must be fitted too.
        transl = nn.Parameter(
            torch.tensor(self._init_transl(target), device=self.device)
        )
        opt = torch.optim.Adam([betas, transl], lr=cfg.lr)

        betas_expand = betas.unsqueeze(0).expand(T, -1)

        for it in range(cfg.n_iters_shape):
            opt.zero_grad()
            out = self._model(
                global_orient=torch.zeros(T, 3, device=self.device),
                hand_pose=torch.zeros(T, self._pose_dim(), device=self.device),
                transl=transl,
                betas=betas_expand,
            )
            pred = mano_output_to_joints21(out)  # (T, 21, 3)
            diff = (pred - target_t)[mask]
            loss = cfg.w_joint * (diff ** 2).sum(-1).mean()
            loss += cfg.w_shape * (betas ** 2).sum()
            loss.backward()
            opt.step()
            if verbose and (it + 1) % 50 == 0:
                print(f"  shape stage  iter {it+1:3d}  loss={loss.item():.6f}")

        return betas.detach().cpu().numpy()

    def _optimise_pose(
        self,
        markers_m: np.ndarray,   # (T, N, 3)
        assign: np.ndarray,      # (21,)
        betas: np.ndarray,       # (10,)
        *,
        verbose: bool,
        target_override: np.ndarray | None = None,  # (T, 21, 3), NaN where unassigned
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``target_override`` lets a caller substitute per-marker
        offset-corrected targets (see ``strategies.mano.relabel``'s bone-
        segment-frame marker offsets) for the raw marker positions
        ``_assigned_targets`` would otherwise use -- the fitter's objective
        is unchanged (still ``||J_j - target_j||^2``), only what counts as
        "target" differs. Markers sit on skin, not joint centres, and the
        offset is systematic (worst at MCP joints, ~20-25mm, vs ~5-16mm at
        PIP/DIP -- measured directly on P10/Trial2 Hands only); this is the
        hook a two-pass calibrate-then-refit workflow needs without
        duplicating this method's optimisation loop (temporal smoothness,
        pose prior, accel penalty, scheduler all stay shared)."""
        cfg = self.cfg
        T = markers_m.shape[0]
        pd = self._pose_dim()

        target = target_override if target_override is not None else self._assigned_targets(markers_m, assign)
        target_t = torch.tensor(target, device=self.device)
        mask = torch.isfinite(target_t).all(-1)

        # Kabsch initialisation: gradient descent alone cannot recover a
        # large global rotation from a zero start, so align the zero-pose
        # joints to the assigned markers per frame first.
        init_go, init_tr = self._init_global_pose(target, betas)
        global_orient = nn.Parameter(torch.tensor(init_go, device=self.device))
        hand_pose = nn.Parameter(torch.zeros(T, pd, device=self.device))
        transl = nn.Parameter(torch.tensor(init_tr, device=self.device))
        betas_t = torch.tensor(betas, device=self.device).unsqueeze(0).expand(T, -1)

        opt = torch.optim.Adam([
            {"params": [hand_pose], "lr": cfg.lr_pose},
            {"params": [global_orient], "lr": cfg.lr_orient},
            {"params": [transl], "lr": cfg.lr},
        ])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_iters_pose, eta_min=cfg.lr * 0.1
        )

        for it in range(cfg.n_iters_pose):
            opt.zero_grad()
            out = self._model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                transl=transl,
                betas=betas_t,
            )
            pred = mano_output_to_joints21(out)
            diff = (pred - target_t)[mask]
            loss = cfg.w_joint * (diff ** 2).sum(-1).mean()
            loss += cfg.w_pose * (hand_pose ** 2).sum(-1).mean()
            # temporal smoothness
            if T > 1:
                loss += cfg.w_temp * ((hand_pose[1:] - hand_pose[:-1]) ** 2).mean()
                loss += cfg.w_temp * ((global_orient[1:] - global_orient[:-1]) ** 2).mean()
                loss += cfg.w_temp * ((transl[1:] - transl[:-1]) ** 2).mean()
            if T > 2:
                accel = pred[2:] - 2 * pred[1:-1] + pred[:-2]
                loss += cfg.w_accel * (accel ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
            sched.step()
            if verbose and (it + 1) % 50 == 0:
                print(f"  pose  stage  iter {it+1:3d}  loss={loss.item():.6f}")

        return (
            global_orient.detach().cpu().numpy(),
            hand_pose.detach().cpu().numpy(),
            transl.detach().cpu().numpy(),
        )

    def _joints_np(
        self,
        global_orient: np.ndarray,  # (T, 3)
        hand_pose: np.ndarray,      # (T, pose_dim)
        transl: np.ndarray,         # (T, 3)
        betas: np.ndarray,          # (10,)
    ) -> np.ndarray:
        """Batched no-grad forward → joints21 (T, 21, 3) only."""
        T = global_orient.shape[0]
        js = []
        bs = 512
        with torch.no_grad():
            for start in range(0, T, bs):
                sl = slice(start, min(start + bs, T))
                go = torch.tensor(global_orient[sl], device=self.device)
                out = self._model(
                    global_orient=go,
                    hand_pose=torch.tensor(hand_pose[sl], device=self.device),
                    transl=torch.tensor(transl[sl], device=self.device),
                    betas=torch.tensor(betas, device=self.device)
                          .unsqueeze(0).expand(go.shape[0], -1),
                )
                js.append(mano_output_to_joints21(out).cpu().numpy())
        return np.concatenate(js, axis=0)

    def _refine_outliers(
        self,
        markers_m: np.ndarray,      # (T, N, 3)
        assign: np.ndarray,         # (21,)
        global_orient: np.ndarray,  # (T, 3)
        hand_pose: np.ndarray,      # (T, pose_dim)
        transl: np.ndarray,         # (T, 3)
        betas: np.ndarray,          # (10,)
        *,
        verbose: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and re-fit frames where the solution teleports.

        Per-frame optimisation occasionally converges to a basin far from
        its neighbours (mean joint acceleration ≫ what markers support).
        Such episodes are re-initialised by interpolating the parameters of
        the clean frames flanking them, then locally re-optimised.  The new
        solution is only accepted if the marker residual does not get worse.
        """
        from scipy.spatial.transform import Rotation, Slerp

        cfg = self.cfg
        T = markers_m.shape[0]
        if T < 5:
            return global_orient, hand_pose, transl

        joints = self._joints_np(global_orient, hand_pose, transl, betas)
        accel = np.linalg.norm(
            joints[2:] - 2 * joints[1:-1] + joints[:-2], axis=-1
        ).mean(-1)                                        # (T-2,) metres
        bad = np.zeros(T, dtype=bool)
        bad[1:-1] = accel > cfg.outlier_accel_m
        if not bad.any():
            return global_orient, hand_pose, transl
        # widen each detection so the episode includes its entry/exit frames
        bad = np.convolve(bad.astype(float), np.ones(5), mode="same") > 0

        target = self._assigned_targets(markers_m, assign)
        episodes = []
        t = 0
        while t < T:
            if bad[t]:
                a = t
                while t < T and bad[t]:
                    t += 1
                episodes.append((a, t - 1))
            else:
                t += 1
        if verbose:
            n_frames = int(bad.sum())
            print(f"[vicon2mano] rescue stage: {len(episodes)} episodes "
                  f"({n_frames} frames) above "
                  f"{cfg.outlier_accel_m * 1000:.0f} mm/frame² acceleration")

        fixed = 0
        for a, b in episodes:
            ca = a - 1 if a > 0 else None
            cb = b + 1 if b < T - 1 else None
            n = b - a + 1
            if ca is None and cb is None:
                continue

            # init: interpolate params between the clean flanking frames
            if ca is not None and cb is not None:
                key = Rotation.from_rotvec([global_orient[ca],
                                            global_orient[cb]])
                slerp = Slerp([0, n + 1], key)
                init_go = slerp(np.arange(1, n + 1)).as_rotvec()
                w = (np.arange(1, n + 1) / (n + 1))[:, None]
                init_hp = (1 - w) * hand_pose[ca] + w * hand_pose[cb]
                init_tr = (1 - w) * transl[ca] + w * transl[cb]
            else:
                src = ca if ca is not None else cb
                init_go = np.repeat(global_orient[src][None], n, axis=0)
                init_hp = np.repeat(hand_pose[src][None], n, axis=0)
                init_tr = np.repeat(transl[src][None], n, axis=0)

            go_p = nn.Parameter(torch.tensor(
                init_go.astype(np.float32), device=self.device))
            hp_p = nn.Parameter(torch.tensor(
                init_hp.astype(np.float32), device=self.device))
            tr_p = nn.Parameter(torch.tensor(
                init_tr.astype(np.float32), device=self.device))
            betas_t = torch.tensor(betas, device=self.device) \
                .unsqueeze(0).expand(n, -1)

            tgt = torch.tensor(target[a:b + 1], device=self.device)
            mask = torch.isfinite(tgt).all(-1)
            ctx_pre = torch.tensor(joints[max(0, a - 2):a],
                                   device=self.device)
            ctx_post = torch.tensor(joints[b + 1:b + 3], device=self.device)

            opt = torch.optim.Adam([
                {"params": [hp_p], "lr": cfg.lr_pose},
                {"params": [go_p], "lr": cfg.lr_orient},
                {"params": [tr_p], "lr": cfg.lr},
            ])
            for _ in range(cfg.n_iters_rescue):
                opt.zero_grad()
                out = self._model(global_orient=go_p, hand_pose=hp_p,
                                  transl=tr_p, betas=betas_t)
                pred = mano_output_to_joints21(out)
                loss = cfg.w_joint * ((pred - tgt)[mask] ** 2).sum(-1).mean()
                loss += cfg.w_pose * (hp_p ** 2).sum(-1).mean()
                seq = torch.cat([ctx_pre, pred, ctx_post], dim=0)
                if seq.shape[0] > 2:
                    acc = seq[2:] - 2 * seq[1:-1] + seq[:-2]
                    loss += cfg.w_accel * (acc ** 2).sum(-1).mean()
                loss.backward()
                opt.step()

            # accept only if the marker residual did not get worse
            with torch.no_grad():
                out = self._model(global_orient=go_p, hand_pose=hp_p,
                                  transl=tr_p, betas=betas_t)
                pred = mano_output_to_joints21(out).cpu().numpy()
            old_res = np.nanmean(np.linalg.norm(
                joints[a:b + 1] - target[a:b + 1], axis=-1))
            new_res = np.nanmean(np.linalg.norm(
                pred - target[a:b + 1], axis=-1))
            if new_res <= old_res * 1.05:
                global_orient[a:b + 1] = go_p.detach().cpu().numpy()
                hand_pose[a:b + 1] = hp_p.detach().cpu().numpy()
                transl[a:b + 1] = tr_p.detach().cpu().numpy()
                joints[a:b + 1] = pred
                fixed += 1

        if verbose and episodes:
            print(f"[vicon2mano] rescue stage: re-fit accepted for "
                  f"{fixed}/{len(episodes)} episodes")
        return global_orient, hand_pose, transl

    def _collect_outputs(
        self,
        global_orient: np.ndarray,  # (T, 3)
        hand_pose: np.ndarray,      # (T, pose_dim)
        transl: np.ndarray,         # (T, 3)
        betas: np.ndarray,          # (10,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched no-grad forward → (joints21 (T,21,3), vertices (T,778,3))."""
        T = global_orient.shape[0]
        js, vs = [], []
        bs = 512
        with torch.no_grad():
            for start in range(0, T, bs):
                sl = slice(start, min(start + bs, T))
                go = torch.tensor(global_orient[sl], device=self.device)
                out = self._model(
                    global_orient=go,
                    hand_pose=torch.tensor(hand_pose[sl], device=self.device),
                    transl=torch.tensor(transl[sl], device=self.device),
                    betas=torch.tensor(betas, device=self.device)
                          .unsqueeze(0).expand(go.shape[0], -1),
                    return_verts=True,
                )
                js.append(mano_output_to_joints21(out).cpu().numpy())
                vs.append(out.vertices.cpu().numpy())
        return np.concatenate(js, axis=0), np.concatenate(vs, axis=0)

    def _init_global_pose(
        self,
        target: np.ndarray,  # (T, 21, 3) with NaN for unassigned joints
        betas: np.ndarray,   # (10,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame rigid (Kabsch) alignment of zero-pose joints to markers.

        Returns (global_orient (T, 3) axis-angle, transl (T, 3)).
        smplx rotates the model about its root joint, so the translation is
        derived consistently with that convention.
        """
        from scipy.spatial.transform import Rotation

        T = target.shape[0]
        with torch.no_grad():
            out = self._model(
                global_orient=torch.zeros(1, 3, device=self.device),
                hand_pose=torch.zeros(1, self._pose_dim(), device=self.device),
                betas=torch.tensor(betas, device=self.device).unsqueeze(0),
            )
            ref = mano_output_to_joints21(out)[0].cpu().numpy()  # (21, 3)
        root = ref[0]  # wrist — smplx applies global_orient about the root

        # Align using only joints that are rigid w.r.t. the palm (wrist +
        # finger MCPs).  Including PIP/DIP markers tilts the alignment on
        # flexed frames — the flat template gets rotated toward the curled
        # fingers, which leaves pose optimisation in a local minimum where
        # the hand stays extended.
        rigid = np.zeros(21, dtype=bool)
        rigid[[0, 5, 9, 13, 17]] = True  # wrist, index/middle/ring/pinky mcp

        go = np.zeros((T, 3), dtype=np.float32)
        tr = self._init_transl(target)
        for t in range(T):
            finite = np.isfinite(target[t]).all(-1)
            valid = finite & rigid
            if valid.sum() < 3:
                valid = finite           # fallback: use whatever exists
            if valid.sum() < 3:
                continue
            A, B = ref[valid], target[t][valid]
            rot, _ = Rotation.align_vectors(B - B.mean(0), A - A.mean(0))
            go[t] = rot.as_rotvec().astype(np.float32)
            tr[t] = (
                B.mean(0) - rot.apply(A.mean(0) - root) - root
            ).astype(np.float32)

        # Unwrap: axis-angle is 2π-periodic along its axis, and align_vectors
        # always returns the minimal (|r| ≤ π) representative, so a rotation
        # crossing π flips to the antipodal vector between adjacent frames.
        # Pick whichever representative is closer to the previous frame so
        # the parameter-space temporal terms see a continuous trajectory.
        for t in range(1, T):
            n = np.linalg.norm(go[t])
            if n > 1e-8:
                cand = go[t] * (1.0 - 2.0 * np.pi / n)
                if (np.linalg.norm(cand - go[t - 1])
                        < np.linalg.norm(go[t] - go[t - 1])):
                    go[t] = cand
        return go, tr

    def _init_transl(self, target: np.ndarray) -> np.ndarray:
        """Per-frame centroid of the assigned markers — translation init.

        target: (T, 21, 3) with NaN for unassigned joints."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tc = np.nanmean(target, axis=1)  # (T, 3)
        return np.nan_to_num(tc).astype(np.float32)

    def _assigned_targets(
        self,
        markers_m: np.ndarray,  # (T, N, 3)
        assign: np.ndarray,     # (21,)
    ) -> np.ndarray:
        """Build (T, 21, 3) target array; NaN where assign[j] == -1."""
        T = markers_m.shape[0]
        out = np.full((T, 21, 3), np.nan, dtype=np.float32)
        for j, mi in enumerate(assign):
            if mi >= 0:
                out[:, j, :] = markers_m[:, mi, :]
        return out

    def _pose_dim(self) -> int:
        return self.cfg.n_pca_comps if self.cfg.use_pca else 45
