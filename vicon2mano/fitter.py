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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import smplx
    _SMPLX_AVAILABLE = True
except ImportError:
    _SMPLX_AVAILABLE = False

from .correspondence import hungarian_assignment, label_seed, sequence_assignment


@dataclass
class FitConfig:
    mano_model_path: str = "../clean_kinematics/mano_v1_2/models"  # directory with MANO_RIGHT.pkl
    hand_side: str = "right"             # "right" | "left"

    # optimisation
    n_iters_shape: int = 100             # stage 1: shape only
    n_iters_pose: int = 200              # stage 2: pose + shape
    lr: float = 3e-3

    # loss weights
    w_joint: float = 1.0
    w_pose: float = 1e-3
    w_shape: float = 1e-2
    w_temp: float = 5e-2                 # temporal smoothness

    # unit conversion: multiply markers to reach metres (MANO is in metres)
    marker_scale: float = 1e-3          # Vicon → mm → m

    use_pca: bool = True
    n_pca_comps: int = 6                # PCA components for hand pose


@dataclass
class FitResult:
    """Per-frame MANO parameters."""
    global_orient: np.ndarray  # (T, 3)   axis-angle
    hand_pose: np.ndarray      # (T, 45)  axis-angle or (T, n_pca) if use_pca
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

    def __init__(self, config: FitConfig | None = None):
        self.cfg = config or FitConfig()
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

        # ---------- stage 0: coarse correspondences via mean pose ----------
        with torch.no_grad():
            init_joints = self._forward_np(
                np.zeros((T, 3)), np.zeros((T, 45)), np.zeros(10)
            )  # (T, 21, 3)

        if marker_labels is not None:
            seed = label_seed(marker_labels)
        else:
            seed = None

        if seed and len(seed) >= 15:
            # Build assignment array from label hints
            assign = np.full(21, -1, dtype=int)
            from .correspondence import MANO_JOINT_NAMES
            for j_idx, jname in enumerate(MANO_JOINT_NAMES):
                if jname in seed:
                    assign[j_idx] = seed[jname]
        else:
            assign = sequence_assignment(markers_m, init_joints)

        if verbose:
            matched = (assign >= 0).sum()
            print(f"[vicon2mano] {matched}/21 joints matched to markers")

        # ---------- stage 1: optimise shape only (mean pose) --------------
        betas = self._optimise_shape(markers_m, assign, verbose=verbose)

        # ---------- stage 2: optimise pose per frame ----------------------
        global_orient, hand_pose = self._optimise_pose(
            markers_m, assign, betas, verbose=verbose
        )

        # ---------- collect final outputs ---------------------------------
        with torch.no_grad():
            out = self._model(
                global_orient=torch.tensor(global_orient, device=self.device),
                hand_pose=torch.tensor(hand_pose, device=self.device),
                betas=torch.tensor(betas, device=self.device).unsqueeze(0).expand(T, -1),
                return_verts=True,
            )
        joints_np = out.joints[:, :21].cpu().numpy()
        verts_np = out.vertices.cpu().numpy()

        return FitResult(
            global_orient=global_orient,
            hand_pose=hand_pose,
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
            sl = slice(start, start + bs)
            out = self._model(
                global_orient=torch.tensor(global_orient[sl], device=self.device),
                hand_pose=torch.tensor(hand_pose[sl], device=self.device),
                betas=torch.tensor(betas, device=self.device)
                      .unsqueeze(0).expand(sl.stop - sl.start if sl.stop else T, -1),
            )
            results.append(out.joints[:, :21].detach().cpu().numpy())
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
        opt = torch.optim.Adam([betas], lr=cfg.lr)

        T = markers_m.shape[0]
        target = self._assigned_targets(markers_m, assign)  # (T, 21, 3) or NaN
        target_t = torch.tensor(target, device=self.device)
        mask = torch.isfinite(target_t).all(-1)  # (T, 21)

        betas_expand = betas.unsqueeze(0).expand(T, -1)

        for it in range(cfg.n_iters_shape):
            opt.zero_grad()
            out = self._model(
                global_orient=torch.zeros(T, 3, device=self.device),
                hand_pose=torch.zeros(T, self._pose_dim(), device=self.device),
                betas=betas_expand,
            )
            pred = out.joints[:, :21]  # (T, 21, 3)
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
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        T = markers_m.shape[0]
        pd = self._pose_dim()

        global_orient = nn.Parameter(torch.zeros(T, 3, device=self.device))
        hand_pose = nn.Parameter(torch.zeros(T, pd, device=self.device))
        betas_t = torch.tensor(betas, device=self.device).unsqueeze(0).expand(T, -1)

        opt = torch.optim.Adam([global_orient, hand_pose], lr=cfg.lr)

        target = self._assigned_targets(markers_m, assign)
        target_t = torch.tensor(target, device=self.device)
        mask = torch.isfinite(target_t).all(-1)

        for it in range(cfg.n_iters_pose):
            opt.zero_grad()
            out = self._model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas_t,
            )
            pred = out.joints[:, :21]
            diff = (pred - target_t)[mask]
            loss = cfg.w_joint * (diff ** 2).sum(-1).mean()
            loss += cfg.w_pose * (hand_pose ** 2).sum(-1).mean()
            # temporal smoothness
            if T > 1:
                loss += cfg.w_temp * ((hand_pose[1:] - hand_pose[:-1]) ** 2).mean()
                loss += cfg.w_temp * ((global_orient[1:] - global_orient[:-1]) ** 2).mean()
            loss.backward()
            opt.step()
            if verbose and (it + 1) % 50 == 0:
                print(f"  pose  stage  iter {it+1:3d}  loss={loss.item():.6f}")

        return (
            global_orient.detach().cpu().numpy(),
            hand_pose.detach().cpu().numpy(),
        )

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
