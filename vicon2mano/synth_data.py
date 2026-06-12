"""Synthetic training data generator for the deep marker labeler.

Generates (voxel_grid, target_heatmaps) pairs by:
1. Sampling random MANO hand poses.
2. Placing one marker per joint with small Gaussian noise.
3. Randomly dropping out up to dropout_max markers (simulates occlusion).
4. Randomly adding up to ghost_max spurious markers (simulates noise).
5. Shuffling marker order (so the network cannot exploit ordering).

All data is voxelised on-the-fly during training to keep the on-disk
representation small.  The HDF5 file stores raw marker positions and ground-
truth joint positions; the DataLoader worker performs voxelisation.

On-disk format (data/synth_labeler.h5):
    /marker_positions  (N, MAX_MARKERS, 3)  float32  metres
                       rows beyond marker_counts[i] are zero-padded
    /marker_counts     (N,)                 int32    actual marker count
    /joint_positions   (N, 21, 3)           float32  metres

Typical file size: ~300 MB for 500 k samples.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_MARKERS = 30  # upper bound on markers per frame (21 base + up to 9 slack)


@dataclass
class SynthConfig:
    mano_model_path: str = "../clean_kinematics/mano_v1_2/models"
    hand_side: str = "right"

    n_samples: int = 500_000

    # Marker placement noise (metres)
    marker_noise_m: float = 3e-3         # 3 mm std per axis

    # Random pose: std of per-joint axis-angle perturbation (radians)
    pose_std: float = 0.5

    # Marker dropout (simulates occlusion)
    dropout_min: int = 0
    dropout_max: int = 6

    # Ghost markers (simulates Vicon noise)
    ghost_min: int = 0
    ghost_max: int = 3
    ghost_range_m: float = 0.05          # ghost markers placed within ±5 cm of bbox

    output_path: str = "data/synth_labeler.h5"
    chunk_size: int = 512                # HDF5 write chunk
    seed: int = 42

    # How many poses to sample per MANO forward batch (memory vs speed)
    mano_batch_size: int = 512


# ---------------------------------------------------------------------------
# Heatmap target construction
# ---------------------------------------------------------------------------

def build_target_heatmaps(
    joint_positions: np.ndarray,
    centroid: np.ndarray,
    *,
    res: int = 64,
    half: float = 0.22,
    sigma: float = 1.5,
) -> np.ndarray:
    """Build (21, res, res, res) Gaussian heatmap targets.

    For each joint, a Gaussian blob centred on the joint's voxel coordinate
    is written into the corresponding channel.  The same voxelisation mapping
    as `voxelise_with_centroid` is used so that input grid and target are
    spatially aligned.

    Args:
        joint_positions: (21, 3) joint positions in metres.
        centroid:        (3,) centroid used to voxelise the marker cloud for
                         this sample (must match the centroid returned by
                         `voxelise_with_centroid`).
        res:             Grid side length.
        half:            Half-extent in metres.
        sigma:           Gaussian blob std in voxel units.

    Returns:
        heatmaps: (21, res, res, res) float32 array, values in [0, 1].
    """
    pts = joint_positions - centroid               # (21, 3) centred
    vf = (pts / half + 1.0) * 0.5 * res           # (21, 3) float voxel coords

    heatmaps = np.zeros((21, res, res, res), dtype=np.float32)
    r = int(np.ceil(3.0 * sigma))

    for j in range(21):
        if not np.isfinite(joint_positions[j]).all():
            continue  # unobserved joint (e.g. fingertip) → zero channel
        fx, fy, fz = float(vf[j, 0]), float(vf[j, 1]), float(vf[j, 2])
        ix, iy, iz = int(round(fx)), int(round(fy)), int(round(fz))

        x0, x1 = max(0, ix - r), min(res, ix + r + 1)
        y0, y1 = max(0, iy - r), min(res, iy + r + 1)
        z0, z1 = max(0, iz - r), min(res, iz + r + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue  # joint outside grid (can happen for fingertips)

        gx = np.exp(-0.5 * ((np.arange(x0, x1) - fx) / sigma) ** 2)
        gy = np.exp(-0.5 * ((np.arange(y0, y1) - fy) / sigma) ** 2)
        gz = np.exp(-0.5 * ((np.arange(z0, z1) - fz) / sigma) ** 2)
        blob = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

        heatmaps[j, z0:z1, y0:y1, x0:x1] = np.maximum(
            heatmaps[j, z0:z1, y0:y1, x0:x1], blob
        )

    return heatmaps


# ---------------------------------------------------------------------------
# Per-sample generation (no MANO dependency — takes pre-sampled joints)
# ---------------------------------------------------------------------------

def augment_markers(
    joint_positions: np.ndarray,
    rng: np.random.Generator,
    cfg: SynthConfig,
) -> tuple[np.ndarray, int]:
    """Place, augment, and shuffle markers for one synthetic frame.

    Args:
        joint_positions: (21, 3) ground-truth joint positions in metres.
        rng:             Random generator.
        cfg:             Augmentation parameters.

    Returns:
        markers_padded: (MAX_MARKERS, 3) float32; rows beyond n_markers are
                        zero-padded.
        n_markers:      Actual number of markers (valid rows).
    """
    # 1. Place one marker per joint with Gaussian noise
    noise = rng.standard_normal((21, 3)).astype(np.float32) * cfg.marker_noise_m
    markers = (joint_positions + noise).astype(np.float32)  # (21, 3)

    # 2. Random dropout
    n_drop = int(rng.integers(cfg.dropout_min, cfg.dropout_max + 1))
    if n_drop > 0:
        keep = rng.choice(21, 21 - n_drop, replace=False)
        keep.sort()
        markers = markers[keep]

    # 3. Ghost markers — random positions inside/near the hand bbox
    n_ghost = int(rng.integers(cfg.ghost_min, cfg.ghost_max + 1))
    if n_ghost > 0:
        bbox_min = joint_positions.min(0)
        bbox_max = joint_positions.max(0)
        ghosts = rng.uniform(
            bbox_min - cfg.ghost_range_m,
            bbox_max + cfg.ghost_range_m,
            (n_ghost, 3),
        ).astype(np.float32)
        markers = np.concatenate([markers, ghosts], axis=0)

    # 4. Shuffle marker order (permutation-invariance is learned via voxelisation)
    markers = markers[rng.permutation(len(markers))]

    n_markers = len(markers)
    assert n_markers <= MAX_MARKERS, (
        f"n_markers={n_markers} exceeds MAX_MARKERS={MAX_MARKERS}; "
        "increase MAX_MARKERS or reduce ghost_max."
    )

    padded = np.zeros((MAX_MARKERS, 3), dtype=np.float32)
    padded[:n_markers] = markers
    return padded, n_markers


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(cfg: SynthConfig | None = None) -> None:
    """Generate a synthetic dataset and write it to an HDF5 file.

    Requires:
        h5py  (``pip install "vicon2mano[train]"``)
        smplx (already a core dependency)
    """
    if cfg is None:
        cfg = SynthConfig()

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for dataset generation.\n"
            "Install with: pip install 'vicon2mano[train]'"
        ) from exc

    try:
        import torch
        import smplx
    except ImportError as exc:
        raise ImportError("torch and smplx are required.") from exc

    from tqdm import tqdm

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # Load MANO on CPU (inference only; no GPU needed for generation)
    # smplx.create wants either a .pkl file or a parent dir with a mano/
    # subdirectory; resolve a plain models dir to the side-specific .pkl.
    model_path = Path(cfg.mano_model_path)
    if model_path.is_dir() and not (model_path / "mano").exists():
        side_str = "RIGHT" if cfg.hand_side == "right" else "LEFT"
        pkl = model_path / f"MANO_{side_str}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"MANO weights not found at {pkl}")
        model_path = pkl
    mano = smplx.create(
        str(model_path),
        model_type="mano",
        is_rhand=(cfg.hand_side == "right"),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=cfg.mano_batch_size,
    ).eval()

    n = cfg.n_samples
    bs = cfg.mano_batch_size
    chunk = min(cfg.chunk_size, n)  # h5py chunks must not exceed data shape

    with h5py.File(output_path, "w") as f:
        ds_markers = f.create_dataset(
            "marker_positions",
            shape=(n, MAX_MARKERS, 3),
            dtype="float32",
            chunks=(chunk, MAX_MARKERS, 3),
            compression="gzip",
            compression_opts=4,
        )
        ds_counts = f.create_dataset(
            "marker_counts",
            shape=(n,),
            dtype="int32",
            chunks=(chunk,),
        )
        ds_joints = f.create_dataset(
            "joint_positions",
            shape=(n, 21, 3),
            dtype="float32",
            chunks=(chunk, 21, 3),
            compression="gzip",
            compression_opts=4,
        )

        written = 0
        with tqdm(total=n, desc="Generating samples") as pbar:
            while written < n:
                batch = min(bs, n - written)

                # Sample random MANO poses
                global_orient = (
                    rng.standard_normal((batch, 3)).astype(np.float32) * cfg.pose_std
                )
                hand_pose = (
                    rng.standard_normal((batch, 45)).astype(np.float32) * cfg.pose_std
                )
                betas = (
                    rng.standard_normal((batch, 10)).astype(np.float32) * 0.5
                )

                # Expand MANO batch if needed (model was created with fixed batch_size)
                if batch < bs:
                    # Pad up to model's expected batch size
                    pad = bs - batch
                    global_orient = np.concatenate(
                        [global_orient, np.zeros((pad, 3), dtype=np.float32)], axis=0
                    )
                    hand_pose = np.concatenate(
                        [hand_pose, np.zeros((pad, 45), dtype=np.float32)], axis=0
                    )
                    betas = np.concatenate(
                        [betas, np.zeros((pad, 10), dtype=np.float32)], axis=0
                    )

                import torch as _torch
                from .fitter import mano_output_to_joints21
                with _torch.no_grad():
                    out = mano(
                        global_orient=_torch.tensor(global_orient),
                        hand_pose=_torch.tensor(hand_pose),
                        betas=_torch.tensor(betas),
                    )
                joints_batch = (
                    mano_output_to_joints21(out)[:batch].cpu().numpy()
                )  # (batch, 21, 3) repo joint order incl. fingertip vertices

                for i in range(batch):
                    padded, n_markers = augment_markers(joints_batch[i], rng, cfg)
                    idx = written + i
                    ds_markers[idx] = padded
                    ds_counts[idx] = n_markers
                    ds_joints[idx] = joints_batch[i]

                written += batch
                pbar.update(batch)

    print(f"Dataset written to {output_path}  ({n} samples)")


# ---------------------------------------------------------------------------
# PyTorch Dataset (for train_labeler.py)
# ---------------------------------------------------------------------------

def make_synth_dataset(h5_path: str, split: str = "train", val_frac: float = 0.02):
    """Return a SynthH5Dataset for the given split.

    Requires h5py and torch.
    """
    try:
        import h5py  # noqa: F401
        import torch.utils.data  # noqa: F401
    except ImportError as exc:
        raise ImportError("h5py and torch are required.") from exc

    return _SynthH5Dataset(h5_path, split=split, val_frac=val_frac)


class _SynthH5Dataset:
    """HDF5-backed dataset that voxelises samples on the fly.

    Opens one h5py file handle per DataLoader worker (fork-safe).
    """

    def __init__(self, h5_path: str, *, split: str = "train", val_frac: float = 0.02):
        self.h5_path = h5_path
        self._file = None   # opened lazily per worker

        # Determine index split from a closed handle
        import h5py
        with h5py.File(h5_path, "r") as f:
            total = f["marker_positions"].shape[0]

        n_val = max(1, int(total * val_frac))
        if split == "train":
            self._indices = np.arange(0, total - n_val)
        elif split == "val":
            self._indices = np.arange(total - n_val, total)
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def __len__(self) -> int:
        return len(self._indices)

    def _get_file(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.h5_path, "r", swmr=False)
        return self._file

    def __getitem__(self, item: int):
        import torch
        from .deep_labeler import voxelise_with_centroid

        idx = int(self._indices[item])
        f = self._get_file()
        n = int(f["marker_counts"][idx])
        markers = f["marker_positions"][idx, :n, :]   # (n, 3)
        joints = f["joint_positions"][idx]             # (21, 3)

        grid, centroid = voxelise_with_centroid(markers)
        heatmaps = build_target_heatmaps(joints, centroid)

        return (
            grid,
            torch.tensor(heatmaps, dtype=torch.float32),
        )
