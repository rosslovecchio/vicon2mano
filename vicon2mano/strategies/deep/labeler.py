"""CNN-based marker labeler.

Architecture
------------
3-D U-Net with four encoder stages and a symmetric decoder.  The network
takes a 128³ binary-occupancy voxel grid (one channel, Gaussian blobs at
each marker position) and outputs 21 heatmap volumes — one per MANO joint.
Each heatmap's sigmoid-activated peak indicates where that joint's marker is
located in the grid.

Inference pipeline (per frame)
-------------------------------
1. Centre the raw marker cloud and rasterise it to a (1, 128, 128, 128)
   voxel grid via `voxelise_with_centroid`.
2. Forward-pass the network → (21, 128, 128, 128) raw logits.
3. Apply sigmoid, find the argmax voxel per joint, back-project to world
   coordinates, and run one final linear-assignment step to resolve any
   remaining conflicts → (21,) integer index array.

Reference
---------
Han et al., "Online Optical Marker-based Hand Tracking with Deep Labels",
SIGGRAPH 2018.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXEL_RES = 64               # grid side length: 64³ voxels
SPATIAL_HALF = 0.22          # half-extent in metres; grid spans [−0.22, +0.22]
GAUSSIAN_SIGMA_VOXELS = 1.5  # Gaussian blob std in voxel units (~7 mm at 64³)
K_JOINTS = 21                # number of MANO joints
CONF_THRESHOLD = 0.10        # sigmoid confidence below which a match is rejected
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "deep_labeler.pt"


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------

def voxelise_with_centroid(
    markers: np.ndarray,
    *,
    res: int = VOXEL_RES,
    half: float = SPATIAL_HALF,
    sigma: float = GAUSSIAN_SIGMA_VOXELS,
) -> tuple[torch.Tensor, np.ndarray]:
    """Rasterise 3-D marker positions into a volumetric occupancy grid.

    The marker cloud is centred on its own centroid before voxelisation so
    that the grid is translation-invariant.  Each marker contributes a 3-D
    Gaussian blob; overlapping blobs are combined with element-wise max so
    the grid never exceeds 1.0.

    Args:
        markers: (N, 3) marker positions in metres.
        res:     Grid side length (default 128).
        half:    Half-extent of the grid in metres.  A marker at
                 position ±half (relative to centroid) lands at the grid
                 boundary.
        sigma:   Gaussian blob std in voxel units.

    Returns:
        grid:     (1, res, res, res) float32 tensor, values in [0, 1].
        centroid: (3,) float32 array — mean marker position in metres.
    """
    markers = np.asarray(markers, dtype=np.float32)
    if markers.shape[0] == 0:
        raise ValueError("voxelise_with_centroid requires at least one marker.")
    centroid = markers.mean(axis=0)          # (3,)
    pts = markers - centroid                  # (N, 3) centred

    # Map world coords to voxel float coords.
    # w ∈ [−half, +half]  →  vf ∈ [0, res]  (0 = left boundary, res = right)
    vf = (pts / half + 1.0) * 0.5 * res      # (N, 3)

    grid = np.zeros((res, res, res), dtype=np.float32)
    r = int(np.ceil(3.0 * sigma))             # support radius (voxels)

    for i in range(len(vf)):
        fx, fy, fz = float(vf[i, 0]), float(vf[i, 1]), float(vf[i, 2])
        ix, iy, iz = int(round(fx)), int(round(fy)), int(round(fz))

        # Bounding box of this blob (clip to grid)
        x0, x1 = max(0, ix - r), min(res, ix + r + 1)
        y0, y1 = max(0, iy - r), min(res, iy + r + 1)
        z0, z1 = max(0, iz - r), min(res, iz + r + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue  # marker entirely outside grid

        # Separable Gaussian (z, y, x order — matches grid[z, y, x])
        gx = np.exp(-0.5 * ((np.arange(x0, x1) - fx) / sigma) ** 2)
        gy = np.exp(-0.5 * ((np.arange(y0, y1) - fy) / sigma) ** 2)
        gz = np.exp(-0.5 * ((np.arange(z0, z1) - fz) / sigma) ** 2)
        blob = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

        grid[z0:z1, y0:y1, x0:x1] = np.maximum(grid[z0:z1, y0:y1, x0:x1], blob)

    return torch.tensor(grid).unsqueeze(0), centroid


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv3d + BN + ReLU layers that preserve spatial dimensions."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class MarkerLabeler(nn.Module):
    """3-D U-Net for volumetric marker-to-joint heatmap regression.

    Input:  (B, 1,  128, 128, 128)
    Output: (B, 21, 128, 128, 128)  raw logits

    Encoder channel progression: 1 → 16 → 32 → 64 → 128
    Bottleneck: 128 → 256
    Decoder mirrors encoder with U-Net skip connections.

    Spatial progression (per axis): 128 → 64 → 32 → 16 → 8 (bottleneck)
                                   → 16 → 32 → 64 → 128
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _conv_block(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = _conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = _conv_block(64, 128)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = _conv_block(128, 256)

        # Decoder (skip channels come from the matching encoder stage)
        self.dec4 = _conv_block(256 + 128, 128)  # up from 8  → 16, skip=enc4(128)
        self.dec3 = _conv_block(128 + 64, 64)    # up from 16 → 32, skip=enc3(64)
        self.dec2 = _conv_block(64 + 32, 32)     # up from 32 → 64, skip=enc2(32)
        self.dec1 = _conv_block(32 + 16, 16)     # up from 64 → 128, skip=enc1(16)

        # Output head: 1×1×1 conv to K_JOINTS heatmaps
        self.head = nn.Conv3d(16, K_JOINTS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)                   # (B, 16,  128, 128, 128)
        s2 = self.enc2(self.pool1(s1))      # (B, 32,   64,  64,  64)
        s3 = self.enc3(self.pool2(s2))      # (B, 64,   32,  32,  32)
        s4 = self.enc4(self.pool3(s3))      # (B, 128,  16,  16,  16)
        bn = self.bottleneck(self.pool4(s4))  # (B, 256,   8,   8,   8)

        # Decoder — trilinear upsampling then concatenate skip feature map
        d4 = self.dec4(_cat_up(bn, s4))     # (B, 128,  16,  16,  16)
        d3 = self.dec3(_cat_up(d4, s3))     # (B, 64,   32,  32,  32)
        d2 = self.dec2(_cat_up(d3, s2))     # (B, 32,   64,  64,  64)
        d1 = self.dec1(_cat_up(d2, s1))     # (B, 16,  128, 128, 128)

        return self.head(d1)                # (B, 21,  128, 128, 128)


def _cat_up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Upsample x by 2× and concatenate with skip along the channel dim."""
    x_up = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
    return torch.cat([x_up, skip], dim=1)


# ---------------------------------------------------------------------------
# Assignment decoding
# ---------------------------------------------------------------------------

def assign_from_heatmaps(
    heatmaps: torch.Tensor,
    markers: np.ndarray,
    centroid: np.ndarray,
    *,
    res: int = VOXEL_RES,
    half: float = SPATIAL_HALF,
    conf_threshold: float = CONF_THRESHOLD,
) -> np.ndarray:
    """Decode 21 heatmap volumes into a marker index per joint.

    Algorithm:
    1. Sigmoid-activate the raw logits to get per-voxel confidence.
    2. For each joint, find the argmax voxel and back-project to world coords.
    3. Build a distance matrix between predicted joint positions and actual
       marker positions.
    4. Solve the linear assignment (Hungarian) to resolve conflicts.
    5. Reject assignments whose confidence is below `conf_threshold`.

    Args:
        heatmaps:       (K_JOINTS, res, res, res) raw logits (CPU tensor).
        markers:        (N, 3) marker positions in metres (the same frame
                        that produced the voxel grid).
        centroid:       (3,) centroid used during voxelisation.
        res:            Grid side length.
        half:           Half-extent of the grid in metres.
        conf_threshold: Minimum sigmoid confidence to accept a match.

    Returns:
        assign: (K_JOINTS,) int array; assign[j] = marker index or -1.
    """
    n_markers = markers.shape[0]
    assign = np.full(K_JOINTS, -1, dtype=int)

    if n_markers == 0:
        return assign

    # Sigmoid confidence and argmax per joint
    probs = torch.sigmoid(heatmaps)          # (21, res, res, res)
    flat = probs.reshape(K_JOINTS, -1)       # (21, res³)
    argmax_flat = flat.argmax(dim=1)         # (21,)
    conf = flat[torch.arange(K_JOINTS), argmax_flat].numpy()  # (21,)

    # Unravel flat index → (iz, iy, ix) in ZYX grid order
    iz = (argmax_flat // (res * res)).numpy().astype(float)
    rem = argmax_flat % (res * res)
    iy = (rem // res).numpy().astype(float)
    ix = (rem % res).numpy().astype(float)

    # Back-project voxel indices to world coordinates
    # Inverse of: vf = (pt / half + 1.0) * 0.5 * res
    world = centroid + (np.stack([ix, iy, iz], axis=1) / res * 2.0 - 1.0) * half  # (21, 3)

    # Distance matrix: D[j, m] = ||world[j] − markers[m]||₂
    diff = world[:, None, :] - markers[None, :, :]  # (21, N, 3)
    D = np.linalg.norm(diff, axis=-1)               # (21, N)

    row_ind, col_ind = linear_sum_assignment(D)

    for j, m in zip(row_ind, col_ind):
        if conf[j] >= conf_threshold:
            assign[j] = m

    return assign


# ---------------------------------------------------------------------------
# High-level inference wrapper
# ---------------------------------------------------------------------------

class DeepLabeler:
    """Load trained weights and label Vicon marker frames.

    Usage::

        labeler = DeepLabeler()               # loads default weights
        assign = labeler.label_frame(markers) # (N, 3) metres → (21,)
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Deep labeler weights not found at {weights_path}.\n"
                "Train the model first with:\n"
                "  python scripts/train_labeler.py train "
                "--data data/synth_labeler.h5 "
                "--mano-dir <path/to/mano>"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = MarkerLabeler().to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def label_frame(self, markers: np.ndarray) -> np.ndarray:
        """Label a single frame.

        Markers with non-finite coordinates (Vicon gaps) are ignored;
        returned indices refer to the original markers array.

        Args:
            markers: (N, 3) marker positions in metres.

        Returns:
            assign: (21,) int; assign[j] = marker index or -1.
        """
        clean, orig_idx = _finite_markers(markers)
        if len(clean) == 0:
            return np.full(K_JOINTS, -1, dtype=int)
        grid, centroid = voxelise_with_centroid(clean)
        with torch.no_grad():
            logits = self.model(grid.unsqueeze(0).to(self.device))  # (1, 21, res, res, res)
        assign = assign_from_heatmaps(logits[0].cpu(), clean, centroid)
        return _remap_assignment(assign, orig_idx)

    def label_sequence(
        self,
        markers_seq: np.ndarray,
        *,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Label every frame in a sequence.

        Args:
            markers_seq: (T, N, 3) marker positions in metres.
            batch_size:  Number of frames to forward-pass simultaneously.
                         Keep small (4–8) for 128³ grids to stay within VRAM.

        Returns:
            (T, 21) int assignment array; -1 where a joint is unmatched.
        """
        T, _N, _ = markers_seq.shape
        result = np.full((T, K_JOINTS), -1, dtype=int)

        grids: list[torch.Tensor] = []
        centroids: list[np.ndarray] = []
        cleans: list[np.ndarray] = []
        orig_idxs: list[np.ndarray] = []
        frame_ids: list[int] = []
        for t in range(T):
            clean, orig_idx = _finite_markers(markers_seq[t])
            if len(clean) == 0:
                continue  # fully occluded frame stays all -1
            g, c = voxelise_with_centroid(clean)
            grids.append(g)
            centroids.append(c)
            cleans.append(clean)
            orig_idxs.append(orig_idx)
            frame_ids.append(t)

        with torch.no_grad():
            for start in range(0, len(grids), batch_size):
                end = min(start + batch_size, len(grids))
                batch = torch.stack(grids[start:end]).to(self.device)  # (B, 1, res, res, res)
                logits = self.model(batch)                              # (B, 21, res, res, res)
                for i, k in enumerate(range(start, end)):
                    assign = assign_from_heatmaps(
                        logits[i].cpu(), cleans[k], centroids[k]
                    )
                    result[frame_ids[k]] = _remap_assignment(assign, orig_idxs[k])

        return result


def _finite_markers(markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with non-finite coordinates.

    Returns (clean_markers, original_indices)."""
    markers = np.asarray(markers, dtype=np.float32)
    finite = np.isfinite(markers).all(axis=1)
    orig_idx = np.flatnonzero(finite)
    return markers[finite], orig_idx


def _remap_assignment(assign: np.ndarray, orig_idx: np.ndarray) -> np.ndarray:
    """Translate indices into a filtered marker array back to original indices."""
    out = np.full_like(assign, -1)
    valid = assign >= 0
    out[valid] = orig_idx[assign[valid]]
    return out
