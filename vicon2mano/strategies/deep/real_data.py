"""Real labeled Vicon CSV loader for training the deep marker labeler.

CSV format (Hands_only1_labeled.csv and equivalents):
    frame, marker, x_mm, y_mm, z_mm

The file contains one row per visible marker per frame.  Marker labels
follow the clean_kinematics naming convention (Palm2, Index1–3, …).
16 of the 22 markers correspond to MANO joints (0–19, skipping the 5
fingertip indices 4, 8, 12, 16, 20).  The remaining markers (Forearm1–4,
Palm1, Palm3) are "unmatched" — they appear in the network input as
distractors but have no heatmap target.

Training pipeline per sample (one frame)
-----------------------------------------
1. Look up the frame's marker positions (metres).
2. Build a (21, 3) joint-position array: observed joints ← marker
   position; unobserved joints ← NaN.
3. Optionally drop a random subset of markers (simulates occlusion).
4. Optionally add ghost markers (simulates Vicon noise).
5. Shuffle marker order (the network must be permutation-invariant).
6. Voxelise → (1, 128, 128, 128) input grid.
7. Build target heatmaps → (21, 128, 128, 128); NaN joints get a
   zero channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Marker → MANO joint index (21-joint space)
# ---------------------------------------------------------------------------

# MANO joint ordering (from correspondence.MANO_JOINT_NAMES):
#   0  wrist          5  index_mcp   9  middle_mcp  13 ring_mcp   17 pinky_mcp
#   1  thumb_mcp      6  index_pip   10 middle_pip  14 ring_pip   18 pinky_pip
#   2  thumb_pip      7  index_dip   11 middle_dip  15 ring_dip   19 pinky_dip
#   3  thumb_dip      8  index_tip   12 middle_tip  16 ring_tip   20 pinky_tip
#   4  thumb_tip   (fingertips 4, 8, 12, 16, 20 are never observed by Vicon)

MARKER_TO_MANO: dict[str, int] = {
    "Palm2":   0,   # wrist
    "Thumb1":  1,   # thumb_mcp
    "Thumb2":  2,   # thumb_pip
    "Thumb3":  3,   # thumb_dip
    "Index1":  5,   # index_mcp
    "Index2":  6,   # index_pip
    "Index3":  7,   # index_dip
    "Middle1": 9,   # middle_mcp
    "Middle2": 10,  # middle_pip
    "Middle3": 11,  # middle_dip
    "Ring1":   13,  # ring_mcp
    "Ring2":   14,  # ring_pip
    "Ring3":   15,  # ring_dip
    "Pinky1":  17,  # pinky_mcp
    "Pinky2":  18,  # pinky_pip
    "Pinky3":  19,  # pinky_dip
}

# Markers present in the CSV that have no MANO joint target
_UNMATCHED = {"Forearm1", "Forearm2", "Forearm3", "Forearm4", "Palm1", "Palm3"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealDataConfig:
    # Minimum number of MANO joints that must be observed in a frame for it
    # to be included in the dataset.
    min_joints: int = 8

    # Unit conversion: CSV stores mm, MANO works in metres.
    marker_scale: float = 1e-3

    # Which hand to extract from wide-format CSVs ("left" or "right").
    side: str = "left"

    # Augmentation — keep mild; real data already has natural occlusions.
    dropout_max: int = 3     # drop up to this many markers per frame
    ghost_max: int = 2       # add up to this many ghost markers
    ghost_range_m: float = 0.05  # ghost positions within ±5 cm of marker bbox

    # Train / validation split (fraction of frames held out as val).
    val_frac: float = 0.05

    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealLabeledDataset:
    """PyTorch-compatible dataset backed by a labeled Vicon CSV.

    Loads the full CSV into memory once, then serves voxelised samples with
    augmentation on every __getitem__ call.  Memory footprint is roughly
    60 k frames × 18 markers × 3 coords × 4 bytes ≈ 13 MB — trivial.

    Args:
        csv_path: Path to the labeled CSV (columns: frame, marker, x_mm,
                  y_mm, z_mm).
        split:    ``"train"`` or ``"val"``.
        cfg:      Augmentation and split parameters.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        cfg: RealDataConfig | None = None,
    ) -> None:
        if cfg is None:
            cfg = RealDataConfig()
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        frames = _load_csv(csv_path, cfg)
        if not frames:
            raise ValueError(f"No usable frames found in {csv_path}")

        n_val = max(1, int(len(frames) * cfg.val_frac))
        if split == "train":
            self._frames = frames[: len(frames) - n_val]
        elif split == "val":
            self._frames = frames[len(frames) - n_val :]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, item: int):
        import torch
        from .labeler import voxelise_with_centroid
        from .synth_data import build_target_heatmaps

        markers_m, joint_pos = self._frames[item]  # (N,3), (21,3) with NaN

        markers_aug = _augment(markers_m, self._rng, self.cfg)

        grid, centroid = voxelise_with_centroid(markers_aug)
        heatmaps = build_target_heatmaps(joint_pos, centroid)

        return (
            grid,                                            # (1, 128, 128, 128) float32
            torch.tensor(heatmaps, dtype=torch.float32),    # (21, 128, 128, 128)
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(
    csv_path: str | Path,
    cfg: RealDataConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse a Vicon CSV and return (markers_m, joint_pos) tuples.

    Supports two formats, auto-detected from column names:
    - Long format: columns ``frame, marker, x_mm, y_mm, z_mm``
    - Wide format: Vicon Nexus export with columns ``_Frame, MarkerName_X, _Y, _Z, …``
                   (one row per frame, all markers as column triplets)
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for real-data training.\n"
            "Install with: pip install 'vicon2mano[train]'"
        ) from exc

    df = pd.read_csv(csv_path)
    if "_Frame" in df.columns:
        return _load_wide_csv(df, cfg)
    return _load_long_csv(df, cfg)


def _load_long_csv(
    df,
    cfg: RealDataConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse long-format CSV (frame, marker, x_mm, y_mm, z_mm)."""
    required = {"frame", "marker", "x_mm", "y_mm", "z_mm"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required}; found {list(df.columns)}"
        )

    scale = cfg.marker_scale
    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for _frame_id, group in df.groupby("frame", sort=True):
        marker_names = group["marker"].tolist()
        positions_m = group[["x_mm", "y_mm", "z_mm"]].to_numpy(np.float32) * scale

        joint_pos = np.full((21, 3), np.nan, dtype=np.float32)
        n_observed = 0
        for name, pos in zip(marker_names, positions_m):
            if name in MARKER_TO_MANO:
                joint_pos[MARKER_TO_MANO[name]] = pos
                n_observed += 1

        if n_observed < cfg.min_joints:
            continue

        frames.append((positions_m, joint_pos))

    return frames


def _load_wide_csv(
    df,
    cfg: RealDataConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse wide-format Vicon Nexus CSV.

    Column layout: ``_Frame, _Sub Frame, MarkerName_X, _Y, _Z, …``
    Marker names contain ``_Left`` or ``_Right``; cfg.side selects which hand.
    The suffix is stripped before looking up MARKER_TO_MANO.
    """
    suffix = f"_{cfg.side.capitalize()}"   # "_Left" or "_Right"
    all_cols = list(df.columns)

    # Find (name, x_idx, y_idx, z_idx) for each marker on the requested side
    triplets: list[tuple[str, int, int, int]] = []
    for i, col in enumerate(all_cols):
        if col.endswith("_X") and suffix in col:
            bare = col.replace("_X", "").replace(suffix, "")  # e.g. "Palm2"
            triplets.append((bare, i, i + 1, i + 2))

    if not triplets:
        raise ValueError(
            f"No '{suffix}' markers found in wide CSV. "
            f"Available columns: {all_cols[:20]}"
        )

    scale = cfg.marker_scale
    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for _, row in df.iterrows():
        positions: list[np.ndarray] = []
        names: list[str] = []
        for bare, xi, yi, zi in triplets:
            x, y, z = row.iloc[xi], row.iloc[yi], row.iloc[zi]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                positions.append([float(x), float(y), float(z)])
                names.append(bare)

        if not positions:
            continue

        positions_m = np.array(positions, dtype=np.float32) * scale

        joint_pos = np.full((21, 3), np.nan, dtype=np.float32)
        n_observed = 0
        for name, pos in zip(names, positions_m):
            if name in MARKER_TO_MANO:
                joint_pos[MARKER_TO_MANO[name]] = pos
                n_observed += 1

        if n_observed < cfg.min_joints:
            continue

        frames.append((positions_m, joint_pos))

    return frames


def _augment(
    markers_m: np.ndarray,
    rng: np.random.Generator,
    cfg: RealDataConfig,
) -> np.ndarray:
    """Apply dropout, ghost injection, and shuffle to a marker cloud.

    Args:
        markers_m: (N, 3) marker positions in metres.
        rng:       Random generator (stateful; call mutates it).
        cfg:       Augmentation parameters.

    Returns:
        (M, 3) augmented marker positions, M ≤ N + cfg.ghost_max.
    """
    result = markers_m.copy()

    # Dropout: randomly remove markers
    n_drop = int(rng.integers(0, cfg.dropout_max + 1))
    if n_drop > 0 and len(result) > n_drop:
        keep = rng.choice(len(result), len(result) - n_drop, replace=False)
        keep.sort()
        result = result[keep]

    # Ghost markers: random positions near the hand bounding box
    n_ghost = int(rng.integers(0, cfg.ghost_max + 1))
    if n_ghost > 0:
        bbox_min = result.min(0)
        bbox_max = result.max(0)
        ghosts = rng.uniform(
            bbox_min - cfg.ghost_range_m,
            bbox_max + cfg.ghost_range_m,
            (n_ghost, 3),
        ).astype(np.float32)
        result = np.concatenate([result, ghosts], axis=0)

    # Shuffle (permutation invariance is enforced via voxelisation)
    result = result[rng.permutation(len(result))]
    return result


# ---------------------------------------------------------------------------
# Factory (mirrors synth_data.make_synth_dataset)
# ---------------------------------------------------------------------------

def make_real_dataset(
    csv_path: str | Path,
    split: str = "train",
    cfg: RealDataConfig | None = None,
) -> RealLabeledDataset:
    """Return a RealLabeledDataset for the given split."""
    return RealLabeledDataset(csv_path, split=split, cfg=cfg)
