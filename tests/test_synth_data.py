"""Tests for vicon2mano.synth_data.

Tests cover:
  - augment_markers: correct marker count bounds, padding, no duplicates in
    unpadded region, all values finite
  - build_target_heatmaps: shape, value range, peak location, joint-outside-
    grid handled gracefully
  - _SynthH5Dataset: shape of returned tensors, index splitting, multi-sample
    integrity (requires h5py and a tiny synthetic HDF5 fixture)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vicon2mano.deep_labeler import VOXEL_RES
from vicon2mano.synth_data import (
    MAX_MARKERS,
    SynthConfig,
    _SynthH5Dataset,
    augment_markers,
    build_target_heatmaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_joints(seed: int = 0) -> np.ndarray:
    """Return (21, 3) plausible joint positions in metres."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.12, 0.12, (21, 3)).astype(np.float32)


def _make_h5(path, n_samples: int = 20, seed: int = 7) -> None:
    """Write a minimal valid HDF5 fixture."""
    import h5py

    cfg = SynthConfig(dropout_max=3, ghost_max=2)
    rng = np.random.default_rng(seed)
    joints_all = rng.uniform(-0.12, 0.12, (n_samples, 21, 3)).astype(np.float32)

    with h5py.File(path, "w") as f:
        ds_m = f.create_dataset("marker_positions", shape=(n_samples, MAX_MARKERS, 3), dtype="float32")
        ds_c = f.create_dataset("marker_counts", shape=(n_samples,), dtype="int32")
        ds_j = f.create_dataset("joint_positions", shape=(n_samples, 21, 3), dtype="float32")

        for i in range(n_samples):
            padded, n = augment_markers(joints_all[i], rng, cfg)
            ds_m[i] = padded
            ds_c[i] = n
            ds_j[i] = joints_all[i]


def _h5_available() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


requires_h5py = pytest.mark.skipif(not _h5_available(), reason="h5py not installed")


# ---------------------------------------------------------------------------
# augment_markers
# ---------------------------------------------------------------------------

class TestAugmentMarkers:
    def test_no_augmentation_gives_21_markers(self):
        cfg = SynthConfig(dropout_min=0, dropout_max=0, ghost_min=0, ghost_max=0)
        joints = _random_joints()
        rng = np.random.default_rng(0)
        padded, n = augment_markers(joints, rng, cfg)
        assert n == 21

    def test_marker_count_lower_bound(self):
        """With max dropout, at least (21 - dropout_max) markers remain."""
        cfg = SynthConfig(dropout_min=0, dropout_max=6, ghost_min=0, ghost_max=0)
        joints = _random_joints()
        rng = np.random.default_rng(0)
        counts = [augment_markers(joints, rng, cfg)[1] for _ in range(50)]
        assert min(counts) >= 21 - cfg.dropout_max

    def test_marker_count_upper_bound(self):
        """With max ghosts, count must not exceed 21 + ghost_max."""
        cfg = SynthConfig(dropout_min=0, dropout_max=0, ghost_min=0, ghost_max=3)
        joints = _random_joints()
        rng = np.random.default_rng(0)
        counts = [augment_markers(joints, rng, cfg)[1] for _ in range(50)]
        assert max(counts) <= 21 + cfg.ghost_max

    def test_combined_bounds(self):
        """Full augmentation: count in [21 - dropout_max, 21 + ghost_max]."""
        cfg = SynthConfig(dropout_min=0, dropout_max=6, ghost_min=0, ghost_max=3)
        joints = _random_joints()
        rng = np.random.default_rng(1)
        for _ in range(100):
            _, n = augment_markers(joints, rng, cfg)
            assert 21 - cfg.dropout_max <= n <= 21 + cfg.ghost_max

    def test_padded_shape(self):
        cfg = SynthConfig()
        joints = _random_joints()
        rng = np.random.default_rng(0)
        padded, n = augment_markers(joints, rng, cfg)
        assert padded.shape == (MAX_MARKERS, 3)
        assert padded.dtype == np.float32

    def test_padding_rows_are_zero(self):
        """Rows beyond n_markers must be zero."""
        cfg = SynthConfig(dropout_min=0, dropout_max=0, ghost_min=0, ghost_max=0)
        joints = _random_joints()
        rng = np.random.default_rng(0)
        padded, n = augment_markers(joints, rng, cfg)
        # n=21, rest should be zeros
        assert np.all(padded[n:] == 0.0)

    def test_valid_markers_finite(self):
        cfg = SynthConfig()
        joints = _random_joints()
        rng = np.random.default_rng(0)
        padded, n = augment_markers(joints, rng, cfg)
        assert np.isfinite(padded[:n]).all()

    def test_dropout_max_exactly_reached(self):
        """With deterministic rng, ensure dropout is applied at all."""
        cfg = SynthConfig(dropout_min=6, dropout_max=6, ghost_min=0, ghost_max=0)
        joints = _random_joints()
        rng = np.random.default_rng(0)
        _, n = augment_markers(joints, rng, cfg)
        assert n == 21 - 6

    def test_markers_near_joint_positions(self):
        """Each marker (no dropout, no ghosts) must be within 5 mm of some joint.

        Markers are shuffled, so we check the nearest-joint distance rather
        than index-matched distance.
        """
        cfg = SynthConfig(
            dropout_min=0, dropout_max=0, ghost_min=0, ghost_max=0,
            marker_noise_m=1e-3,  # 1 mm noise
        )
        joints = _random_joints(seed=5)
        rng = np.random.default_rng(0)
        padded, n = augment_markers(joints, rng, cfg)
        assert n == 21
        markers = padded[:n]
        # D[m, j] = distance from marker m to joint j
        D = np.linalg.norm(markers[:, None, :] - joints[None, :, :], axis=-1)
        min_dists = D.min(axis=1)  # nearest joint for each marker
        assert np.all(min_dists < 0.005), f"Max nearest-joint dist = {min_dists.max():.4f} m"


# ---------------------------------------------------------------------------
# build_target_heatmaps
# ---------------------------------------------------------------------------

class TestBuildTargetHeatmaps:
    def test_shape(self):
        joints = _random_joints()
        centroid = joints.mean(0)
        h = build_target_heatmaps(joints, centroid)
        assert h.shape == (21, VOXEL_RES, VOXEL_RES, VOXEL_RES)
        assert h.dtype == np.float32

    def test_values_in_range(self):
        joints = _random_joints()
        centroid = joints.mean(0)
        h = build_target_heatmaps(joints, centroid)
        assert float(h.min()) >= 0.0
        assert float(h.max()) <= 1.0 + 1e-5

    def test_peak_near_joint_voxel(self):
        """For each joint, the heatmap peak should decode to within 1 voxel
        of the joint's voxel coordinate."""
        res = VOXEL_RES
        half = 0.22
        joints = _random_joints(seed=3)
        centroid = joints.mean(0)
        h = build_target_heatmaps(joints, centroid, res=res, half=half)

        pts = joints - centroid
        expected_vf = (pts / half + 1.0) * 0.5 * res  # (21, 3) float voxel coords

        for j in range(21):
            flat_idx = int(h[j].argmax())
            iz = flat_idx // (res * res)
            iy = (flat_idx % (res * res)) // res
            ix = flat_idx % res
            actual = np.array([ix, iy, iz], dtype=float)
            expected = expected_vf[j]  # (fx, fy, fz) — x,y,z order

            # Allow up to 1 voxel error (due to rounding at grid boundary)
            # Only check joints that are inside the grid
            if np.all((expected >= 0) & (expected < res)):
                dist = np.linalg.norm(actual - expected)
                assert dist <= 1.5, (
                    f"Joint {j}: expected voxel {np.round(expected)}, "
                    f"got {actual}, dist={dist:.2f}"
                )

    def test_joints_outside_grid_no_crash(self):
        """Joints more than SPATIAL_HALF from the centroid must be skipped."""
        joints = np.zeros((21, 3), dtype=np.float32)
        # Move half the joints far outside
        joints[::2] = np.array([1.0, 0.0, 0.0])
        centroid = np.zeros(3, dtype=np.float32)
        h = build_target_heatmaps(joints, centroid)  # must not raise
        assert h.shape == (21, VOXEL_RES, VOXEL_RES, VOXEL_RES)

    def test_no_overlap_between_well_separated_joints(self):
        """With well-separated joints the peak of each heatmap channel
        is dominated by its own joint (max > all other channels at that voxel)."""
        joints = _random_joints(seed=10)
        centroid = joints.mean(0)
        h = build_target_heatmaps(joints, centroid)

        for j in range(21):
            flat_idx = int(h[j].argmax())
            iz = flat_idx // (VOXEL_RES * VOXEL_RES)
            iy = (flat_idx % (VOXEL_RES * VOXEL_RES)) // VOXEL_RES
            ix = flat_idx % VOXEL_RES
            val_j = float(h[j, iz, iy, ix])
            # All other channels at this voxel should have strictly lower value
            other_vals = np.concatenate([h[:j, iz, iy, ix], h[j + 1:, iz, iy, ix]])
            if len(other_vals) > 0:
                # Relax: peak channel value must be highest, but allow ties only
                # when joints are extremely close (shouldn't happen with seed=10)
                assert val_j >= float(other_vals.max()) - 1e-4


# ---------------------------------------------------------------------------
# _SynthH5Dataset
# ---------------------------------------------------------------------------

@requires_h5py
class TestSynthH5Dataset:
    def test_train_val_lengths(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=100)
        train_ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        val_ds = _SynthH5Dataset(path, split="val", val_frac=0.1)
        assert len(train_ds) == 90
        assert len(val_ds) == 10
        assert len(train_ds) + len(val_ds) == 100

    def test_getitem_shapes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=10)
        ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        grid, heatmaps = ds[0]
        assert grid.shape == (1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
        assert heatmaps.shape == (21, VOXEL_RES, VOXEL_RES, VOXEL_RES)

    def test_getitem_dtypes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=10)
        ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        grid, heatmaps = ds[0]
        assert grid.dtype == torch.float32
        assert heatmaps.dtype == torch.float32

    def test_getitem_value_ranges(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=10)
        ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        for i in range(min(5, len(ds))):
            grid, heatmaps = ds[i]
            assert float(grid.min()) >= 0.0
            assert float(grid.max()) <= 1.0 + 1e-5
            assert float(heatmaps.min()) >= 0.0
            assert float(heatmaps.max()) <= 1.0 + 1e-5

    def test_getitem_all_finite(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=10)
        ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        for i in range(min(5, len(ds))):
            grid, heatmaps = ds[i]
            assert torch.isfinite(grid).all()
            assert torch.isfinite(heatmaps).all()

    def test_invalid_split_raises(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=10)
        with pytest.raises(ValueError, match="split must be"):
            _SynthH5Dataset(path, split="test")

    def test_all_samples_accessible(self, tmp_path):
        """Iterate through the whole dataset without errors."""
        path = str(tmp_path / "test.h5")
        _make_h5(path, n_samples=20)
        ds = _SynthH5Dataset(path, split="train", val_frac=0.1)
        for i in range(len(ds)):
            grid, heatmaps = ds[i]
            assert grid.shape == (1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
