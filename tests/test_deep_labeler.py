"""Tests for vicon2mano.deep_labeler.

All tests run on CPU without trained weights, covering:
  - voxelise_with_centroid correctness
  - MarkerLabeler forward-pass tensor shapes
  - assign_from_heatmaps correctness (identity, missing, fewer markers)
  - DeepLabeler integration (random weights, bad path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vicon2mano.deep_labeler import (
    CONF_THRESHOLD,
    K_JOINTS,
    SPATIAL_HALF,
    VOXEL_RES,
    DeepLabeler,
    MarkerLabeler,
    assign_from_heatmaps,
    voxelise_with_centroid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_random_weights(path: Path) -> None:
    """Save an untrained MarkerLabeler to *path* in the expected checkpoint format."""
    model = MarkerLabeler()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def _spike_heatmaps(
    markers: np.ndarray,
    centroid: np.ndarray,
    *,
    res: int = VOXEL_RES,
    half: float = SPATIAL_HALF,
    high_logit: float = 10.0,
    bg_logit: float = -10.0,
) -> torch.Tensor:
    """Build (K_JOINTS, res, res, res) heatmaps with one spike per joint.

    Spike for joint j is placed at the voxel corresponding to markers[j].
    Works for K_JOINTS == len(markers).
    """
    assert len(markers) == K_JOINTS
    pts = markers - centroid
    vf = (pts / half + 1.0) * 0.5 * res  # (21, 3) float voxel coords

    heatmaps = torch.full((K_JOINTS, res, res, res), bg_logit)
    for j in range(K_JOINTS):
        fx, fy, fz = vf[j]
        ix = max(0, min(res - 1, int(round(fx))))
        iy = max(0, min(res - 1, int(round(fy))))
        iz = max(0, min(res - 1, int(round(fz))))
        heatmaps[j, iz, iy, ix] = high_logit
    return heatmaps


# ---------------------------------------------------------------------------
# voxelise_with_centroid
# ---------------------------------------------------------------------------

def test_voxelise_shape_and_dtype():
    rng = np.random.default_rng(0)
    markers = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    grid, centroid = voxelise_with_centroid(markers)

    assert grid.shape == (1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
    assert grid.dtype == torch.float32
    assert float(grid.min()) >= 0.0
    assert float(grid.max()) <= 1.0 + 1e-5
    assert centroid.shape == (3,)
    assert centroid.dtype == np.float32


def test_voxelise_values_in_range():
    """Grid values must be in [0, 1] even with overlapping blobs."""
    rng = np.random.default_rng(1)
    # Pack markers tightly to force blob overlap
    markers = rng.uniform(-0.01, 0.01, (10, 3)).astype(np.float32)
    grid, _ = voxelise_with_centroid(markers)
    assert float(grid.min()) >= 0.0
    assert float(grid.max()) <= 1.0 + 1e-5


def test_voxelise_centroid_invariance():
    """Translating all markers by a constant must not change the grid."""
    rng = np.random.default_rng(2)
    markers = rng.uniform(-0.10, 0.10, (12, 3)).astype(np.float32)
    offset = np.array([0.5, -0.3, 0.2], dtype=np.float32)

    grid1, _ = voxelise_with_centroid(markers)
    grid2, _ = voxelise_with_centroid(markers + offset)
    assert torch.allclose(grid1, grid2, atol=1e-5)


def test_voxelise_single_marker_at_origin_peaks_at_centre():
    """A single marker at the centroid (origin) must peak at voxel (64,64,64)."""
    markers = np.zeros((1, 3), dtype=np.float32)
    grid, _ = voxelise_with_centroid(markers)

    # grid shape: (1, res, res, res); argmax over the spatial dims
    flat_idx = int(grid[0].argmax())
    iz = flat_idx // (VOXEL_RES * VOXEL_RES)
    iy = (flat_idx % (VOXEL_RES * VOXEL_RES)) // VOXEL_RES
    ix = flat_idx % VOXEL_RES
    centre = VOXEL_RES // 2  # 64

    assert iz == centre
    assert iy == centre
    assert ix == centre


def test_voxelise_peak_value_is_one():
    """The peak of a single-marker grid must be exactly 1.0."""
    markers = np.zeros((1, 3), dtype=np.float32)
    grid, _ = voxelise_with_centroid(markers)
    assert abs(float(grid.max()) - 1.0) < 1e-5


def test_voxelise_marker_outside_bounds_ignored():
    """Markers more than SPATIAL_HALF from the cloud centroid fall outside the
    grid and must be silently clipped (no crash, grid stays all-zero there).

    Two markers 2 m apart: after centering each is 1 m from the centroid,
    which is >> SPATIAL_HALF=0.22 m, so both blobs land entirely outside the
    128³ grid → grid should be all zeros.
    """
    markers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    grid, centroid = voxelise_with_centroid(markers)
    assert grid.shape == (1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
    assert float(grid.sum()) == 0.0


def test_voxelise_no_markers_raises():
    """An empty marker array must raise ValueError."""
    with pytest.raises(ValueError, match="at least one marker"):
        voxelise_with_centroid(np.zeros((0, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# MarkerLabeler — forward pass
# ---------------------------------------------------------------------------

def test_marker_labeler_forward_shape_batch1():
    model = MarkerLabeler()
    model.eval()
    x = torch.zeros(1, 1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES)


def test_marker_labeler_forward_shape_batch2():
    model = MarkerLabeler()
    model.eval()
    x = torch.zeros(2, 1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES)


def test_marker_labeler_output_finite():
    """Outputs must be finite (no NaN/Inf) for a random input."""
    model = MarkerLabeler()
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(0)
    x = torch.rand(1, 1, VOXEL_RES, VOXEL_RES, VOXEL_RES, generator=rng)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_marker_labeler_param_count_reasonable():
    """Sanity-check that the architecture isn't accidentally tiny or huge."""
    model = MarkerLabeler()
    n_params = sum(p.numel() for p in model.parameters())
    # Expect roughly 4–20 M parameters
    assert 1_000_000 < n_params < 50_000_000, f"Unexpected param count: {n_params:,}"


# ---------------------------------------------------------------------------
# assign_from_heatmaps
# ---------------------------------------------------------------------------

def test_assign_shape():
    rng = np.random.default_rng(0)
    markers = rng.uniform(-0.15, 0.15, (K_JOINTS, 3)).astype(np.float32)
    centroid = markers.mean(0)
    heatmaps = _spike_heatmaps(markers, centroid)
    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    assert assign.shape == (K_JOINTS,)
    assert assign.dtype == int


def test_assign_from_heatmaps_identity():
    """Perfect spike heatmaps → each joint maps to its own marker index."""
    rng = np.random.default_rng(42)
    markers = rng.uniform(-0.15, 0.15, (K_JOINTS, 3)).astype(np.float32)
    centroid = markers.mean(0)
    heatmaps = _spike_heatmaps(markers, centroid)

    assign = assign_from_heatmaps(heatmaps, markers, centroid)

    assert np.all(assign >= 0), f"Unmatched joints: {np.where(assign < 0)[0]}"
    assert np.all(assign == np.arange(K_JOINTS)), f"Wrong assignment:\n{assign}"


def test_assign_from_heatmaps_low_confidence_all_minus1():
    """Heatmaps with very negative logits (sigmoid ≈ 0) → all -1."""
    rng = np.random.default_rng(0)
    markers = rng.uniform(-0.15, 0.15, (5, 3)).astype(np.float32)
    centroid = markers.mean(0)
    heatmaps = torch.full((K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES), -100.0)

    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    assert np.all(assign == -1)


def test_assign_from_heatmaps_fewer_markers():
    """With N < 21 markers, at most N joints should be assigned."""
    n_markers = 10
    rng = np.random.default_rng(7)
    markers = rng.uniform(-0.15, 0.15, (n_markers, 3)).astype(np.float32)
    centroid = markers.mean(0)
    # Random heatmaps with high confidence
    heatmaps = torch.randn(K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES) + 5.0

    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    assert (assign >= 0).sum() <= n_markers
    assert (assign < n_markers).all() or (assign == -1).any()


def test_assign_from_heatmaps_empty_markers():
    """Zero markers → all joints unmatched (-1)."""
    markers = np.zeros((0, 3), dtype=np.float32)
    centroid = np.zeros(3, dtype=np.float32)
    heatmaps = torch.zeros(K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES)

    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    assert np.all(assign == -1)


def test_assign_marker_indices_in_bounds():
    """All non-negative assigned marker indices must be valid."""
    rng = np.random.default_rng(3)
    n_markers = 15
    markers = rng.uniform(-0.15, 0.15, (n_markers, 3)).astype(np.float32)
    centroid = markers.mean(0)
    heatmaps = torch.randn(K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES) + 3.0

    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    valid = assign[assign >= 0]
    assert np.all(valid < n_markers)


def test_assign_no_duplicate_marker_indices():
    """Each marker can be assigned to at most one joint."""
    rng = np.random.default_rng(5)
    markers = rng.uniform(-0.15, 0.15, (K_JOINTS, 3)).astype(np.float32)
    centroid = markers.mean(0)
    heatmaps = torch.randn(K_JOINTS, VOXEL_RES, VOXEL_RES, VOXEL_RES) + 3.0

    assign = assign_from_heatmaps(heatmaps, markers, centroid)
    valid = assign[assign >= 0]
    assert len(valid) == len(np.unique(valid)), "Duplicate marker index in assignment"


# ---------------------------------------------------------------------------
# DeepLabeler — integration (uses random weights saved to a tmp file)
# ---------------------------------------------------------------------------

def test_deep_labeler_no_weights_raises():
    with pytest.raises(FileNotFoundError):
        DeepLabeler(weights_path="/nonexistent/path/deep_labeler.pt")


def test_deep_labeler_label_frame_shape(tmp_path):
    _save_random_weights(tmp_path / "weights.pt")
    labeler = DeepLabeler(weights_path=tmp_path / "weights.pt", device="cpu")

    rng = np.random.default_rng(0)
    markers = rng.uniform(-0.15, 0.15, (K_JOINTS, 3)).astype(np.float32)
    assign = labeler.label_frame(markers)

    assert assign.shape == (K_JOINTS,)
    assert assign.dtype == int


def test_deep_labeler_label_sequence_shape(tmp_path):
    _save_random_weights(tmp_path / "weights.pt")
    labeler = DeepLabeler(weights_path=tmp_path / "weights.pt", device="cpu")

    rng = np.random.default_rng(1)
    T, N = 5, K_JOINTS
    markers_seq = rng.uniform(-0.15, 0.15, (T, N, 3)).astype(np.float32)
    result = labeler.label_sequence(markers_seq, batch_size=2)

    assert result.shape == (T, K_JOINTS)
    assert result.dtype == int


def test_deep_labeler_label_sequence_values_in_bounds(tmp_path):
    """All non-negative entries must be valid marker indices."""
    _save_random_weights(tmp_path / "weights.pt")
    labeler = DeepLabeler(weights_path=tmp_path / "weights.pt", device="cpu")

    rng = np.random.default_rng(2)
    T, N = 3, 18
    markers_seq = rng.uniform(-0.15, 0.15, (T, N, 3)).astype(np.float32)
    result = labeler.label_sequence(markers_seq)

    valid = result[result >= 0]
    assert np.all(valid < N)


def test_deep_labeler_loads_raw_state_dict(tmp_path):
    """Checkpoint may be a raw state_dict (no 'model' key)."""
    model = MarkerLabeler()
    path = tmp_path / "raw_weights.pt"
    torch.save(model.state_dict(), path)  # raw dict, no wrapper

    labeler = DeepLabeler(weights_path=path, device="cpu")
    rng = np.random.default_rng(0)
    markers = rng.uniform(-0.15, 0.15, (K_JOINTS, 3)).astype(np.float32)
    assign = labeler.label_frame(markers)
    assert assign.shape == (K_JOINTS,)
