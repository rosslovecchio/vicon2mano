# vicon2mano — Claude session context

Fit MANO hand model parameters to Vicon 3D optical marker data.

## Environment

- Python env: `.viconvenv/` — use `.viconvenv/bin/python` and `.viconvenv/bin/pytest`
- Install: `pip install -e ".[dev,train]"` (train extras add h5py for dataset work)
- MANO weights: expected at `../clean_kinematics/mano_v1_2/models/` (sibling repo); override with `--mano-dir`

## Running tests

```bash
.viconvenv/bin/pytest tests/ -v
```

All 60 tests should pass. The three test files added in the deep-labeler work:
- `tests/test_correspondence_get_assignment.py` — `get_assignment` + `_majority_vote`
- `tests/test_deep_labeler.py` — voxelisation, U-Net forward pass, `DeepLabeler`
- `tests/test_synth_data.py` — augmentation, heatmap targets, `_SynthH5Dataset`

## Architecture

```
vicon2mano/
  cli.py              Entry point: parse args, load labeler, call MANOFitter
  fitter.py           MANOFitter: two-stage MANO optimisation
  correspondence.py   Marker↔joint assignment (get_assignment unified API)
  deep_labeler.py     CNN-based labeler (3-D U-Net, VOXEL_RES=64)
  synth_data.py       Synthetic data generation + HDF5 dataset
  real_data.py        Real Vicon data utilities
  loader.py           .c3d / CSV loaders
  export.py           save_npz / save_obj

scripts/
  train_labeler.py    CLI: generate synthetic data + train MarkerLabeler
  eval_labeler.py     CLI: evaluate labeler on held-out data
```

## Key constants (deep_labeler.py)

| Constant | Value | Meaning |
|---|---|---|
| `VOXEL_RES` | 64 | Grid side length (64³ voxels) |
| `SPATIAL_HALF` | 0.22 m | Grid half-extent |
| `GAUSSIAN_SIGMA_VOXELS` | 1.5 | Blob std for heatmap targets |
| `CONF_THRESHOLD` | 0.10 | Min sigmoid confidence to accept a match |

**Important:** tests must import `VOXEL_RES` from `vicon2mano.deep_labeler` rather than hardcoding the grid size.

## Assignment pipeline (correspondence.py — `get_assignment`)

Priority order:
1. **DeepLabeler** (`labeler.label_sequence`) — CNN heatmap + linear assignment
2. **Label-seed heuristic** — if ≥15 Vicon labels are recognised
3. **Hungarian on mean pose** — always-available fallback

`_majority_vote` collapses (T, 21) per-frame assignments to (21,) via `np.bincount` mode; out-of-bounds indices and -1 are ignored.

## CLI

```bash
# Basic fit
vicon2mano recording.c3d output.npz --side right

# With trained CNN weights
vicon2mano recording.c3d output.npz --side right \
    --labeler-weights vicon2mano/weights/deep_labeler.pt

# Export meshes too
vicon2mano recording.c3d output.npz --side right --obj-dir meshes/
```

## Training the deep labeler

```bash
# Step 1: generate synthetic frames
python scripts/train_labeler.py generate \
    --mano-dir ../clean_kinematics/mano_v1_2/models \
    --n-samples 500000 \
    --out data/synth_labeler.h5

# Step 2: train
python scripts/train_labeler.py train \
    --data data/synth_labeler.h5 \
    --out-weights vicon2mano/weights/deep_labeler.pt \
    --epochs 20 --batch-size 2 --lr 3e-4
```

Weights are saved to `vicon2mano/weights/deep_labeler.pt` as `{"model": state_dict}`.

## Work completed in prior sessions

- `deep_labeler.py`: 3-D U-Net (`MarkerLabeler`), `voxelise_with_centroid`, `assign_from_heatmaps`, `DeepLabeler` inference wrapper
- `synth_data.py`: `augment_markers`, `build_target_heatmaps`, `_SynthH5Dataset`, `generate_dataset`
- `correspondence.py`: added `get_assignment` + `_majority_vote` (refactored fitter to use unified API)
- `fitter.py`: `MANOFitter.__init__` now accepts `labeler=` kwarg
- `cli.py`: added `--labeler-weights` flag with graceful fallback
- `scripts/train_labeler.py`, `scripts/eval_labeler.py`: full train/eval pipelines
- All tests passing (60/60)
