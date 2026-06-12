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

All 75 tests should pass. The three test files added in the deep-labeler work:
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
1. **Label-seed heuristic** — if ≥15 Vicon labels are recognised. Labels are
   normalised first (lowercased, "left"/"right" and punctuation stripped, so
   `Index_Left1` matches the `index1` hint). Pass `side=` to exclude
   opposite-hand labels in two-hand exports.
2. **DeepLabeler** (`labeler.label_sequence`) — CNN heatmap + linear assignment
3. **Hungarian on mean pose** — always-available fallback (NaN-gap safe)

`_majority_vote` collapses (T, 21) per-frame assignments to (21,) via `np.bincount` mode; out-of-bounds indices and -1 are ignored.

**Warning:** the checked-in `vicon2mano/weights/deep_labeler.pt` is a
smoke-test checkpoint (trained with `max_samples=500` on one real recording);
its predictions are unreliable. The label-seed path is what makes real
recordings work. Retrain with the full synthetic pipeline before trusting the CNN.

## smplx MANO conventions (fitter.py)

smplx returns **16 joints** in MANO-native order (wrist, index, middle, pinky,
ring, thumb); fingertips are mesh vertices (ids 744, 320, 443, 554, 671).
`mano_output_to_joints21()` converts to the repo's 21-joint order — always use
it, never `out.joints[:, :21]`. The fitter optimises per-frame `transl`
(Vicon world coords vs MANO origin) and initialises global orientation with
per-frame Kabsch alignment. `FitResult` and the output `.npz` include `transl`.
Expected fit residual on real data: ~15–20 mm mean (markers sit on the skin,
not at joint centres).

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

## Work completed 2026-06-12 (pipeline repair)

- `loader.py`: `load_csv` auto-detects the single-header wide Nexus layout
  (`Marker_X,_Y,_Z`); blank cells → NaN; clear error for unknown layouts
- `fitter.py`: fixed stage-0 crash (45-dim pose fed to a PCA-6 model); fixed
  `_forward_np` final-minibatch betas expansion; added `mano_output_to_joints21`
  (16-joint reorder + fingertip vertices); added per-frame `transl` parameter
  and Kabsch init (`_init_global_pose`); `FitResult.transl`
- `correspondence.py`: label normalisation (`_normalise_label`), side filtering,
  label-seed now outranks the DeepLabeler, NaN-safe Hungarian/`nanmean`
- `deep_labeler.py`: inference skips non-finite markers, indices remapped
- `synth_data.py`: `generate_dataset` uses `mano_output_to_joints21`
- `export.py`: `.npz` / `to_dict` include `transl`
- Verified end-to-end on `data/Pxh8` (left hand): 16/21 joints matched by
  label, ~17 mm mean joint-to-marker residual
- All tests passing (75/75)

## Work completed 2026-06-12 (articulation fix)

The fit used to track global motion but kept the hand permanently extended
(flat mean pose). Three compounding causes, all in `fitter.py`:

1. **Mis-scaled pose prior (root cause).** The joint loss is squared metres
   (~6e-4 at a good fit) while grasps need PCA coefficients ±3, so with
   `w_pose=1e-3` the flat hand was the *global optimum* of the objective.
   Rescaled `w_pose` 1e-3→1e-5, `w_temp` 5e-2→5e-3. Keep this unit mismatch
   in mind when adding loss terms.
2. **Optimizer step budget.** Adam moves each param ≤ lr per step; one small
   lr (3e-3 × 200 iters) could never reach flexed pose space. Now per-group
   rates (`lr_pose=5e-2`, `lr_orient=1e-2`, transl `lr=3e-3`), cosine decay,
   `n_iters_pose=300`.
3. **Kabsch init on flexed frames.** Aligning the flat template to *all*
   markers tilted the palm toward curled fingers. `_init_global_pose` now
   anchors only to palm-rigid joints (wrist + 4 finger MCPs).

Result on full `data/Pxh8` recording: 18.7 mm mean residual, fitted
articulation correlates 0.978 with the raw marker flexion signal.

`scripts/animate_fit.py` renders a side-by-side raw-vs-fitted animation:
timelapse mode (default) or real-time clip mode (`--speed 1.5 --rate 100`,
auto-picks the most dynamic window; `.mp4` needs imageio-ffmpeg, `.gif` works
everywhere). The capture rate of the Pxh8 CSV is *assumed* 100 Hz (not
recorded anywhere) — `--rate` overrides.

## Work completed 2026-06-12 (both hands)

- Right hand fitted over the full recording with the fixed objective:
  `data/Pxh8/Hands_only_Left/mano_fit_right.npz` (18.0 mm mean residual;
  16/21 joints label-matched, all to `*_Right*` markers).
- `scripts/animate_fit.py` accepts multiple `--npz` files (one per hand) and
  draws every skeleton in the "after" panel; the camera follows the combined
  joints with smoothed per-axis adaptive zoom (`set_box_aspect`), and
  `.html` output writes a self-contained scrubbing player for phones that
  can't render GIF/mp4.

## Work completed 2026-06-12 (jitter / robustness fix)

Finger-angle analysis exposed 712 frames (124 short episodes) where the
fitted left wrist teleported up to 179 mm/frame while markers moved <1.5 mm —
per-frame solutions falling into wrong basins (axis-angle ±π wraparound +
weak temporal coupling after the articulation fix). Three-layer fix in
`fitter.py`, none of which suppresses real gestures:

1. `_init_global_pose` unwraps the Kabsch rotvec sequence (picks the 2π-
   equivalent representative closest to the previous frame).
2. Joint-space **acceleration** penalty `w_accel=2.0` in the pose stage:
   gauge-invariant, wraparound-safe, constant velocity costs zero. Don't
   raise it past ~5: at 20 it visibly damped fast flexions (articulation
   corr 0.98→0.84).
3. `_refine_outliers` rescue stage: frames whose mean joint acceleration
   exceeds `outlier_accel_m` (5 mm/frame²) are re-initialised by slerp/lerp
   from clean flanking frames and locally re-optimised; accepted only if the
   marker residual doesn't degrade.

Validation: spike windows 25200–25800 / 27300–28100 went from 33/46 spikes
to 0 (max step 3 mm); articulation window 18165–21165 keeps corr 0.969 and
full 104–146 mm dip-wrist range; residuals unchanged or better.

`scripts/export_finger_angles.py` exports per-frame MCP/PIP/DIP flexion
angles (degrees, bone-to-bone, 0° = straight) for any number of fitted
hands to CSV.
