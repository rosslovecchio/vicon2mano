# Core: MANO fitting pipeline

*Split out of CLAUDE.md; see `../../CLAUDE.md` for the index.*

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

`scripts/shared/animate_fit.py` renders a side-by-side raw-vs-fitted animation:
timelapse mode (default) or real-time clip mode (`--speed 1.5 --rate 100`,
auto-picks the most dynamic window; `.mp4` needs imageio-ffmpeg, `.gif` works
everywhere). The capture rate of the Pxh8 CSV is *assumed* 100 Hz (not
recorded anywhere) — `--rate` overrides.

## Work completed 2026-06-12 (both hands)

- Right hand fitted over the full recording with the fixed objective:
  `data/Pxh8/Hands_only_Left/mano_fit_right.npz` (18.0 mm mean residual;
  16/21 joints label-matched, all to `*_Right*` markers).
- `scripts/shared/animate_fit.py` accepts multiple `--npz` files (one per hand) and
  draws every skeleton in the "after" panel; the camera follows the combined
  joints with smoothed per-axis adaptive zoom (`set_box_aspect`), and
  `.html` output writes a self-contained scrubbing player for phones that
  can't render GIF/mp4.
