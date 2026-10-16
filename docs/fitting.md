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

## Full-dataset speed and frame flags (2026-10-16, RTX 5090)

Measured on P10 Trial2 hands-only with SOMA labels, `configs/surface_fit/P10.json`
settings, GPU shared with another job (65-93% busy). Stage 3 (labelling) takes
13 s for the 42k-frame trial.

**Where the surface fit's time went.** The movement fit is ~5,500 Adam
iterations per chunk and was launch-latency bound (~22 ms/iteration at any batch
size), not compute bound. Two hidden causes: smplx's `batch_rigid_transform`
indexes a Python list with a CUDA tensor (`transform_chain[parents[i]]`, a
device->host sync per kinematic step), and the loss used boolean-mask indexing.
Fixes, all numerically equivalent (eager refactor is bit-identical to the old
code; graph mode differs by ~0.08 deg mean, 1 deg max):

- `core.fitter`: patched `smplx.lbs.batch_rigid_transform` (cached parent list)
  and cached index tensors in `mano_output_to_joints21`.
- `fit_surface_markers.py --cuda-graph`: whole step (forward, backward, Adam)
  captured in a CUDA graph; falls back to eager if capture fails. Masked means
  replace boolean indexing; constants are sliced outside the loss closures
  (a Python-list index or a scalar write into a CUDA tensor is a host->device
  copy and cannot be captured). `torch.compile` is unavailable here (no Triton /
  C++ compiler on Windows).
- `--mv-overlap N`: each chunk also fits N frames either side and keeps only its
  own. Independent chunks used to jump by up to 19 deg at the boundary
  (typical frame step 0.34 deg); with 150 frames of overlap the boundary steps
  are 0.9-2.4 deg.

| 6,000-frame window (frames 6274-12273) | ms/frame | angle err vs old baseline |
|---|---|---|
| old: eager, chunk 1500 | 104 | - |
| `--cuda-graph`, chunk 1500 | 18.0 | 0.21 deg mean, p99 2.1 |
| `--cuda-graph --mv-chunk 1500 --mv-overlap 150` | 19.7 | 0.25 deg mean, p99 2.3 |
| `--cuda-graph`, one 6000 chunk | 15.7 | 0.53 deg mean, p99 3.1 |
| `--cuda-graph`, one 12000 chunk | 17.6 | - |
| 3 processes x (chunk 3000, overlap 150) | 46 each = 15.4 aggregate | - |

Once launch overhead is gone the GPU is the limit: bigger chunks and extra
processes add little (~15%). **Recommended: `--cuda-graph --mv-chunk 3000
--mv-overlap 150`**, one or two processes, ~19 ms/frame, i.e. ~105 h for 20M
frames (the old eager chunk-1500 setting was ~580 h).

Not worth doing (measured against the old baseline on the same window):
fewer iterations (2,000 vs 5,500) cost 5.7 deg mean / 16% of joint-frames
>10 deg; fitting every 2nd/4th/8th frame and interpolating (slerp) cost
1.3 / 2.4 / 4.1 deg mean, and the error is already present *at the fitted
frames*, so it is the sparse fit drifting, not the interpolation. A post-fit
Gaussian/Savitzky-Golay smoother changes angles by <0.13 deg mean: the fit is
already smooth (jitter 0.033 deg/frame^2); the only large steps were the chunk
boundaries.

**Frame flags.** `scripts/soma/flag_frames.py` grades every fitted frame
0 (no signal) / 1 (review) / 2 (unreliable) from three independent signals:
label doubt (low confidence, per-frame label != piece vote), fit residual (a
marker far above its own robust median), and implausible angles. On P10 only
0.114% of points are wrong after the vote and 60% of those wrong labels have
confidence >= 0.9, so confidence alone catches at most ~40% of them; the
thresholds are uncalibrated starting points and need checking against reviewed
frames (`eval_sample.py`) and injected swaps before being trusted on other
participants.
