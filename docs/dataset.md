# Final dataset: EMG + RGB-D + MANO mesh + markers

**Decisions (2026-11-03):** one **plain HDF5** file per trial, every stream on the Vicon
clock. **Depth is required**, and so is a **camera-to-Vicon extrinsic calibration**. The
dataset backs one publication, so every number in it must be reproducible from the repo
plus the raw data.

Principles:

- **Keep and mask, don't delete.** Store raw data next to cleaned data, with per-frame and
  per-channel quality masks, plus a recorded reason for every exclusion.
- **Manual decisions live in version-controlled files** (`configs/`), never in one-off CLI
  flags, so a rebuild reproduces them.
- **Every trial × data type gets a status** (`usable` / `partial` / `excluded` / `todo`) in
  the manifest (`scripts/shared/dataset_manifest.py` → `results/dataset/manifest.csv`).

## Current status (manifest, 2026-11-03)

| stream | usable | check | draft | todo | excluded |
|---|---|---|---|---|---|
| video sync | 61 | 6 | – | – | 1 |
| EMG sync | 53 | 5 | – | 8 | 2 |
| EMG cleaning | – | – | 66 | – | 2 |
| markers | – | – | 52 | 16 | – |
| mesh | – | – | 50 | 18 | – |
| depth | – | – | – | 68 | – |
| calibration | – | – | – | 68 | – |

Regenerate it with `scripts/shared/dataset_manifest.py`. Every row carries a `<stream>_why`
column.

## Steps and status

| # | step | status |
|---|---|---|
| 1 | **Clock.** Video and EMG offsets for every trial, manual checks recorded in `configs/sync_overrides.csv` | auto sync done (67 video, 58 EMG); manual checks pending, see below |
| 2 | **Markers.** SOMA relabel + gap fill for every participant, with per-frame flags | 9 participants with a final static done; P1, P2, P5, most of P13/P17 missing |
| 3 | **Mesh.** MANO refit on the final labels, keeping the per-frame residual | 50 trials fitted (on current labels) |
| 4 | **EMG cleaning.** 50 Hz comb notch, channel masks, faulty grids excluded, raw kept | quality report done (`docs/emg.md`); cleaning not started |
| 5 | **RGB-D.** Depth export aligned to colour, intrinsics, **camera → Vicon extrinsics** | colour exported; depth and calibration not started |
| 6 | **Package.** Per-trial HDF5 + manifest + report | not started |

### Step 1 – manual checks still needed

From `docs/video_sync.md` and `docs/emg.md`:

- **EMG, no automatic sync** (set by eye from the player's EMG vs finger-motion strip):
  - P4/Trial2/HOI, P5/Trial1/Hands_only, P6/Trial1/Hands_only, P7/Trial1/Hands_only, P12/Trial1/HOI
  - AUX 1 flat: P1/Trial1/Hands_only, P12/Trial2/Hands_only, P17/Trial2/HOI
  - P1/Trial1/HOI has no `.otb4` at all, so it will be `excluded` for EMG
- **EMG, automatic but check:** P9/Trial1/Hands_only and P12/Trial1/Hands_only (RealSense-trigger
  route), P15/Trial1/Hands_only (IoU 0.93), P5/Trial2/HOI (ambiguity 0.64)
- **Video, cameras disagree:** P16/Trial1/Hands_only, P14/Trial2/Hands_only, P7/Trial2/Hands_only,
  P12/Trial1/Hands_only, P12/Trial2/Hands_only

### Step 5 – camera calibration: feasibility (P16/Trial2/HOI)

- Each bag carries both cameras' intrinsics and the depth → colour extrinsics: cam1 D435 colour
  fx 603.6, depth scale 1 mm; cam2 D405 fx 433.5, 0.1 mm, with Brown–Conrady distortion.
- Depth aligned to colour is 89–93 % valid. The reflective markers show as **white dots in
  colour and do not leave holes in depth**.
- **Plan:** detect the dots in colour, lift them to 3D with the aligned depth, then match
  them to the Vicon markers at the same instant. With correspondences unknown, solve with
  RANSAC + ICP pooled over many frames. That yields the camera pose in Vicon coordinates;
  refine it by minimising the marker reprojection error in colour.
- **cam2 is a D405**, a short-range camera (best at about 7–50 cm), but its scene sits at
  about 2.5 m, so its depth is noisy. Calibrate it mainly by colour reprojection.
- Still to check: whether the cameras stay fixed within a session. If so, calibrate once per
  session and check the residual on every trial.

## HDF5 layout (draft)

```
/meta              attrs: participant, trial, condition, vicon_rate, provenance (git sha, versions)
/sync              per stream: offset_s, method, accuracy_s, checked_by (from sync.json / emg.json / overrides)
/markers           xyz (T,N,3) mm, labels, quality flags (T,N); raw Vicon xyz kept
/mano              global_orient, hand_pose, transl, betas, scale; residual (T,); joints21 (T,21,3)
/emg/grid{1-4}     raw int32 (96,S), cleaned uV (96,S), channel_mask, layout 16x6, t0_vicon_s
/video/cam{1,2}    frame_vicon_t (F,), frame_number, intrinsics, T_vicon_from_cam (4,4), mp4/depth file refs
```

Video and depth stay as files next to the HDF5 (MP4 for colour; depth as lossless 16-bit PNG
or FFV1), referenced by path and per-frame timestamps.
