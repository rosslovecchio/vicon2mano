# RealSense camera → Vicon calibration and depth export

Code: [`vicon2mano/core/camera_calib.py`](../vicon2mano/core/camera_calib.py) (method),
[`scripts/shared/calibrate_cameras.py`](../scripts/shared/calibrate_cameras.py) (per trial),
[`tests/core/test_camera_calib.py`](../tests/core/test_camera_calib.py).
Part of the final dataset, step 5 in [dataset.md](dataset.md). Clock and video export:
[video_sync.md](video_sync.md).

```bash
.viconvenv/Scripts/python.exe scripts/shared/calibrate_cameras.py results/video/P16/Trial2_HOI
```

It needs the trial already exported and synced (`realsense_player.py`) and its SOMA MANO fit
(`results/soma/fullset`). It takes about 5 min per trial (both cameras, every frame). On first
use it downloads the MediaPipe model to `models/hand_landmarker.task` (gitignored). Extra
dependency: `mediapipe` (1.2, Tasks API only).

## The cameras

| | device | mount | hand distance | intrinsics (P16) |
|---|---|---|---|---|
| cam1 (`rsframes1`) | D435 | **head-mounted** (egocentric; view swings across the room) | ~0.4–0.6 m | fx 603.6, no distortion, depth 1 mm units |
| cam2 (`rsframes2`) | D405 | **tripod, fixed** within a trial | ~0.4–0.6 m | fx 433.5, inverse Brown–Conrady, depth 0.1 mm units |

Intrinsics and the depth → colour extrinsics are in every bag, so depth is aligned to colour
with librealsense (`rs.align`) and only the camera → Vicon pose has to be estimated.

The script decides fixed vs moving from the images (`is_moving`): a camera counts as moving
if more than 5 % of its frames are shifted > 1 px from the temporal median image (phase
correlation). On P16, cam1 scores 0.22 and cam2 0.00. Simpler tests fail:

- **Pixel change:** a head-mounted view of a uniform light floor barely changes.
- **Spread of the per-frame camera centres:** one hand at 0.5 m pins the camera centre only
  to about 10 cm (median scatter 12.5 cm even for the fixed cam2).

## Method

Pose convention: `x_cam = R x_vicon + t` (metres). Outputs store the inverse,
`T_vicon_from_cam`.

1. **Correspondences from MediaPipe Hands.** Its 21 landmarks are in exactly the repo's
   21-joint order (wrist, then thumb → pinky, base → tip), so they pair one-to-one with the
   fitted MANO joints (`animate_mesh.fit_joints21_m`, Vicon frame) at the same Vicon frame.
2. **Per-frame pose** (`pose_from_landmarks`): lift the landmarks to 3D with the aligned
   depth (5×5 median), Kabsch onto the joints, drop landmarks more than 3 cm off, then a
   robust 2-D reprojection solve. If MediaPipe finds two hands, keep the one that fits best.
3. **Fixed camera:** start from the per-frame pose that best explains a sample of frames, then
   `pooled_landmark_pose`: one pose for all frames, with gates shrinking 80 → 12 px.
4. **Marker refinement** (`refine_on_dots`): project the Vicon markers, pair each with the
   nearest detected dot (top-hat filter) inside a gate (fixed: 15 → 6 px pooled; moving:
   12 → 6 px per frame), and re-solve. This is what brings the error down from ~10 px
   (MediaPipe vs MANO joint definitions differ) to ~2–3 px.

Why it is built this way. Tried on P16/Trial2/HOI, with visuals in OneDrive
`soma_transfer/calibration/P16_Trial2_HOI/`:

- **Dots alone (RANSAC + ICP over marker triangles) fail.** The EMG sleeve's electrodes,
  wires and tape give about 100 bright false detections per frame, so a wrong pose finds
  plenty of inliers.
- **A time-paired trajectory fit** (depth foreground centroid vs Vicon marker centroid) gets
  scale and position right to a few cm, with no dots. MediaPipe made it unnecessary.
- **MediaPipe was the user's suggestion** and is what makes the correspondences reliable.

## Outputs (in the trial's `results/video/<P>/<Trial>_<cond>/`)

| file | content |
|---|---|
| `calib.json` | per camera: `camera` fixed/moving, `intrinsics`, `sensor_depth_scale_m`, frame counts, residuals; fixed cameras also `T_vicon_from_cam` (4×4, m) |
| `cam*_poses.csv` | moving cameras only: `frame_number, timestamp_ms, vicon_frame, lm_px, T00…T23` (`T_vicon_from_cam`, top 3 rows) for every colour frame with the hand in view |
| `depth_cam*.mkv` | depth aligned to colour, **mm, uint16, FFV1 lossless**, one frame per `cam*.mp4` frame with the same pts; 0 = no depth |
| `calib_cam*.png` | QA: projected Vicon markers and bones on six frames |

Depth is about 110 MB (cam1) and 260 MB (cam2) per trial. Both cameras are stored in mm,
which is finer than either sensor's noise at hand distance; cam2's native 0.1 mm units made
its files 50 % larger for nothing.

## Results

**P16/Trial2/HOI (2026-11-03):**

| | camera | frames | with hand | landmark px | marker px |
|---|---|---|---|---|---|
| cam1 | moving | 2082 | 1368 | 6.8 | 1.7 median, ~5 markers per frame |
| cam2 | fixed | 1629 | 1197 | 10.0 | 3.3 median, 5.4 p90 (3827 pairs) |

## Limits and open items

- **Moving-camera poses are local.** They project the hand well (that is what fits them)
  but place the camera centre only to about 10 cm. Use them for overlaying the
  hand/markers/mesh, not for registering the rest of the scene. No temporal smoothing yet.
- **cam1 marker residuals are optimistic:** about 5 markers fit 6 pose parameters per frame.
- **Needs a MANO fit.** Trials without one (P1, P2, P5, most of P13/P17) need marker-based
  correspondences, or, for cam2, the pose of another trial in the same session if the
  tripod did not move. Not done yet.
- **Overlays use the raw Vicon labels**, which are wrong in places, so tangled bones there
  are label errors, not calibration errors.
- **Check across a session:** compare cam2's pose between the trials of one session.
