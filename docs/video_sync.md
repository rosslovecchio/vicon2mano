# RealSense video: export, player, and Vicon sync

Code: [`vicon2mano/core/video_sync.py`](../vicon2mano/core/video_sync.py) (clock model,
pulse detection, motion cross-correlation), [`scripts/shared/realsense_player.py`](../scripts/shared/realsense_player.py)
(export + sync + player), [`scripts/shared/realsense_player.html.in`](../scripts/shared/realsense_player.html.in)
(player template; `.html.in` because the repo ignores `*.html`).
Tests: [`tests/core/test_video_sync.py`](../tests/core/test_video_sync.py).

## Usage

```bash
.viconvenv/Scripts/python.exe scripts/shared/realsense_player.py D:/june25_data/P15/Trial1/Hands_only/RealSense
#   --vicon <csv>   Vicon trial CSV, if the manual_frames.csv lookup fails
#   --offset <s>    write a hand-tuned RealSense->Vicon offset (method "manual")
#   --force         re-encode MP4s (otherwise reused; a re-sync takes seconds)
```

Output, in `results/video/<P>/<Trial>_<cond>/` (about 35 MB per trial):

| file | content |
|---|---|
| `player.html` | open in a browser; references the MP4s next to it |
| `cam1.mp4`, `cam2.mp4` | colour stream, H.264, **variable frame rate**: each frame's presentation time is its real timestamp |
| `cam*_frames.csv` | `frame_number` (sensor counter), `timestamp_ms` (epoch, RealSense PC clock), `pts_ms`, `motion`, `vicon_t_s`, `vicon_frame` (the Vicon CSV's 1-based `Frame`) |
| `sync.json` | `vicon_start_ms`, `vicon_rate`, `offset_s`, `method`, `pulse_onset_s`, `trigger_ms`, per-camera `xcorr` lag and r |

Extra dependencies, not in pyproject: `pyrealsense2`, `av` (PyAV, which bundles libx264).

## The data

- `D:\june25_data\<P>\Trial<n>\<cond>\RealSense\` holds `rsframes1_*.bag`, `rsframes2_*.bag`
  and `triggerlog_*.csv`. 68 folders, including oddities: `P10/Trial1/NO_VICON_Handsonly`
  and `P17/Trial2/Hands_only_goodEMG`.
- Vicon trials are in `D:\june25_data\<P>\<P>\*.csv`, with `.xcp`. **The OneDrive copy
  (`core.dataset.DATA_ROOT`) has no `.xcp`, so syncing must read the D: copy.**
- The bags are ROS1 bags from the RealSense SDK. Each holds colour (rgb8) and depth (z16)
  at 848×480. Two devices: rsframes1 = **D435**, rsframes2 = **D405**. Only colour is exported.
- Reading them: `rosbags` lists topics but cannot decode the `realsense_msgs` types (and
  fails on `sensor_msgs/Image` too), so use `pyrealsense2`. In non-real-time playback it
  **repeats framesets**, so frames are de-duplicated by frame number. Doing that
  reproduces the bags' message counts exactly.

### Frame rate: 60 fps configured, about 7–20 fps recorded, irregularly

P15 Trial1 Hands_only, colour stream:

| | device | configured | frames kept | recorded fps | sensor fps (frame counter) | median / max gap |
|---|---|---|---|---|---|---|
| cam1 | D435 | 60 | 4,599 in 235.4 s | 19.5 | 59.5 | 50 / 319 ms |
| cam2 | D405 | 60 | 1,739 in 236.4 s | 7.4 | 59.9 | 84 / 1,753 ms |

The sensors ran at 60 fps, and the recorder dropped most frames. P10 Trial2 shows the same
pattern (21.2 / 7.9 fps). **Never assume a constant fps; sync on per-frame timestamps.**
The timestamps are in the `global_time` domain: the RealSense PC's clock in epoch ms (UTC).
The triggerlog's `frame_timestamp` column uses the same clock.

## Clock model

```
vicon_t [s] = (rs_timestamp_ms - vicon_start_ms) / 1000 + offset
vicon_frame = floor(vicon_t * vicon_rate) + 1
```

- `vicon_start_ms`: the last `<Capture ... START_TIME>` in the trial's `.xcp`, minus 1 h.
  Vicon writes local time labelled "DST" that is **UTC+1**. This was inferred, not
  documented, and confirmed by every pulse trial. The capture span equals the CSV
  duration (P15 Trial1: 244.011 s vs 244.0 s).
- `offset`: the difference between the two PCs' clocks. **It is not constant:**

| date | participant | offset (pulse) |
|---|---|---|
| 06-18 11:25–11:48 | P9 | 0.944 → 0.847 → 0.797 (drifting) |
| 06-18 13:56–17:16 | P10, P11 | 0.560–0.585 (stable for 3 h) |
| 06-23 | P13 | 0.380–0.392 |
| 06-24 11:35 | P14 | 0.452 |
| 06-24 16:56 | P15 (xcorr) | 0.00–0.04: a jump of about 0.45 s within the same day |
| 06-25 | P16, P17 | 0.381–0.392 |

So an offset can't be borrowed from another trial, even one recorded the same day.
Each trial is measured on its own.

### Method 1: trigger pulse (preferred)

When it starts recording, the RealSense recorder sends a **1 s TTL pulse**. Its send
time is the first `trigger_sent=true` row of `triggerlog_*.csv`; its `utc_timestamp` uses
the RealSense PC's clock. Vicon records the pulse on the `Sync` analog channel
(Devices section, 1000 Hz) as a single 1.000 s plateau at +1.25 V.
`offset = pulse_onset_s - (trigger_ms - vicon_start_ms) / 1000`.

The pulse was caught in only **18 of about 60 movement trials**: P9 (3), P10 (4), P11 (4), P13 (4),
P14 Trial2 HOI, P16 trial2_hoi, P17 trial2_only2. None for P1–P8, P12 or P15. Without
it, the channel shows **clipped 50 Hz mains bursts** (exactly 20 ms period) and the odd short
spike, such as P15 Trial1's 223 ms spike at 12.5 s. Those are not the trigger, and
`find_trigger_pulse` requires a single plateau of 1 s ± 5 %.

Triggerlog format quirk: the header has 5 columns but the rows have 4 (`utc, frame_cam,
trigger_sent, frame_timestamp`, with `frame_cam` like `243_1`), so pandas needs
`index_col=False`.

### Method 2: motion cross-correlation (fallback)

- Vicon side: the median speed of the hand markers (all except `Forearm*`), from the
  trajectories (200 Hz).
- Video side: per recorded frame, the mean absolute grey-level change since the
  previous recorded frame, per second. It is computed on **8×8 block means of the raw
  frame**. Plain subsampling keeps sensor noise and roughly halved r (P10 cam2: 0.25 vs 0.56).
- The lag maximises the Pearson r of log1p(speed) against log1p(motion). It is searched
  over ±3 s around zero, in 5 ms steps, separately per camera. `sync.json` uses the camera
  with the higher r, plus `XCORR_BIAS_S`.

Validated against all 18 pulse trials (36 camera lags; details under
[Cohort run](#cohort-run-2026-10-29)):

- 35 of 36 lags are within −49 to +13 ms of the pulse: median −16 ms, sd 14 ms,
  and 89 % within ±30 ms. The method reads systematically early, so
  `XCORR_BIAS_S = 0.016` is added to the chosen lag (the raw lags stay in `sync.json`).
- One gross miss: P13 Trial1 Hands_only cam1 locked onto a peak 3 s off. Picking the
  camera with the higher r avoided it there, and in both P12 trials where the cameras
  disagreed by 2–3 s.
- With the bias applied, the chosen camera is within about ±30 ms of the pulse. That is
  under one video frame (cam1 ~50 ms) and about 6 Vicon frames.
- **Warning sign:** the two cameras disagreeing by more than 50 ms (`cam_spread_ms` in
  the summary). Check those trials in the player.

## Player

- Both cameras run on one shared clock: play/pause (Space), step one frame of a chosen
  camera with ←/→ (Shift ×10), speed 0.1×–4×, scrub bar, and a per-camera offset in ms.
- Each camera's label shows the sensor frame number, the recorded index, the timestamp,
  and how many frames were dropped before this one. A camera with no frame at the
  current time is dimmed.
- With sync: the current Vicon time and frame, "go to Vicon frame", and an editable
  RealSense→Vicon offset (with reset to the automatic value).
- A strip chart covers ±3 s around the cursor: Vicon hand speed (white) against each
  camera's motion, in the camera's colour.
- `player.html#vf=4000` opens the player at Vicon frame 4000, which is handy for
  sharing a moment.
- **Markers pane** (every synced trial): the raw Vicon markers (`markers_pane`, int16 mm
  at 100 Hz embedded in the page, about 3–4 MB) drawn live as a stick figure in finger
  colours (`core.viz.MARKER_BONES`). Drag to rotate, wheel to zoom. The view is centred
  on the hand averaged over ±0.5 s, so a reach still reads as movement. Raw markers are
  the timing ground truth: they don't depend on SOMA or the fit.
- **Manual refinement:** shift the offset until the events line up, then record it in
  `configs/sync_overrides.csv` and rebuild (see below). `--offset` / `--emg-offset` are
  for quick tries only; they are not kept.

### Checking a sync by eye

The strip chart is not independent: it is the signal the xcorr optimised. Compare the
video against the Vicon panes instead.

1. **Pick sharp events:** touch or lift-off on the table or mug, the end of a fast reach,
   a tap. Smooth movement hides 100 ms errors.
2. **Video side:** "step on" the faster camera (usually cam1). Note the last frame before
   and the first frame after the event; the header shows each one's Vicon frame.
3. **Vicon side:** "step on" `mesh` (5–20 ms steps), or watch the markers pane, and find
   the Vicon frame of the same event.
4. **Read the error:** if the Vicon event falls between the camera's before and after
   frames, the sync is right to one camera frame, which is the video's resolution limit.
   If it is consistently outside, adjust the offset box and record the value in
   `configs/sync_overrides.csv`.
5. **Repeat at the start, middle and end.** A constant error is an offset, fixable with
   `--offset`. A growing error is clock drift, which one offset cannot fix: report it.

Calibrate your eye on a pulse trial first (P10 Trial2, P16 Trial2 HOI). Then check
the camera-disagreement trials listed under the cohort run.

### Recording manual decisions: `configs/sync_overrides.csv`

```
trial,stream,offset_s,checked_by,date,note
P5/Trial1_Hands_only,emg,6.25,RL,2026-11-03,set by eye from the EMG strip
P16/Trial2_HOI,emg,,RL,2026-11-03,auto value confirmed
```

`trial` is the `results/video` folder (`<P>/<Trial>_<cond>`) and `stream` is `video` or
`emg`.

- **With `offset_s`:** the row replaces the automatic offset; method becomes `manual`.
- **Without it:** the row records that the automatic value was checked.

Both kinds land in `sync.json` / `emg.json` as `checked_by`, `date` and `note`, and the
manifest counts them as usable. Rebuild after editing:
`realsense_player.py <RealSense dir>` for video, `emg_pane.py <video dir>` for EMG.
`rebuild()` reads the file and never keeps an offset from an earlier CLI run.

### Mesh pane (SOMA/MANO fit beside the cameras)

```bash
.viconvenv/Scripts/python.exe scripts/soma_all/mesh_video.py results/video/P16/Trial2_HOI --n 10000   # 10k frames at 200 Hz
.viconvenv/Scripts/python.exe scripts/soma_all/mesh_video.py results/video/P10/Trial1_HOI --step 4    # whole trial at 50 Hz
```

**All fitted trials** (2026-10-29) were rendered as the whole trial at `--step 4` (50 Hz:
still 2–6× the cameras, and about 10k frames per trial). P16 Trial2 HOI is instead
rendered at every frame for the whole trial (200 Hz, 42,292 frames, 5 min, 77 MB). The fit folder is matched by normalised name, since fits
spell `Trial 1` as `Trial_1`. Trials without a SOMA fit (P1, P2, P5, P13 except
Trial1 Hands_only, P17 except Trial1 Hands_only) get no mesh pane but do get the markers pane.

[`scripts/soma_all/mesh_video.py`](../scripts/soma_all/mesh_video.py):

- **What it renders:** consecutive Vicon frames of the fitted mesh plus the
  SOMA-labelled markers, using the same rendering as `scripts/mano/animate_mesh.py`
  (`render_mesh_frames`, `fit_vertices_mm`, `view_box` were extracted from it for this).
  It renders in parallel and encodes `mesh.mp4` on the Vicon clock, with
  `pts = (frame - start) / rate`.
- **Its frame table:** `mesh_frames.csv` (`vicon_frame, pts_ms`).
- **Then:** it rebuilds the player. `realsense_player.py` adds any `mesh*_frames.csv` as
  an extra pane, placed on the RealSense clock through `sync.json`, so a later
  `--offset` moves the mesh too.

Notes:

- **The view follows the hand:** axes are centred on each frame's mesh, at a fixed size.
  The fixed whole-window box `animate_mesh.py` uses spans about ±3 m over 10k frames
  (arm travel plus stray SOMA markers), which shrinks the hand to a dot.
  `render_mesh_frames` accepts a per-frame `center` for this.
- **Render time:** about 1.5 min per 10k frames with 20 workers on this 24-core machine, when nothing else is running (the first P16 run took 20 min next to other jobs).
  The resulting `mesh.mp4` is about 17 MB (P16 Trial2 HOI).
- The default window starts at the first Vicon frame where every camera is running.
- The fit is read from `results/soma/fullset/<P>/<video dir name>` (`--fit-dir` to override).
  Fit row `k` is Vicon frame `k + 1`.
- Why not the existing animation format: 10k frames as base64 JPEGs inside the HTML
  would be about 500 MB, while the MP4 is about 17 MB.

### Mesh quality and the learned-fill fit (2026-11-03, P16 so far)

`mesh_video.py` now prefers the learned-gap-fill fit (`results/soma/real_fill_w01/movement`,
every 10th frame = 20 Hz, filled values weighted 0.1) and falls back to `results/soma/fullset`.
Each rendered frame gets a grade in `mesh_frames.csv` (`grade`, `note`):

- **Red, not usable:** the hand turned more than 45° in one fitted step, or half or more of
  the markers are missing or on a flagged finger. For fullset fits: `flag_frames.py` flag 2.
- **Yellow, partly usable:** a finger past the natural-pose limits (`frame_diagnostics.py`,
  2° tolerance) or a too-fast joint, or a marker missing. For fullset fits: flag 1.

The player draws a yellow/red border with the reason on the mesh pane, and a clickable
quality strip for the whole trial under the scrub bar. In the render, markers are dark when
observed, blue when gap-filled, orange when on a flagged finger.

P16 (learned fill), share of fitted frames ok / yellow / red: Trial1 Hands only 72/22/5.5%,
Trial1 HOI 56/43/1.1%, Trial2 Hands only 73/25/2.1%, Trial2 HOI 60/40/0.1%. Red is mostly
the hand out of view. Yellow in HOI is mostly thumb1 abduction a few degrees past its limit.

**Marker coverage after the learned fill (P16, 18 hand markers):** 78–86% observed,
1–2% spline, 12–21% learned. 95–99.9% of fitted frames have all 18 markers. What stays
missing is whole frames with the hand out of view (every marker gone at once): 52–2350
frames per trial, most in Trial1 Hands only.

**Labellings player** (`scripts/soma_all/labels_player.py <video dir> [--mesh]`). Built on 2026-11-03 for the
51 trials that have a SOMA fit and/or a learned-fill fit (33 have both, 17 only the SOMA fit, P17
Trial1 HOI only the learned fill); the 16 with neither (P1, P2, P5, and all of P13 and P17 except
the trials above) get no page, since the raw markers are already in the standard player. A pane
whose fit is missing is simply left out. Both pages sit beside the trial's player and need its
`cam1.mp4` and `sync.json`:

- `labels_player.html` (about 10–15 MB): cam1 and the marker panes in a row (raw Vicon labels, SOMA
  labels without fill, learned fill), with the sync strip and offset controls.
- `labels_player_mesh.html` (`--mesh`, 30–100 MB, 2.9 GB for all): a 2×2 grid, with no sync strip
  or offset controls. With both fits it is:

  | | |
  |---|---|
  | cam1 | Vicon labels (raw markers) |
  | SOMA labels, no fill, + fitted mesh | learned gap fill + fitted mesh |

Common to both: the marker panes share one view (drag to rotate, wheel to zoom). The SOMA pane is
graded by `flag_frames.py`, the learned-fill pane by the fit's `flags.json` (nearest fitted frame).
Filled markers are hollow, markers on a flagged finger get an orange ring, and a yellow/red border
means partly/not usable. Markers are drawn at 100 Hz. Nothing is re-fitted or re-encoded.

In the mesh page the meshes are drawn live from the fits' vertices (int16 mm, painter's algorithm),
so they rotate with the markers. Mesh frames are every 10th Vicon frame (20 Hz), the learned-fill
fit's frames; the SOMA mesh takes the same frames from its every-frame fit. The player template
takes `hide_sync`, `cols` and `faces` in its metadata for this layout, and a marker set may carry a
`mesh` (`frames` as Vicon rows, int16 `verts`).

### EMG pane (HD-sEMG heatmaps)

```bash
.viconvenv/Scripts/python.exe scripts/emg/emg_pane.py results/video/P16/Trial2_HOI
```

The EMG goes on the Vicon clock by matching the sync pattern that both Novecento AUX 1 and
Vicon `Sync` record (`sync_mask` + `mask_offset`; 58 of 67 trials). The player then shows a
grid-RMS vs finger-motion strip, per-grid 16×6 electrode maps, and channel × time heatmaps. Details, the grid layout and the
remaining sync cases are in [emg.md](emg.md).

`realsense_player.py` embeds `emg.json` when it is present. `mesh_video.py` and
`emg_pane.py` both finish with `realsense_player.rebuild(video_dir)`, which reuses the
MP4s and the Vicon CSV and keeps a manual offset.

## Cohort run (2026-10-29)

All 68 RealSense folders were exported and synced (MP4s total about 2.4 GB). The
per-trial table is `results/video/sync_summary.csv`, with columns `trial, vicon, method,
offset_s, cam1_lag, cam2_lag, cam1_r, cam2_r, cam_spread_ms, xcorr_err_ms` (the last only
for pulse trials). Rebuild it with `scripts/shared/video_sync_summary.py`.

- **67 synced: 18 by trigger pulse, 49 by xcorr.**
- `P10/Trial1/NO_VICON_Handsonly` has no Vicon recording.
- `P17/Trial2/Hands_only_goodEMG` isn't in `manual_frames.csv`. It is
  `P17/P17/trial2_only2.csv` (it carries a pulse), synced with `--vicon`.

| participant | trials | pulse | offset_s (min … max) |
|---|---|---|---|
| P1 | 2 | 0 | +0.961 |
| P2 | 4 | 0 | +1.086 … +1.131 |
| P3 | 4 | 0 | +0.026 … +0.031 |
| P4 | 4 | 0 | −0.014 … +0.021 |
| P5 | 4 | 0 | −0.024 … +0.021 |
| P6 | 4 | 0 | −0.044 … −0.009 |
| P7 | 4 | 0 | +0.466 … +0.486 |
| P8 | 4 | 0 | +0.446 … +0.476 |
| P9 | 4 | 3 | +0.797 … +1.066 (drifting) |
| P10 | 4 | 4 | +0.560 … +0.585 |
| P11 | 4 | 4 | +0.561 … +0.575 |
| P12 | 4 | 0 | +0.376 … +0.401 |
| P13 | 4 | 4 | +0.380 … +0.392 |
| P14 | 4 | 1 | +0.452 … +0.501 |
| P15 | 4 | 0 | +0.001 … +0.041 |
| P16 | 4 | 1 | +0.391 … +0.421 |
| P17 | 5 | 1 | +0.371 … +0.406 |

Within a participant the offsets agree to within about 50 ms (P9 excepted), which is an
independent check that the xcorr estimates are sane. Between sessions the offset jumps
by up to 1.1 s. That looks like the RealSense PC's clock being re-synced between sessions.

**Check by eye** (cameras disagree by more than 50 ms):

| trial | note |
|---|---|
| P16 Trial1 Hands_only | 130 ms spread, low r (0.11 / 0.24) |
| P14 Trial2 Hands_only | 70 ms spread |
| P7 Trial2 Hands_only | 70 ms spread |
| P12 Trial1/Trial2 Hands_only | cam1 locked 2–3 s off; cam2 is consistent with P12's other trials |

### Bugs found by the cohort run (fixed)

- Some Vicon exports are **trajectories-only**, with no Devices section and so no `Sync`
  channel (e.g. P5/P6/P7/P9/P12 Trial1 Hands only). `read_sync_channel` returns None for these.
- The CSVs start with a UTF-8 BOM. Read as latin-1, it arrives as `ï»¿`, not `﻿`,
  so section headers are matched with `endswith`. The first version of the check silently
  disabled pulse detection for every trial.
- Vicon labels carry a subject prefix (`P15:Forearm1`), so the forearm exclusion in
  `hand_speed` never matched until the prefix was stripped.

## Open questions

- Why the pulse is caught so rarely, and why P9's clock offset drifted about 6 ms/min.
- Depth streams are not exported.

## Shared views

`OneDrive/.../data_June25/soma_transfer/video/` mirrors `results/video/`: `<P>/<Trial>_<cond>/player.html`
with its MP4s, frame CSVs and `sync.json` beside it, plus `sync_summary.csv`. Every synced
trial has the markers pane, and fitted trials also have the whole-trial 50 Hz mesh.
P16 Trial2 HOI instead has the whole-trial mesh at every frame (200 Hz). 58 trials
also have the HD-sEMG panes (see emg.md). The 51 trials with a SOMA and/or learned-fill fit also
have `labels_player.html` and `labels_player_mesh.html` next to `player.html` (3.5 GB together).
To refresh it, copy `results/video/.` over it again.
