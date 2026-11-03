# EMG (Novecento+ HD-sEMG): reading, quality, sync

Code: [`vicon2mano/core/emg.py`](../vicon2mano/core/emg.py) (reader + metrics),
[`scripts/emg/emg_quality.py`](../scripts/emg/emg_quality.py) (cohort report),
[`tests/core/test_emg.py`](../tests/core/test_emg.py).

```bash
.viconvenv/Scripts/python.exe -W ignore scripts/emg/emg_quality.py              # all trials, ~25 min at --workers 6
.viconvenv/Scripts/python.exe -W ignore scripts/emg/emg_quality.py D:/june25_data/P15/Trial1/Hands_only/Novecento
```

Outputs go to `results/emg/quality/` (gitignored, regenerable). Running on a
subset **overwrites `summary.csv` with only that subset**.

## Where the data is

`D:\june25_data\<P>\<Trial1|Trial2>\<Hands_only|HOI>\Novecento\*.otb4`
(`EMG_ROOT` in `core/emg.py`). 67 recordings over P1-P17. Extras that are not
regular trials: `P10/Trial1/NO_VICON_Handsonly`, `P17/Trial2/Hands_only_goodEMG`,
`P16/EXTRA`. Beside each `.otb4` sit OTBiolab `.npy` exports
(`track1/5/9/13_novecento+_(...)_.npy` = 4 grids in mV, `track17_..._aux_1.npy`);
P1/P2 also have older `emg_data_grid*.npy`. The reader uses only the `.otb4`.

The Vicon side of each trial is `D:\june25_data\<P>\<P>\<trial>.csv` (+ `.c3d`).
Its first block (`Devices`, 1000 Hz) carries the analog **`Sync`** channel
(V), followed by `Trajectories` at 200 Hz.

## The `.otb4` format (Novecento+)

A plain tar archive:

| member | content |
|---|---|
| `Tracks_000.xml` | one `TrackInfo` per track: file, channel offset, fs, ADC bits/range, gain, sensor |
| 4 x `<id>.sig` (202 MB) | one per probe: 102 int32 channels interleaved = 96 EMG + 4 quaternion + buffer + ramp |
| `<id>.sig` (32 MB) | 16 AUX channels: AUX 1-14 + 2 load cells |
| `<id>.sig` (63 MB) | 8 device-control channels at 8 kHz (ignored) |
| `<timestamp>.sig` (900 MB) | whole raw interleaved stream, duplicate of the above (ignored) |
| `DeviceParameters.xml`, `DockPanel.xml` | GUI state |

All signals run at 2000 Hz. The grids are HD08MM1606 (8 mm pitch, 96
electrodes, Bio96HD adapter), monopolar, 1 Hz hardware high-pass. The four
grids are channels 1-96, 103-198, 205-300 and 307-402.

**Decoding.** Reshape each `.sig` as int32 Fortran-order to
`(ChannelsInBlock, T)`. Convert to mV with `raw * ADC_Range / 2**ADC_Nbits *
1000 / Gain`, which for EMG is 2.861e-4 mV/count. This is the formula OT's
own reader uses
([OTB-Python `OpenOTB4.py`](https://github.com/OTBioelettronica/OTB-Python)).
That script can't be imported: it opens a Tk file dialog at module level and
reads `NumberOfChannels`/`AcquisitionChannel`, which Novecento+ XML doesn't
have. `read_otb4` reads `ChannelsInBlock`/`ChannelOffsetInSubPacket` instead,
streams the members out of the tar (about 0.6 s per file), and matches the
OTBiolab `.npy` exports to 5e-17 mV.

Quirks worth knowing:

- **The EMG never leaves the int16 range**, although the XML says 24 bit.
  ±32767/-32768 (±9.4 mV) is the clipping rail (`RAIL`).
- **The ramp channel is a per-probe 16-bit counter**, +1 per sample.
  `ramp_faults` returns skips (lost samples) and repeats. A repeat is a
  duplicated sample (the whole EMG frame equals the previous one): the
  receiver fills a slot to compensate clock rate, about 1-5 per probe per
  trial (~10 ppm). Lost samples are rarer (P15/T1: 1, on grid 4).
- **The probes' ramps start 5-19 counts apart and drift.** That is *not* an
  inter-grid misalignment: the shared common-mode signal lines up across
  grids within 1-3 samples. Do not shift grids by ramp offset.
- **AUX values are kept as raw counts** (±32768 full scale). The XML gain
  formula gives 4x OTBiolab's own AUX `.npy` scale, so volts are ambiguous.
  Sync only needs edge times anyway.

## Quality metrics (`channel_quality`, `flag_channels`)

Rail and flat checks use raw counts. Everything else is computed after a
**common-average reference** (grid median removed) and a 20-450 Hz band-pass.
Without the CAR, the shared common mode makes every channel correlate ~0.98
with its neighbours, which hides bad contacts. `mono_noise_uv` keeps the
monopolar floor for reference.

| metric | definition |
|---|---|
| `rail_frac` | fraction of samples at the ±32767 rail |
| `flat_s` | seconds (1 s windows) with peak-to-peak ≤ 2 counts |
| `noise_uv` / `active_uv` | 5th / 99th percentile of 0.5 s RMS windows |
| `snr_db` | `20 log10(active / noise)` |
| `line_ratio` | power at 50-250 Hz harmonics (±1 Hz) / 20-450 Hz power |
| `lowfreq_ratio` | 1-20 Hz power / 20-450 Hz power (motion artefact, lifted electrode) |
| `nbr_corr` | mean of the channel's 3 best correlations within its grid (neighbours on an 8 mm grid share signal) |

Flags (a channel can carry several; relative to its own grid):
`dead` (flat for the whole trial), `clipping` (>0.1% at rail), `dropout`
(flat >1 s but not dead), `noisy` (noise >3x grid median), `line` (line ratio
>2x grid median), `motion` (lowfreq_ratio >1), `isolated` (nbr_corr <0.5).
The thresholds were set by eye on P15/Trial1. They are a first triage; check
the heat map before discarding a channel.

The heat map (`<tag>.png`) shows log10 RMS per channel × time for all four
grids. White rows are dead channels, cyan bands are AUX1 sync bursts. A drop
shows as a dark streak, a lifted region as a featureless block.

## Results

### P15 / Trial1 / Hands_only (worked example)

| grid | flagged | dead | CAR noise µV | SNR dB | lost / dup samples |
|---|---|---|---|---|---|
| 1 | 10 | ch 89 | 11.5 | 23.5 | 0 / 5 |
| 2 | 5 | ch 8, 89 | 12.7 | 23.9 | 0 / 1 |
| 3 | 21 | – | 6.4 | 20.7 | 0 / 3 |
| 4 | 7 | – | 5.2 | 24.3 | 1 / 4 |

- Partial dropouts: grid 1 ch 8 is flat for 125 s, grid 2 ch 90 for 177 s,
  grid 1 ch 90 and grid 2 ch 7 for 7-11 s.
- Grid 3 ch 79-94 carry no muscle modulation, only artefact (lowfreq_ratio
  2-46). Most likely that edge of the grid lifted. Grid 1 ch 4-10 show the
  same on a smaller scale.
- No clipping. Line noise is about 6% of band power. Monopolar noise floor is
  30-58 µV, CAR floor 5-13 µV.

### Cohort (67 recordings, 268 grids, 25,728 channels; run 2026-10-29)

Typical grid: median CAR noise **7.5 µV**, SNR **22 dB**, line ratio
**0.06**, about **30 of 384 channels flagged per trial** (8%). Flag totals:
line 2236, noisy 1320, isolated 823, motion 467, dead 78, clipping 45,
dropout 32 (a channel can carry several).

Mean flagged channels per trial (of 384), by participant:
P3 1 · P12 9 · P4 23 · P7 23 · P10 23 · P5 24 · P2 28 · P8 28 · P1 30 ·
P6 32 · P15 34 · P13 36 · P14 47 · P11 76 · P16 108 · P9 161 · P17 181.

**Recurring dead channels:** ch 8 and ch 89 are the dead ones on every grid
(grid 2 ch 89 ×9, grid 2 ch 8 ×9, grid 1 ch 89 ×7, grid 4 ch 89 ×7 ...).
Probably adapter/connector pins rather than skin contact; check the
Bio96HD pinout before blaming the subject.

**Grids to drop or treat with care** (≥48 of 96 flagged):

| trial | grid(s) | failure |
|---|---|---|
| P9, all 4 trials | 4 (and 3 in both HOI trials) | **probe link fault**: ramp counter broken (1.7k-24k skip events from ~20 s), single-sample spikes to the rail, decorrelated signal. All 96 channels `isolated` |
| P11/Trial1/HOI | 4 | same probe link fault (6.1k ramp skips) |
| P17/Trial2/Hands_only | all | 250 µV floor, 98% of band power is mains: unusable. **Use `Hands_only_goodEMG`** (the re-recording, 28 flagged) |
| P17/Trial1/HOI, P17/Trial2/HOI | 1-4 / 3 | mains-dominated (line ratio 0.66-0.94) at a low floor. Probably recoverable with a 50 Hz comb notch |
| P16/Trial1/Hands_only | 1, 3 | 24-44 µV floor and line ratio ~0.53. P16/Trial2/HOI is also line-heavy (0.25) |

Rule of thumb: `line` (and mains-driven `noisy`) channels are fixable with a
notch. `dead`, `dropout`, `isolated` and a probe link fault are not.
`motion` (lifted electrodes) is usable only after high-pass filtering, and
with caution.

**Ramp:** apart from the probe faults above, every grid has 0-15 duplicated
samples and 0-14 skips per trial. These are negligible for EMG and well under
1 ms for sync.

**AUX1 sync per trial** (first interval, from `aux1_bursts`):

- **No AUX1 activity:** P1/Trial1/Hands_only, P12/Trial2/Hands_only,
  P17/Trial2/HOI.
- **AUX1 already high when recording starts** (`0.0-x`): P3 (all 4 trials),
  P6/Trial2 (both). Only the falling edge at `x` is usable.
- **Long merged intervals** (e.g. P4/Trial1/HOI `14.684-76.99`,
  P6/Trial1/Hands_only `21.478-97.962`). Several events lie closer than 0.5 s
  and merged, so inspect them before trusting the interval boundaries.
- Everything else has 1-8 clean intervals. P2/Trial2/Hands_only has a
  28-pulse train.

## Grid layout (16 × 6)

The HD08MM1606 is 16 rows × 6 columns. The `.otb4` does not record the
channel-to-electrode map, so it was inferred from the data. Neighbouring 8 mm
electrodes share signal, so only the true layout gives high correlation between
grid neighbours. On P16/Trial2/HOI (CAR, 20-450 Hz), **column-major** wins on all
four grids: channel `k` (0-based) sits at row `k % 16`, column `k // 16`
(`GRID_ROWS, GRID_COLS` in `core/emg.py`).

| layout | mean neighbour r (4 grids) |
|---|---|
| column-major | 0.51-0.68 |
| row-snake / column-snake / row-major | 0.16-0.36 |

Within column-major, vertical neighbours give r 0.61-0.70, horizontal 0.40-0.65 and
diagonal 0.34-0.62. The column wrap (16→17, consecutive numbers but not
adjacent electrodes) gives r ≤ 0.13. The data cannot tell which end of the grid is
proximal, so the maps show channel 1 top-left.

## Sync with Vicon (1 s pulse case implemented)

**Implemented:** `core.video_sync.aux_pulse_offset(aux, fs, vicon_csv)` returns
`emg_t - vicon_t` when both AUX 1 and Vicon `Sync` show the single 1 s level pulse
(P16/Trial2/HOI: AUX 5.838 s vs Vicon 1.906 s, so 3.9325 s). The pulse-train and
burst shapes below still need a matching step; until then pass `--emg-offset` to
`scripts/emg/emg_pane.py`.

### Visualization (`scripts/emg/emg_pane.py`)

Adds the EMG to a trial's RealSense player (`docs/video_sync.md`), on the same clock
as the cameras, the Vicon data and the mesh. It writes `emg.json`: per-channel RMS of the CAR 20-450 Hz
signal in 50 ms windows (`core.emg.rms_envelope`), as log10 µV quantised to uint8
(0 = 1 µV, 255 = 1 mV), about 2 MB per trial. The player then shows:

- one 16×6 electrode map per grid at the cursor (mean of ±100 ms), and
- a channel × time heatmap per grid on the strip's ±3 s axis.

Both have a scale toggle: **per channel** (each channel stretched between its own 5th
and 99th percentile, so activation stands out) or **absolute** (the quality heat
map's scale). Channels flagged in `results/emg/quality/<tag>.csv` are marked red, and
the EMG offset can be adjusted live.

The same sync signal reaches Novecento **AUX 1** and Vicon's analog
**`Sync`** channel (first block of the trial CSV, 1000 Hz, header
`Imported Generic Analog #1 - Electric Potential`). The offset is the
difference in time of the same event: `t_vicon = t_emg - offset`.
Vicon trajectories are 200 Hz with 5 analog subframes per frame.

**The pulse pattern is not the same in every trial.** Three shapes were seen:

| shape | example | AUX1 | Vicon `Sync` | EMG starts earlier by |
|---|---|---|---|---|
| single ~1 s level pulse | P10/Trial2/Hands_only | 14.813-15.812 s | 8.705-9.704 s | 6.108 s |
| lone pulse + two ~50 Hz square-wave bursts | P15/Trial1/Hands_only | 22.318-22.54, 23.464-25.163, 128.458-135.518 s | lone pulse ~12.55 s, burst ~13.69 s | ~9.77 s |
| train of ~0.2 s pulses every ~7 s | P2/Trial2/Hands_only | 52.398, 57.326, 65.202 ... s | 43.041, 47.97, 55.846 ... s | ~9.36 s, but the first Vicon pulses have no AUX1 partner, so match the *train*, not the first pulse |

- AUX1 is slew-limited and clips at full scale. The Vicon copy is clean.
  Align on interval/edge times, not amplitude.
- Where a trial has two or more events far apart, the clock drift between the
  2000 Hz EMG and the 1000 Hz Vicon analog can be measured, not just the offset.
- `summary.csv` column `aux1_bursts` lists every AUX1 interval per trial
  (`aux_bursts` in the script; intervals closer than 0.5 s merge).
- **No sync recorded:** P12/Trial2/Hands_only has flat AUX1, and its Vicon
  `Sync` (`Static_trial03.csv`, per the P12 naming note in CLAUDE.md) is
  pure noise. Other trials with an empty `aux1_bursts` are in the cohort
  table below; check their Vicon side before giving up on them.

## Changelog

- **2026-10-29**: reader, metrics, cohort script, tests (3). Format, quirks
  and P15/Trial1 results above.
- **2026-10-29**: after the first cohort pass, three fixes:
  1. The noisy and line flags got absolute floors (50 µV, 0.5). Purely
     relative thresholds let P17/Trial2/Hands_only (mains everywhere) show
     only 35 flagged channels; it now shows 382.
  2. AUX1 detection groups *intervals* off baseline, not edges. The edge
     version split a 1 s level pulse into two "bursts".
  3. The `lost_samples` column was renamed to `ramp_skips`: it counts skip
     events, and on a broken probe the counter is garbage, so the number is
     not a sample count.

  Also: the sync pattern varies by trial (1 s pulse / pulse + bursts /
  pulse train), and the cohort results above were added.
- **2026-10-29**: grid layout inferred (16×6, column-major; see Grid layout).
  `rms_envelope` / `car_bandpass` were factored out of `channel_quality`
  (same numbers). The 1 s pulse sync (`aux_pulse_offset`) and the player panes
  (`scripts/emg/emg_pane.py`) were added. First used on P16/Trial2/HOI.
