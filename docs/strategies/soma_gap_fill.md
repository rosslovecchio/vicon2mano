# P16: labeller retrain and gap filling before the movement fit (2026-10-28/29)

This page records two experiments on P16, both run with its accepted final
static (`soma_transfer/static_final/P16/`):

1. Retraining the SOMA labeller on a library built from that final static.
2. Filling marker gaps before the movement fit: the existing rigid fill, a
   learned filler trained on synthetic clouds, and the same filler trained on
   real recordings.

Every number below comes from files kept in `results/soma/` or on OneDrive,
and the [Reproduce](#reproduce) section gives the command for each one.
Results that come straight from a log say which log.

## 1. Labeller retrained on the final static

**Question:** does a SOMA labeller trained on synthetic clouds of P16's final
static label P16's trials better than the existing one, which was trained on
the older static calibration (`library_P16.npz`)?

**Library:**
- `calibrate_static.py --final params.npz` builds a library from a final
  static fit instead of fitting the static again (`final_static_library` in
  `vicon2mano/strategies/soma/calibrate.py`).
- It moves each marker's attachment to the final fit's vertex.
- It takes the final fit's betas and scale. The scale is a new
  `MarkerLibrary.scale` field, which `SynthGenerator` applies to the mesh.
- It re-centres each marker's residuals on its offset in the static cloud.

**What the final static changes:**
- **Shape:** the betas are much larger than the old calibration's. Final:
  `[-2.44 -5.05 -1.47 6.67 -7.16 -7.34 -0.91 4.4 3.58 0.71]`, scale 1.049.
  Old: `[-2.53 -1.69 -0.48 -0.43 0.41 0.21 0.51 -0.09 -0.39 0.22]`.
- **Attachments:** some sit on different bones. Marker 2 straddles the PIP,
  marker 3 is on the distal phalanx, and Thumb1 is on the root.
- **Synthetic clouds:** posing this shape with P10's library poses stretched
  some segments in the clouds beyond what P16's recordings show (Thumb1–Thumb2
  52 mm against about 40 mm, Middle1–Middle2 70 against about 65). These were
  ad-hoc checks during the session; no script for them is kept.
- **The static itself:** it was reviewed and judged fine. The stretching is a
  property of the synthetic clouds generated from it.

**Training:** `train.py`, 10,000 steps. The MANUAL monitor scores labelling
accuracy against P16's manually labelled `trial1_handsonly` at step 10,000
(from the training logs):

| Labeller | MANUAL | argmax |
|---|---|---|
| Old (`fullset/P16/train.log`) | 85.0% | 87.9% |
| Retrained (`retrain_final/train_P16.log`) | 80.5% | 86.1% |

**Labels compared with the old ones** (`compare_labels.py`), hand slots only:

| Trial | Slots differing | Labelled only in old | Labelled only in new | Frames with any change |
|---|---|---|---|---|
| Trial 1 Hands only | 4.9% | 3.5% | 0.9% | 48.9% |
| Trial 1 HOI | 8.5% | 4.4% | 1.5% | 70.5% |
| Trial2 Hands only | 4.7% | 3.2% | 0.8% | 42.2% |
| Trial2 HOI | 8.8% | 4.0% | 1.8% | 70.9% |

- Most of the changes are on Pinky2/3 and Ring2/3: 20–45% of their slots in
  the HOI trials.
- The retrained labeller leaves more points unlabelled. After the rigid fill,
  12–23% more values are still empty per trial, for example 121k against 99k
  on Trial2 HOI.

**Fit**, scored on observed markers (`score_observed.py`; the metric is
explained in section 4):

| Trial | Old labels | Retrained labels |
|---|---|---|
| Trial 1 Hands only | 12.2 / 6.4 / 12.6 | 11.7 / 6.2 / 12.1 |
| Trial 1 HOI | 14.3 / 7.6 / 14.4 | 14.6 / 8.0 / 15.3 |
| Trial2 Hands only | 11.7 / 6.7 / 11.9 | 11.6 / 6.6 / 12.1 |
| Trial2 HOI | 14.7 / 7.8 / 14.9 | 14.5 / 8.2 / 14.9 |

The two label sets observe different markers, so this comparison is not on
identical markers. The per-batch summaries the run printed, which include
filled markers, were worse for the retrained labels on HOI (for example
Trial 1 HOI) and flagged 2–9 points fewer frames.

**Conclusion:** retraining did not help. The monitor accuracy dropped, the
labels changed on the hard fingers without a clear gain in fit, and more
points went unlabelled. The old labels (`results/soma/fullset/P16`) remain
the ones in use. One likely reason is the stretched synthetic clouds above.

## 2. Gap-filling methods

The movement fit (`scripts/soma/movement_fit.py`) fits each frame to the
labelled markers it has, and a missing marker leaves its part of the hand
unconstrained. `scripts/soma/gap_fill.py` fills the gaps first.

Every method starts the same way:
- Labelled positions that jump more than `--jump` mm (8) are dropped as label
  switches.
- Gaps of up to `--spline` frames (10) are filled by a cubic spline.

The methods differ in how they fill longer gaps, up to `--max-gap` frames:

| Method | What fills a long gap | Flag |
|---|---|---|
| **Rigid** (the default, `--max-gap` 2000) | Three donor markers whose distance to the missing marker is steadiest around the gap. The marker's position in the donors' frame is taken from both edges of the gap and blended across it. | — |
| **Learned, synthetic** | An MLP predicts the marker from the other markers visible in that frame. It is trained on synthetic clouds of the participant's library. | `--filler filler.pt` |
| **Learned, real** | The same network, trained on the participant's own labelled recordings. | `--filler filler.pt` |

For both learned methods, the prediction is shifted so that it meets the
observed position at each edge of the gap. The shift is blended linearly
across the gap. In these runs `--max-gap 100000` let the filler cover gaps of
any length.

**The network** (`vicon2mano/strategies/soma/fill.py`):
- Each frame is first aligned to a template by a weighted Kabsch fit on the
  near-rigid markers (Palm1–3, Index1, Middle1, Ring1, Pinky1). The network
  therefore sees hand-frame coordinates only, and its output is rotated back.
- It is a 4-layer GELU MLP, 512 wide. It takes in the coordinates of every
  marker plus a mask saying which markers are visible.
- Training hides each observed marker with a per-frame probability between 0
  and 0.5. The loss is the distance error on the hidden markers only.
- The template comes from a generalised Procrustes fit on the training
  frames. A plain mean would not work for real frames, because the hand moves
  and rotates.
- It returns NaN in frames that have fewer than 3 near-rigid markers visible.

**Training data:**
- **Synthetic:** `SynthGenerator.sample(complete=True)` returns clouds before
  any markers are dropped, ghosts are added, or the cloud is rotated and
  centred.
- **Real:** `frame_sampler` draws random frames from labelled CSVs. A frame
  does not need to be complete to be used, since every frame contributes its
  observed markers. P16's four trials give about 190k frames, even though only
  2–23% of them have every marker present.
- **Both:** with `--library` and `--csv` together, each batch is half
  synthetic and half real.

**Fill status:** `gap_fill.py` writes an `.npz` next to its output CSV. Its
`how` array gives each value's source: 0 observed, 1 spline, 2 rigid, 3 jump
left empty, 4 learned.

## 3. Fill accuracy

**How it is measured** (`scripts/soma/gap_fill_check.py`):
- It cuts artificial gaps into stretches where a marker *was* observed. The
  markers cut are Palm1, Index3, Middle3, Ring2, Ring3, Pinky1, Pinky3 and
  Thumb3, with up to 6 cuts per marker and length.
- It refills the gaps with `gap_fill.py` and reports the median and worst
  error against the real positions.
- The cuts come from a fixed seed, but they depend on the file and on the
  `--lengths` list. Within one table the cuts are identical across rows; the
  two tables below used different `--lengths`, so their cuts differ.
- At 200 Hz: 200 frames = 1 s, 1000 = 5 s, 2000 = 10 s, 5000 = 25 s.

**Limitation:** the cuts fall where the marker was visible, so mostly in easy,
open-hand poses. Real gaps happen when a marker is occluded, mostly during
grasps, and nothing here measures fill error there. These numbers are
optimistic for real gaps.

### Rigid fill vs synthetic filler (all four P16 trials)

The synthetic filler is `results/soma/models/filler_P16.pt`, trained on
`library_P16.npz` (the older calibration) for 30k steps. Its synthetic
hidden-marker error was 5.1 mm (`retrain_final/filler_P16.log`). The check
used `--lengths 50,200,1000,2000,5000`. Median error / worst error, mm:

| Trial | Filler | 1 s gap | 5 s gap | 10 s gap |
|---|---|---|---|---|
| Trial 1 Hands only | rigid | 3.2 / 54 | 11.4 / 46 | 14.0 / 44 |
| | synthetic | 2.7 / 24 | 4.5 / 18 | 4.7 / 10 |
| Trial 1 HOI | rigid | 1.5 / 27 | 2.4 / 47 | 2.7 / 29 |
| | synthetic | 1.7 / 25 | 2.7 / 20 | 1.8 / 6.4 |
| Trial2 Hands only | rigid | 3.9 / 57 | 6.5 / 26 | 6.9 / 38 |
| | synthetic | 3.8 / 58 | 5.6 / 22 | 5.9 / 12 |
| Trial2 HOI | rigid | 3.3 / 27 | 5.0 / 36 | 13.3 / 35 |
| | synthetic | 4.2 / 31 | 3.8 / 27 | 3.2 / 22 |

- On short gaps the two methods are about the same.
- On gaps of 5 s or more, the synthetic filler roughly halves the worst
  error. Its median is also lower, except on Trial 1 HOI, where the rigid
  fill already does well.
- At 25 s on Trial 1 HOI (6 cuts), the rigid fill gave 3.8 / 11.6 mm and the
  synthetic filler 2.4 / 3.0 mm.

### Which training data (held-out P16 trials)

Real-data fillers were trained **leave-one-trial-out**: on P16's other three
trials, then tested on the fourth. Every variant trained for 20k steps. The
check used `--lengths 200,1000,2000,5000`. Median error / worst error, mm:

| Trained on | Trial2 HOI 1 s | Trial2 HOI 5 s | Trial2 HOI 10 s | Trial 1 HOI 1 s | Trial 1 HOI 5 s | Trial 1 HOI 10 s | Trial 1 HOI 25 s |
|---|---|---|---|---|---|---|---|
| Synthetic (`filler_P16.pt`) | 3.6 / 26 | 3.8 / 20 | 2.5 / 7.6 | 2.2 / 21 | 2.4 / 32 | 2.5 / 14 | 5.6 / 8.3 |
| P10 manual trial only | 5.9 / 27 | 5.6 / 20 | 3.9 / 11 | 4.1 / 28 | 3.5 / 42 | 5.3 / 16 | 4.7 / 12 |
| P16's other 3 trials | 2.4 / 10 | 2.2 / 8.3 | 1.5 / 2.4 | 1.5 / 8.3 | 3.0 / 7.9 | 3.3 / 7.2 | 3.5 / 4.5 |
| P16 + P10 | 2.7 / 19 | 2.1 / 7.7 | 0.9 / 2.2 | 1.4 / 8.5 | 2.6 / 9.7 | 3.2 / 6.9 | 3.9 / 4.5 |
| P16 + synthetic (`library_P16_final`) | 2.6 / 14 | 2.5 / 7.3 | 1.0 / 2.3 | 1.5 / 11 | 2.2 / 11 | 2.9 / 6.6 | 3.9 / 4.5 |

Hidden-marker training error at the last step, from the training logs:

| Variant | Error |
|---|---|
| P10 only | 1.70 mm |
| P16 other 3 (two folds) | 2.86 / 2.68 mm |
| P16 + P10 | 2.93 / 2.93 mm |
| P16 + synthetic | 5.01 / 4.94 mm |
| P16, all four trials (`filler_P16_all.pt`) | 2.82 mm |

- **The participant's own frames are the best training data.** They cut the
  worst errors from 20–32 mm to 7–10 mm, and most medians are lower.
- **Not every median improves.** On Trial 1 HOI at 5–10 s gaps, the synthetic
  filler's median (2.4–2.5 mm) is lower than the real-data filler's
  (3.0–3.3 mm).
- **Adding P10's frames or synthetic clouds changes little.** The effect is
  small and goes both ways, and no mix is clearly better.
- **P10's trial alone is worse than synthetic.** P10's hand and marker
  placement differ from P16's, so the filler carries P10's offsets onto P16.
  P10's low training error (1.70 mm) only says that it learned P10's own hand.
- **P10's gap-fills were removed first.** P10's manual relabel
  (`Trial2_handsonly_manuallylabelled_filled.csv`) has unreliable gap-fills.
  `keep_observed.py` kept only values with a raw Vicon point within 0.01 mm in
  the same frame: 94.1% of all values, and at least 75.3% for every marker.

## 4. Effect on the movement fit (P16, final static, every 10th frame)

### How much each method fills

Values missing after jump removal, and what each method did with them.
Left = still empty after filling. Taken from each run's
`soma_labelled_filled.json`.

| Trial | Missing | Spline | Rigid: filled / left | Real filler: filled / left |
|---|---|---|---|---|
| Trial 1 Hands only | 163k | 14.5k | 63k / 86k | 104k / 44k |
| Trial 1 HOI | 152k | 11.3k | 45k / 96k | 132k / 9k |
| Trial2 Hands only | 160k | 16.5k | 70k / 74k | 127k / 17k |
| Trial2 HOI | 167k | 9.2k | 59k / 99k | 157k / 1.1k |

- The rigid fill stops at 2000-frame gaps and needs three suitable donor
  markers, so it leaves more than half of the missing values empty.
- What the learned filler leaves empty is frames with fewer than 3 near-rigid
  markers visible. Most of those are in frames 0–10k of Trial 1 Hands only
  (21% of markers missing there), where the hand is not in view.

### Fit error on the markers that were really observed

`score_observed.py` scores the fit on observed markers only (`how == 0`).
The fill does not change which markers count as observed, so every column is
scored on the same markers.

Each cell gives three numbers, each a median over the fitted frames, in mm:
the largest observed-marker distance per frame / the median observed-marker
distance per frame / the largest distance over all markers, filled ones
included.

| Trial | Rigid fill | Synthetic filler | Real-data filler | Real-data filler, filled weighted 0.1 |
|---|---|---|---|---|
| Trial 1 Hands only | 12.2 / 6.4 / 12.6 | 12.8 / 6.6 / 14.5 | 12.4 / 6.4 / 12.9 | 11.4 / 6.3 / 14.6 |
| Trial 1 HOI | 14.3 / 7.6 / 14.4 | 15.5 / 8.5 / 16.3 | 15.7 / 8.0 / 16.4 | 14.7 / 7.5 / 16.9 |
| Trial2 Hands only | 11.7 / 6.7 / 11.9 | 12.2 / 6.7 / 13.0 | 11.9 / 6.6 / 12.2 | 11.5 / 6.5 / 13.7 |
| Trial2 HOI | 14.7 / 7.8 / 14.9 | 16.4 / 8.7 / 17.5 | 15.9 / 8.2 / 16.5 | 15.2 / 7.7 / 17.6 |

- **Hands only:** the real-data filler fits within 0.1–0.3 mm of the rigid
  fill, and the trials become nearly complete.
- **HOI:** filling makes the fit about 1 mm worse on the real markers, with
  either filler.
- **The static is not the cause.** It was reviewed and judged fine, and the
  fit with the rigid fill is the better one on the same static.
- **Likely reason:** the HOI gaps are mostly grasp occlusions, which the
  accuracy test does not sample (see the limitation in section 3). A filled
  marker a few mm off in a grasp pulls the mesh away from the real markers.
  Hands-only trials, with fewer grasps, barely change.
- **The fix: down-weight the filled values.** `movement_fit.py --w-filled`
  gives filled values a lower weight than observed ones. It reads the
  fill-status `.npz` next to `--csv`, and the default of 1.0 keeps the old
  behaviour. With a low weight, a fill only guides the mesh where nothing
  real constrains it.
- **Result at weight 0.1:**
  - The median observed-marker distance is lower than with the rigid fill in
    all four trials (by 0.1–0.2 mm).
  - The largest observed-marker distance is lower in both Hands-only trials:
    11.4 against 12.2 mm, and 11.5 against 11.7.
  - On HOI that largest distance is still 0.4–0.5 mm higher than with the
    rigid fill, against 1.2–1.4 mm higher unweighted.
  - The fit now sits further from the filled values: the all-marker column
    rises by 1–2 mm. That is the intended trade, since filled values are
    estimates.
- **Not fitted in any run:** frames 50k–57k of Trial2 Hands only.
  `fit_trials.py` fits at most 5 batches of 10k frames per trial, so all four
  columns cover the same frames.

## Recommendation

1. Train each participant's filler on that participant's own labelled trials
   (`train_filler.py --csv`), not on synthetic clouds or another
   participant's recording.
2. Fill every gap with it (`gap_fill.py --filler ... --max-gap 100000`), and
   fit with filled values down-weighted (`movement_fit.py --w-filled 0.1`).
   The trials become nearly complete, and the fit to the real markers is at
   least as good as with the rigid fill, except the largest-marker error on
   HOI (+0.4–0.5 mm).
3. Only 0.1 was tried. A sweep over the weight (for example 0.03 and 0.3),
   and a check of fill error inside real grasp occlusions, would settle the
   remaining HOI gap.
4. Keep P16's old labels; the final-static labeller did not improve them.

## Reproduce

**Environment:**
- Windows, `.viconvenv`, run from the repo root. `$D` is the data root,
  `C:/Users/RL000009/OneDrive - Vrije Universiteit Brussel/A-Skills/data_June25`.
- MANO weights: pass `--mano-dir "$D/mano_v1_2/models"` wherever a script
  defaults to `../clean_kinematics`, which does not exist on this machine.
- `train_filler.py` already defaults to that path.

**Commits:**
- `e090368`: library from a final static fit.
- `7c19595`: learned filler.
- The next commit adds `--w-filled`, `--seed` and the experiment scripts
  (`fit_trials.py`, `score_observed.py`, `keep_observed.py`,
  `compare_labels.py`).

**Caveats on exact reproduction:**
- **Runs made before the scripts were committed** used copies of the
  `fit_trials.py` loop with hard-coded paths. Those copies are kept as
  `results/soma/{retrain_final,learned_fill,real_fill,real_fill_w01}/run_trials.py`.
  The committed `fit_trials.py` reproduces their steps and arguments.
- **The rigid baseline** was fitted from `../vicon2mano_frozen`, a second
  checkout whose `results/` is a link to this repo's `results/`. Its code was
  on this branch, with the same `gap_fill.py` rigid path and the same
  `movement_fit.py`.
- **Fillers are not bit-for-bit reproducible.** They were trained before
  `--seed` existed: the synthetic generator was seeded (0), but the network
  initialisation and the real-frame sampling were not. Retraining gives
  statistically similar fillers, not identical ones. The trained `.pt` files
  are kept and give the exact numbers above.

```bash
PY=.viconvenv/Scripts/python.exe; D="C:/Users/RL000009/OneDrive - Vrije Universiteit Brussel/A-Skills/data_June25"
L=results/soma/fullset/P16                       # old SOMA labels: <trial>/soma_labelled.csv

# 1. Labeller retrained on the final static
$PY scripts/soma/calibrate_static.py --participant P16 --static $L/static/soma_labelled.csv \
    --final "$D/soma_transfer/static_final/P16/P16_static_final_params.npz" \
    --mano-dir "$D/mano_v1_2/models" --out results/soma/libraries/library_P16_final.npz
$PY scripts/soma/train.py --library results/soma/libraries/library_P16_final.npz --mano-dir "$D/mano_v1_2/models" \
    --steps 10000 --out results/soma/models/labeler_P16_final.pt --monitor-raw "$D/P16/trial1_handsonly.csv" \
    --monitor-truth "$D/P16/trial1_handsonly_manuallylabelled.csv" --monitor-name MANUAL
$PY scripts/soma/compare_labels.py P16 --model results/soma/models/labeler_P16_final.pt --new results/soma/retrain_final
$PY scripts/soma/fit_trials.py P16 --labels results/soma/retrain_final --work results/soma/retrain_final/movement \
    --name {p}_retrained --note "labeller retrained on the final static"

# 2. Rigid baseline
$PY scripts/soma/fit_trials.py P16                # -> results/soma/movement, OneDrive results/soma/P16

# 3. Synthetic filler, accuracy against the rigid fill
$PY scripts/soma/train_filler.py --library results/soma/libraries/library_P16.npz --steps 30000 \
    --out results/soma/models/filler_P16.pt
for t in Trial_1_Hands_only Trial_1_HOI Trial2_Hands_only Trial2_HOI; do
  $PY scripts/soma/gap_fill_check.py --filled results/soma/movement/P16/$t/soma_labelled_filled.csv \
      --lengths 50,200,1000,2000,5000 --max-gap 100000                                      # rigid
  $PY scripts/soma/gap_fill_check.py --filled results/soma/movement/P16/$t/soma_labelled_filled.csv \
      --lengths 50,200,1000,2000,5000 --max-gap 100000 --filler results/soma/models/filler_P16.pt
done
$PY scripts/soma/fit_trials.py P16 --filler results/soma/models/filler_P16.pt --max-gap 100000 \
    --work results/soma/learned_fill/movement --name {p}_learnedfill --note "learned gap fill"

# 4. Real-data fillers, held out by trial
R=results/soma/real_fill; P10=$R/P10_trial2_manual_observed.csv
$PY scripts/soma/keep_observed.py --labelled "$D/P10/Trial2_handsonly_manuallylabelled_filled.csv" \
    --raw "$D/P10/Trial2_handsonly.csv" --out $P10
$PY scripts/soma/train_filler.py --csv $P10 --out $R/filler_P10.pt
all="Trial2_HOI Trial2_Hands_only Trial_1_HOI Trial_1_Hands_only"
for test in Trial2_HOI Trial_1_HOI; do
  rest=$(for t in $all; do [ $t != $test ] && echo $L/$t/soma_labelled.csv; done)
  $PY scripts/soma/train_filler.py --csv $rest --out $R/filler_P16_no_$test.pt
  $PY scripts/soma/train_filler.py --csv $rest $P10 --out $R/filler_P16P10_no_$test.pt
  $PY scripts/soma/train_filler.py --library results/soma/libraries/library_P16_final.npz --csv $rest \
      --out $R/filler_synP16_no_$test.pt
  for f in results/soma/models/filler_P16 $R/filler_P10 $R/filler_P16_no_$test $R/filler_P16P10_no_$test $R/filler_synP16_no_$test; do
    $PY scripts/soma/gap_fill_check.py --filled results/soma/movement/P16/$test/soma_labelled_filled.csv \
        --lengths 200,1000,2000,5000 --filler $f.pt --max-gap 100000
  done
done

# 5. Real-data filler on all four trials, fits, and fits with filled values weighted 0.1
$PY scripts/soma/train_filler.py --csv $(for t in $all; do echo $L/$t/soma_labelled.csv; done) --out $R/filler_P16_all.pt
$PY scripts/soma/fit_trials.py P16 --filler $R/filler_{p}_all.pt --max-gap 100000 --work $R/movement \
    --name {p}_realfill --note "gap fill learned from real frames"
mkdir -p results/soma/real_fill_w01/movement/P16
for t in $all; do mkdir -p results/soma/real_fill_w01/movement/P16/$t
  cp $R/movement/P16/$t/soma_labelled_filled.* results/soma/real_fill_w01/movement/P16/$t/; done   # reuse the fill
$PY scripts/soma/fit_trials.py P16 --w-filled 0.1 --work results/soma/real_fill_w01/movement \
    --name {p}_realfill_w01 --note "gap fill learned from real frames, filled values weighted 0.1"

# 6. Scores
$PY scripts/soma/score_observed.py P16 rigid=results/soma/movement synthetic=results/soma/learned_fill/movement \
    real=$R/movement real_w01=results/soma/real_fill_w01/movement
$PY scripts/soma/score_observed.py P16 old_labels=results/soma/movement retrained=results/soma/retrain_final/movement
```

## Checks already made on these outputs

- **`score_observed.py`** reproduces, to the 0.1 mm printed, the observed-only
  table that the session's ad-hoc scoring script produced earlier.
- **`keep_observed.py`** writes a byte-identical file to the one used to train
  `filler_P10.pt`.
- **`compare_labels.py`** reproduces the label comparison of the session's
  original script.
- **`tests/soma/test_fill.py`:**
  - the Kabsch alignment recovers a rigid motion;
  - the filler's prediction moves with the hand (rotation and translation
    equivariance);
  - training on real frames of a moving rigid hand recovers a hidden marker
    to within 10 mm.

## Where things are

- **Code:**
  - `vicon2mano/strategies/soma/fill.py`: the network, `train_filler`,
    `frame_sampler`, `predict`, `save` and `load`
  - `vicon2mano/strategies/soma/calibrate.py`: `final_static_library`
  - `vicon2mano/strategies/soma/library.py`: `MarkerLibrary.scale`
  - `vicon2mano/strategies/soma/synth.py`: `sample(complete=True)` and the
    scale
  - scripts: `train_filler.py`, `gap_fill.py --filler`, `gap_fill_check.py`,
    `movement_fit.py --w-filled`, `calibrate_static.py --final`,
    `fit_trials.py`, `score_observed.py`, `keep_observed.py`,
    `compare_labels.py`
- **Models:**
  - `results/soma/libraries/library_P16_final.npz`
  - `results/soma/models/labeler_P16_final.pt`
  - `results/soma/models/filler_P16.pt` (synthetic)
  - `results/soma/real_fill/filler_*.pt` (real-data and mixed variants;
    `filler_P16_all.pt` is the one in use)
- **Logs:** each `results/soma/{retrain_final,learned_fill,real_fill,real_fill_w01}/`
  folder holds the training logs (`train_*.log`, `filler_P16.log`) and the
  batch log `trials_P16.log`. `real_fill/run.log` holds the held-out
  gap-check output.
- **Views and data** on OneDrive, under `soma_transfer/results/soma/`. Each
  trial folder holds the live player HTML, plus `fit.npz`, `pred.npz`,
  `summary.json` and `flags.json` per batch, and the filled CSV with its
  report and status files.

  | Folder | Run |
  |---|---|
  | `P16` | rigid fill |
  | `P16_retrained` | retrained labels |
  | `P16_learnedfill` | synthetic filler |
  | `P16_realfill` | real-data filler |
  | `P16_realfill_w01` | real-data filler, filled values weighted 0.1 (its filled CSVs are the ones in `P16_realfill`) |
