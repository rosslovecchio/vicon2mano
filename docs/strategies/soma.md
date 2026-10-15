# Strategy: SOMA (learned, label-free labelling)

*See `../../CLAUDE.md` for the index.*

Hand adaptation of SOMA (Ghorbani & Black, ICCV 2021,
[arXiv 2110.04431](https://arxiv.org/abs/2110.04431)): a self-attention
network over the raw, unordered marker cloud of one frame, with an
optimal-transport (Sinkhorn) assignment that has "dustbin" slots for ghosts
and missing labels, trained **only on synthetic clouds**. Per-frame labels are
then voted per raw trajectory piece (SOMA's "tracklet labelling").

Self-contained by design: imports `vicon2mano.core` only, never another
strategy or `scripts/`.

- Code: `vicon2mano/strategies/soma/` (`library.py`, `synth.py`, `model.py`, `realdata.py`, `calibrate.py`)
- Scripts: `scripts/soma/` (`build_library.py`, `preview_synth.py`, `train.py`, `eval_trial.py`, `calibrate_static.py`)
- Tests: `tests/soma/`
- Output: `results/soma/` (gitignored, regenerable)

Status (2026-10-15): **trained; on raw P10/Trial2 it cuts Vicon's labelling
errors ~20x (97.72% -> 99.89% correct).** Generalisation to other trials and
participants not yet tested -- see the caveat under Results.

## Why this, after the hand-built scores

On P10/Trial2 every hand-built identity score (pair distances, palm-frame
position, gap continuity) plateaued at ~20% of wrong labels fixed: Vicon
misnames whole runs of trajectory pieces consistently, so any score that
judges a piece against its neighbours' current labels finds the wrong run
self-consistent (details in `mano.md`, "Tested and rejected"). A network that
sees the whole cloud at once, and has seen every confusion many times in
training, does not depend on the current labels at all.

## Marker library (`library.py`, `scripts/soma/build_library.py`)

From a trusted labelled trial plus its surface fit, each of the 22 markers
gets an **attachment** (one MANO face + barycentric point) and a per-frame
**residual**: the real marker's offset from that point in the face's local
frame. Re-applying a frame's residuals to its own fitted pose reproduces the
real cloud exactly (tested to 1e-3 mm). Residuals are NaN where the manual
position is a gap-fill rather than a real observation (it must coincide with
a raw Vicon point to count) or the fit is poor there, so only measured
geometry is ever copied. Forearm markers are not on MANO; they ride on the
nearest wrist face, so their residual carries that frame's real wrist angle.

Built from P10/Trial2: 36,620 library frames, median 21/22 real markers.
Attachment choice matters: taking the majority nearest face put Pinky3 89mm
from its marker (a fingertip votes over 4+ small faces), so the top voted
faces are tried and the one with the smallest median distance kept (10.8mm).

## Generator (`synth.py`)

Per cloud: pose (one library frame, or in `p_mix` = 20% of clouds each finger
from its own frame) + Gaussian pose noise, `betas` from the subject or the
MANO prior, attachment + residual, extra occlusions weighted by real miss
rates up to 4 absent in total, 0-2 ghosts, uniform random rotation,
median-centred, shuffled. 20,000 clouds in 0.3 s on the laptop GPU.

Realism vs real P10/Trial2 frames (`preview_synth.py`):

| | real | synthetic |
|---|---|---|
| pair-distance medians, mean / worst \|diff\| | -- | 0.6 / 3.5 mm |
| pair-distance spread (median IQR) | 9.8 mm | 10.8 mm |
| fingertip nearest neighbour, p10 (Index3/Middle3/Ring3/Pinky3) | 18.8/16.5/15.1/15.2 | 23.8/18.6/16.9/20.3 |

Mixing every finger independently (`p_mix` = 1) broke the real coupling
between neighbouring fingers and made converging fingertips -- the confusable
case -- rare (worst median error 19mm, Middle3-Ring3). Pose noise 0.08 rad did
the same. Both were cut. Close Index3/Pinky3 approaches are still somewhat
under-represented; if the trained network confuses exactly those, weight
library frames by fingertip proximity.

## Network (`model.py`)

Per-point MLP embedding -> 8 pre-norm transformer encoder layers (d 128,
4 heads) -> linear score per label -> log-space Sinkhorn (30 iters) with one
learnable dustbin score bordering the matrix (SuperGlue-style marginals:
1 per point / label, M for the dustbin row, n for the dustbin column).
1.08M parameters. Loss: NLL of the true assignment (point -> label, ghost ->
dustbin column, absent label -> dustbin row). Inference: one-to-one
Hungarian where each point may also take its own dustbin; then per piece
(raw column cut at gaps / >5mm jumps) the label with the largest summed
confidence, clashes resolved by vote share (`realdata.vote_pieces`).

Early in training Sinkhorn leaks a present label's mass into the dustbin row,
leaving label-vs-ghost ties (hard accuracy ~0 while argmax was ~40% at 400
steps). It is gone by step 1000 (98.73% vs 98.71%), so no fix was needed.

## Results (2026-10-15)

`scripts/soma/train.py`: 30k steps, batch 256, AdamW 1e-3 cosine, synthetic
only, 44 min on the RTX A2000 laptop GPU. Real-data monitor (2,000 raw
Trial2 frames, never trained on, never used for checkpoint choice): 98.73%
at step 1k, 99.94% at 3k, flat at 99.95% thereafter.

`scripts/soma/eval_trial.py` on the whole raw P10/Trial2 (874,091 judgeable
slots, truth = manual labels; 34 s):

| labels | all frames | <=4 missing (96% of frames) |
|---|---|---|
| raw Vicon | 97.72% | 97.84% |
| SOMA per frame | 99.83% (0.05% unlabelled) | 99.92% |
| **SOMA + piece vote** | **99.89%** (0.01% unlabelled) | **99.95%** |

Where the voted labels differ from Vicon's (20,677 slots) SOMA is right 95.9%,
Vicon 3.9%. For scale, the best hand-built piece labeller reached 99.20% on
the held-out half (`mano.md`), GMM was net harmful.

**Caveat:** the library -- attachments, residuals, poses -- comes from this
same trial, so the network has seen Trial2's hand and marker placement
(never its labels). This is a best case. P10's other trials and other
participants are the real test.

## Joint angles, P10 Trial2 (2026-10-16)

Surface fits (`scripts/mano/fit_surface_markers.py`, P10 configs, seed gap
fix) on the raw trial with Vicon's labels vs SOMA's, scored against the
manual-label fit. In frames where Vicon's labels were wrong and the manual
reference is a real observation (not a fill):

| fit | mean angle error | joint-frames > 10 deg |
|---|---|---|
| Vicon labels | 20.12 deg | 55.6% |
| **SOMA labels** | **0.91 deg** | **1.1%** |

Per finger at the PIP: index 35.1 -> 0.3, middle 19.8 -> 0.6, ring 18.3 ->
1.0, pinky 16.8 -> 0.7 deg. Where Vicon was right both fits agree with the
reference to ~0.5 deg. Model-free check (correlation of PIP flexion with the
bend measured off the three real markers): SOMA 0.994-0.997, manual-label fit
0.995-0.998, Vicon 0.944-0.985. Frames with any finger joint > 120 deg:
SOMA 176, manual-label fit 688 (its gap-fills), Vicon 2,276.

This comparison also exposed a fitter bug (fixed in `56998db`): unobservable
seed angles were set to 0, so on data with gaps fingers started straight and
stayed there. Before the fix, the SOMA-label fit looked worse than Vicon's
on ring and pinky.

## Label-free evaluation (`eval_trial.py`)

Without trusted labels, `eval_trial.py` reports how decisive the piece votes
are, what is left unlabelled, where SOMA differs from Vicon, and **segment
consistency**: per finger segment, the share of frames within 6 mm of that
labelling's own median length. Another finger's marker in a slot stretches
or shrinks a segment, so wrong labels show up there. It is weaker evidence
for SOMA than for Vicon: a network that prefers plausible geometry can be
consistently wrong (e.g. shift a marker one segment along a finger) and still
score well. `--truth-name` marks a labelled export as partly trusted (prints
"agree", not "correct").

## P10 Trial2 HOI (2026-10-16)

Same model, no retraining. SOMA differs from Vicon on 11.8% of points,
mostly fingertips (Pinky3 72%, Ring3 64%, Middle3 41%), and agrees less with
the HOI manual file (91.7% vs Vicon's 95.5%). But the manual file keeps
Vicon's physically impossible fingertip segments (Middle2-Middle3 upper
quartile ~53 mm vs 38 mm hands-only); share of frames within 6 mm of the
hands-only length, Vicon / manual / SOMA: Middle 67 / 62 / 99.9%, Ring
64 / 60 / 88%, Pinky 66 / 76 / 99.6%. The HOI manual file is not usable as
truth at the fingertips. ~5% of raw points are object markers/reflections the
manual file deleted; SOMA (trained with <= 2 ghosts) tends to label them.

## Static calibration for a new participant (`calibrate.py`)

Shape and each marker's mean offset come from the participant's static;
poses and motion offsets (sliding, noise) stay from the base library.
Forearm markers are not fitted (their offset depends on wrist angle) and keep
the base offsets. A marker still >15 mm off after the fit is flagged suspect
and excluded -- this catches a palm marker on a finger, but **not** a
fingertip swap (a finger bends to reach it), so the script also screens
segment lengths against the base subject.

**P12** (`manual_frames.csv`: Trial 1 Hands only = `Static_trial01`, Trial 1
HOI = `02`, Trial2 Hands only = `03`, Trial2 HOI = `04` -- movement trials
despite the names; the real static is the unmapped `Static_trial.csv`,
0.06 mm/frame). **Vicon's labels on that static are wrong almost everywhere**:
calibrating on them gave fit errors to 217 mm, betas to -4.5 and every
segment outside P10's range. Relabelled by the P10 SOMA model first (95% of
points differ from Vicon, incl. all forearm and palm columns), it calibrates
cleanly: hand-marker fit 0.6-9.3 mm, |betas| <= 1.4, all 18 hand markers
calibrated. Distal segments then match P10 within ~1 mm; every proximal
segment (X1-X2) is 8-14 mm longer -- a bigger hand or MCP markers placed
nearer the palm, consistent across fingers (beta0 = -1.36).

Uncalibrated P10 model on P12's four trials: vote share >= 0.9 for 81-90% of
points (99.9% on P10), 1.3-4.5% left unlabelled, segment consistency mostly
above Vicon's and the manual files', but agreement with the (partial)
manual files 70-78% vs Vicon's 79-91%. Inconclusive on its own; the
calibrated model is the test.

**P12-calibrated model** (10k steps, 12.6 min; synthetic 99.88%). On the four
P12 trials, vs the uncalibrated P10 model: confident pieces (vote share
>= 0.9) 90.0 -> 98.9%, 88.1 -> 91.6%, 83.6 -> 97.3%, 80.6 -> 86.2%;
unlabelled 1.3-4.5% -> 0.6-1.9%; agreement with the manual files up
0.3-3.4 points. Calibration helps.

The remaining disagreement is systematic: SOMA swaps Index/Middle at levels
2 and 3 relative to Vicon *and* the manual files. Segment consistency cannot
judge a whole-chain swap; **side-to-side finger order** can (`eval_trial.py`
prints it): along the Index1->Pinky1 axis, index markers lie before middle,
etc. Verified P10 labels are in order in 98-100% of frames at every level.
P12 Trial 1 hands only, Index<Middle at level 2 / 3: Vicon 62.8 / 65.4%,
manual 67.4 / 66.9%, **SOMA 100.0 / 98.8%** (Trial2 hands only: Vicon 74%,
manual 70%, SOMA 99.9%). Under Vicon's and the manual labels index and middle
would be crossed in about a third of frames, which a hand does not do: the
P12 manual files carry an index/middle confusion at PIP and fingertip, and
"agreement with manual" penalises SOMA for not copying it.

## Next

1. Joint angles: fit SOMA-labelled Trial2 with the calibrated surface fit and
   compare with the manual-label fit (raw Vicon labels cost ~20 deg on
   mislabelled frames).
2. Generalisation: P10 Trial1 / HOI trials, then other participants (no
   manual labels -- judge by fit residuals, implausible angles, spot checks).
