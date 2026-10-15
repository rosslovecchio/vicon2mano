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

- Code: `vicon2mano/strategies/soma/` (`library.py`, `synth.py`, `model.py`, `realdata.py`)
- Scripts: `scripts/soma/` (`build_library.py`, `preview_synth.py`, `train.py`, `eval_trial.py`)
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

## Next

1. Joint angles: fit SOMA-labelled Trial2 with the calibrated surface fit and
   compare with the manual-label fit (raw Vicon labels cost ~20 deg on
   mislabelled frames).
2. Generalisation: P10 Trial1 / HOI trials, then other participants (no
   manual labels -- judge by fit residuals, implausible angles, spot checks).
