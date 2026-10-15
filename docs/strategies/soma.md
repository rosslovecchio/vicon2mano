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

- Code: `vicon2mano/strategies/soma/` (`library.py`, `synth.py`)
- Scripts: `scripts/soma/` (`build_library.py`, `preview_synth.py`)
- Tests: `tests/soma/`
- Output: `results/soma/` (gitignored, regenerable)

Status (2026-10-15): **generator done and checked; network not built yet.**

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

## Next

1. Network: SOMA-style transformer + Sinkhorn with dustbins, ~1-2M params.
2. Evaluate with no real data in training: label raw P10/Trial2 per frame,
   vote per piece, score against the manual labels. The bar is the raw Vicon
   labels' 97.6%.
3. Only if needed: fine-tune on half of Trial2's real observations, score the
   other half.

The library is built from P10's own Trial2, so evaluating on Trial2 shares
its geometry (not its labels -- the network never sees them). P10's other
trials and other participants are the honest generalisation test.
