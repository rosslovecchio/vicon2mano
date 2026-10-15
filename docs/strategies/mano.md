# Strategy: MANO-based relabelling

*See `../../CLAUDE.md` for the index.*

Repairs mislabelled markers by fitting a subject-specific MANO model and
asking where each marker *should* be.

- Model: `vicon2mano/strategies/mano/relabel.py`
- Driver: `scripts/mano/relabel_trial.py` (+ `relabel_trial_loop.py`, a
  preset that defaults `--fit-on label-agnostic`)
- Tests: `tests/mano/`
- Output: `results/mano/`

Three steps, documented at length in the module docstring:

1. **Subject shape** — fit `betas` once per subject on cascade-verified
   frames, then freeze it (removes 10 free parameters from every later fit).
2. **Skin offsets** — a marker sits on skin, not at a joint centre (~15-22mm
   in this repo). Each marker's constant offset from its joint is calibrated
   in that joint's own *bone-segment* frame, which is what makes the offset
   pose-invariant; expressing it in the palm frame instead silently bakes in
   the finger's flexion at calibration time.
3. **Relabel** — fit the frozen-shape model to trusted markers only, predict
   every marker, and solve a small assignment problem between flagged
   positions and predicted ones.

**Circularity warning:** only ever fit on markers the cascade verified.
Fitting on a mislabelled marker bends the model toward the wrong data, which
then "confirms" the wrong label.

**`model_reliability` is the gate that matters.** It compares the model's
95th-percentile prediction error against the median nearest-neighbour marker
spacing. A ratio near or above 1 means the predictions carry no usable
identity information:

    P7   p95 10.2mm / spacing 25.7mm = 0.40  ->  +12.3% correctness
    P8   p95 17.5mm / spacing 22.8mm = 0.77  ->   +1.0%
    P10  p95 50.2mm / spacing 28.6mm = 1.75  ->   +1.3%   (unusable)

P10 is the cautionary case: calibrated from a static recording alone, its
model is less accurate than the distance between adjacent markers, so only
18% of its 13796 proposed reassignments verify, against 86% on P7.

**Label-agnostic mode** (`--fit-on label-agnostic`) lets a marker move even
if its own label was never flagged, iterating fit -> reassign -> refit until
the mapping stabilises, with cycle detection for mappings that oscillate.

## Tested and rejected (2026-10-15): robust loss in the surface fit

Hypothesis: a Geman-McClure loss on the marker terms of
`scripts/mano/fit_surface_markers.py` would stop a mislabelled marker pulling
the pose, so model predictions could relabel it. Measured on P10/Trial2 raw
labels against the manual labels, two 1000-frame windows, scales 10/20mm vs
plain squared loss: no improvement anywhere, slightly worse at every setting
(mislabelled-frame angle error 17.5 -> 18.1 deg on the typical window; net
relabel effect +500 -> -330 on the dense one). The option was removed.

Why it cannot work: 90% of P10/Trial2's wrong labels are a *different
finger's* marker (median 40mm away), and each finger has only 3 markers, each
setting its own joint. A wrong fingertip marker is explained exactly by
bending or spreading the finger toward it, so it leaves no residual for a
robust loss to reject. Per-frame fit residual carries no identity information
for these markers.

The same measurement showed what does carry it: split each raw trajectory at
gaps and >5mm jumps and 99.9% of the 7,833 pieces have a single true identity,
with 99% of wrong slots in pieces that are wrong end to end (973 pieces). Vicon
tracks correctly and misnames pieces after gaps, so labelling is a per-piece
problem, not per-frame.

Same session, for scale: fitting raw instead of manual labels leaves angles
within 1.2 deg where the raw labels are right, but ~20 deg off (62% of frames
>10 deg) where they are wrong. And with a clean pose, `relabel_frame` at a 30mm
gate fixes 70% of wrong labels and breaks 0.5% of right ones; with the pose
fitted on raw labels it is net harmful at every gate (circularity).
