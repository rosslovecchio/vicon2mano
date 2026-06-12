# Cohort strategy — leveraging 38 participants of heterogeneous-quality mocap

Context: 38 participants. Some recordings are well-labeled and complete;
others are degraded (mislabeled, up to ~half the markers missing). This
document is the plan for turning that heterogeneity into the paper's
contribution rather than a data-cleaning chore.

Companion docs: `PAPER.md` (paper plan), `vicon2mano/eval/fit/synth_eval_results.md`
(synthetic error budget), `CLAUDE.md` (pipeline internals).

---

## Core idea

**Use the well-labeled recordings to teach the model how to handle the
degraded ones.** This is a semi-supervised framing and is more novel than
"fit MANO to markers": the clean subset is a *supervision source* for
labeling, shape, pose prior, and the marker forward model, all of which are
then deployed to rescue the disaster recordings. Validation is a
marker-dropout robustness curve measured across the full cohort — no
goniometer required, because the full-marker fits are the reference.

The degraded recordings are the **headline experiment**: model-based fitting
degrades gracefully under marker loss where classical IK / marker-triad
angle methods fail outright. The worst recordings demonstrate why the method
is worth publishing.

## Leverage points (ordered by impact)

### 1. Learn a data-driven pose prior — attacks the dominant error
The synthetic budget shows ~2.5° comes from real poses lying outside MANO's
truncated PCA subspace, and raising PCA order does not fix it. Fit a prior
(GMM / low-dim manifold / small temporal VAE) on the hundreds of thousands
of *real* fitted poses from the clean recordings, and use it in place of the
generic PCA penalty. Double benefit: shrinks the intrinsic 2.5°, and is
exactly the regularizer that rescues sparse-marker fits (pulls solutions
toward poses real hands adopt instead of the flat mean). Highest leverage.

### 2. Train the labeler on real labels — attacks correspondence at scale
The label-seed heuristic needs ≥15 recognized labels and will fail on the
disasters; the current deep labeler is synthetic-only and a smoke-test. But
a well-labeled recording is a *free* (marker cloud → label) training pair —
the labels are the supervision. Train/fine-tune the labeler on the clean
subset's real marker geometry, then deploy on recordings with missing/wrong
labels. The clean data teaches the model to label the broken data. N=38
gives cross-subject generalization for free.

### 3. Per-subject shape transfer — constrains sparse fits
Fit `betas` once on each participant's clean recording (well-determined
there), then freeze it for that same participant's degraded recordings —
removing 10 free parameters when markers are sparse. Across the cohort this
also yields a real distribution of 38 hand shapes (tighter shape prior for
all; a small anthropometric result).

### 4. Calibrate the marker-to-joint offset model
Estimate each marker's consistent offset from its underlying joint (the
"skin offset", assumed 8 mm in synthetic) from clean recordings, and fold
the per-marker offset vectors into the forward model. Removes a systematic
bias on all recordings; well-precedented (MoSh latent marker placement).

### 5. Validation + clinical findings (already planned)
- **Marker-dropout robustness curve**: synthetically ablate markers from
  clean recordings (25/50/75 %, realistic dropout patterns), measure angle
  degradation vs the full-marker reference. Real-data accuracy, no goniometer.
- **Minimal-marker-set analysis**: which markers can be lost before angles
  degrade — generalizes the synthetic fingertip-marker finding into an
  actionable, real-data protocol recommendation.

## How it fits together

Clean recordings → {pose prior (1), trained labeler (2), per-subject shapes
(3), offset model (4)} → deployed to make the degraded recordings fittable →
robustness/minimal-set validation across 38 subjects (5).

Paper becomes: *"Bootstrapping robust finger kinematics from degraded
clinical optical mocap using a well-labeled subset."* A method contribution,
not just an application of MANO.

## Recommended starting order

1. **Triage/QA scan** (built — `scripts/triage_dataset.py`): characterize the
   whole dataset, produce the clean/degraded split and the (marker→label)
   export that feeds #2. Cheap, no heavy compute, decides the reference set.
2. **#2 trained labeler** and **#3 shape transfer** — highest value, lowest
   risk; #2 is the prerequisite for fitting the disasters at all.
3. **#1 learned pose prior** — biggest accuracy lever, more research risk.
4. **#5 validation** once the pipeline runs on the cohort.

## Open decisions / dependencies

- Data not yet on this machine; format assumed to match the wide-Nexus CSV
  of `data/Pxh8` (triage adapts if labels/layout differ — it audits the label
  vocabulary across all files).
- Compute: ~25–40 min/fit on the 4 GB GPU × (38 × recordings × hands) is
  potentially days. Triage gates out unsalvageable files before fitting;
  batch scheduling likely needed.
- Labeler decision (mature it via #2, or drop it and rely on label-seed +
  Hungarian) should be settled early — see the warning in `CLAUDE.md`.
