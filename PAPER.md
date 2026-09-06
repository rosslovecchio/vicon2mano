# Paper plan — sparse-marker MANO fitting for clinical finger kinematics

Working title: **"Recovering finger joint angles from sparse optical marker
sets via a hand-model prior: accuracy limits and protocol implications"**

Status: draft outline (2026-06-12; updated 2026-09-07 with the raw-marker
quality-screening step, see §3/§6). Numbers below are from
`vicon2mano/eval/fit/synth_eval_results.md`; real-data validation still TODO.

---

## 1. One-line thesis

A standard clinical hand marker set (palm + 3 markers/finger) is
under-determined for classical inverse kinematics, but fitting the MANO hand
prior to it recovers proximal finger angles reliably and exposes a
*quantifiable, protocol-level* limit on distal-joint accuracy.

## 2. Target venue

Primary: clinical/biomechanics methods — *Journal of Biomechanics* (methods
note), *Gait & Posture*, *IEEE TNSRE*, *Sensors*.
Secondary / fallback: tool paper at *JOSS*; or a CV hands workshop if the
learned labeler is matured.

## 3. Novelty — what we claim, and the honest delta vs prior art

Prior art (must cite, must differentiate):
- MANO (Romero 2017) — the model; built by fitting to *dense* marker scans.
- MoSh++ / AMASS (Mahmood 2019) — fits SMPL+H to marker sets at scale;
  heavyweight, body-centric.
- GRAB (Taheri 2020) — full-body+hands from mocap.
- SOMA (Ghorbani 2021) — auto-labeling sparse markers with a transformer.

Our delta (none of these is "we invented MANO fitting"):
1. **Protocol-design finding (the strongest claim):** distal (PIP/DIP) angle
   fidelity is fundamentally limited by the *missing fingertip marker*, not
   by the algorithm — adding one tip marker lifts DIP correlation ~0.68→0.91
   and cuts MAE 1.9°. Actionable, algorithm-independent capture-protocol
   recommendation.
2. **Error-budget decomposition for the *sparse clinical* regime**, including
   a useful **negative result**: raising the MANO pose-PCA order (6→12)
   barely helps (~0.5°), because the unrecovered articulation lies in
   directions 3 near-collinear markers/finger cannot observe, and extra DOF
   absorbs skin-offset bias. This pre-empts the obvious "use more components"
   objection and argues against over-parameterization. Prior work reports
   aggregate accuracy on rich marker sets; this per-source budget at 3
   markers/finger is new.
3. **A lightweight, tested, open pipeline** (Vicon CSV → MANO + per-joint
   angles) filling a practical gap MoSh++/SOMA don't serve for small labs.

What we do NOT claim: a novel optimizer, a novel network, SOTA on a public
hand benchmark.

## 4. Key findings so far (synthetic ground truth)

Full table in `vicon2mano/eval/fit/synth_eval_results.md`. Decomposition:

| Error source | Contribution to angle MAE |
|---|---|
| Marker skin offset + noise (8mm/1mm) | ~0.7° (negligible) |
| Fitter pose-PCA order (6→12) | ~0.5° (negative result — nearly flat) |
| Truth outside fit's PCA subspace | ~2.5° (intrinsic to truncated prior; not closed by higher order) |
| Sparse protocol / no fingertip marker | ~1.9°, distal joints only |

- Realistic sparse condition: **9.8° MAE**, 8.1 mm joint position error.
- Proximal (MCP) joints recovered well under all conditions (corr 0.87–0.97);
  the open problem is distal joints, and it is a *protocol* problem.
- Best controlled case (clean markers + tips, truth in subspace): **4.7°
  MAE**, corr 0.96 — the algorithm is sound; residual is protocol/model.
- PCA-12 fitter confirmed: realistic 9.8°→9.5°, clean 9.1°→8.5°. Higher
  order does **not** rescue accuracy — a reportable design curve.

Robustness (real data, `data/Pxh8`): articulation correlation 0.97 with raw
marker flexion; temporal jitter eliminated (wrist spikes >20 mm: 712 → 0)
without damping gesture — see fitter design notes in CLAUDE.md.

## 5. Proposed structure

1. **Introduction** — clinical need for finger kinematics; why sparse optical
   protocols are attractive but IK-underdetermined; the model-prior idea.
2. **Related work** — MANO, MoSh++/AMASS, SOMA, clinical hand kinematics.
3. **Method** — raw-marker quality screening (a trust-chain cascade —
   forearm rigidity/drift, palm rigidity + forearm-anchor distance, then
   finger chains — that labels every marker in every frame
   correct/incorrect/missing before any fitting happens, catching
   occlusion-fill jumps and label swaps a downstream fitter would
   otherwise silently absorb; `scripts/label_marker_quality.py`); two-stage
   MANO fit (shape, then per-frame pose+transl); correspondence
   (label-seed → Hungarian; learned labeler optional); robustness terms
   (joint-space acceleration prior; outlier rescue).
4. **Synthetic evaluation** — generator; error-budget ablation (Table in §4);
   the tip-marker experiment.
5. **Real-data results** — `data/Pxh8` two-hand fit; residuals; articulation
   fidelity; jitter before/after.
6. **Discussion** — protocol recommendation (add fingertip markers; choose
   PCA order); limitations.
7. **Conclusion.**

## 6. What's needed before submission (gap list, ordered)

1. **Real ground truth** — electrogoniometer, instrumented glove, or a
   dense-marker protocol subsampled to sparse. Without this it's synthetic-
   only and reviewers will balk for a clinical venue.
2. **Subjects / recordings** — currently N=1 (`Pxh8`). Need ≥5–10 with
   varied hand sizes and a defined task battery (flexion/extension, grasps,
   pinch). Progress: the raw-marker quality cascade (§3) now automatically
   screens per-frame correctness for the `P7` batch (4 trials); still need
   to run it over the rest of the multi-participant set and hand-verify a
   `manual_frames.csv` reference row per trial before those recordings are
   usable ground truth for fitting.
3. **Baseline comparison** — MoSh++ and/or SOMA on the same sparse input, or
   a direct-IK baseline, to show the prior earns its place.
4. **PCA-order sweep** as a reported design curve (accuracy vs n_comps vs
   overfitting to noise).
5. **Labeler decision** — either retrain the deep labeler on the full
   synthetic pipeline and report its accuracy, or drop it from the paper and
   lead with the label-seed + Hungarian correspondence that actually works
   (the checked-in checkpoint is a smoke test — see CLAUDE.md warning).
6. Reproducibility: release synthetic generator + eval scripts (already in
   `scripts/`), pin environment.

## 7. Risks / reviewer objections to pre-empt

- "Just MoSh++ for hands." → §3 delta + baseline comparison (gap #3).
- "Synthetic only." → gap #1 real ground truth.
- "N=1." → gap #2.
- "PCA truncation is a known SMPL/MANO issue." → true; we *quantify* it for
  the sparse clinical regime and turn it into a design curve, not a novelty
  claim.
