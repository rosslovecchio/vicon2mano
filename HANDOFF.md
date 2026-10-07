# Handoff — marker→joint mapping bug, joint-angle export, GMM ground-truth eval

**Date:** 2026-10-07
**Scope:** Delivered joint angles for 4 trials, then found and fixed a dataset-wide
marker→joint mapping bug in `core/correspondence.py` that invalidates every MANO fit
made before this session. Also: first evaluation of strategyGMM against real full-trial
human ground truth, two corrections to claims previously recorded in `CLAUDE.md`, and
GPU (CUDA) enabled for the fitter.

---

## New / modified scripts

### 1. `vicon2mano/core/correspondence.py` (MODIFIED — the important one)
`_LABEL_HINTS` assumed the `data_June25` 3-marker-per-finger protocol is
`mcp/pip/dip`. It is not: marker 1 is at the MCP, marker 2 sits on the middle
phalanx (~60% along the finger), and **marker 3 is at the FINGERTIP**.

| Finger | old: marker3→DIP | new: marker3→TIP |
|---|---:|---:|
| Index  | +20.4mm error | −5.2mm |
| Middle | +30.3mm error | +4.0mm |
| Pinky  | +22.8mm error | +2.8mm |

Verified on 6 participants (P3/P5/P7/P10/P13/P15) — the tip reading wins in all
18 finger/subject combinations.

Key implementation notes:
- Marker 2 is mapped to the **DIP** as nearest joint; it is really ~60% along
  (PIP is 41%, DIP 68%). The leftover is what `strategies/mano/relabel.py`'s
  per-marker offset calibration is for.
- **The thumb is deliberately NOT remapped.** The same test is ambiguous there:
  `Thumb1→Thumb3` spans 74–87mm against MANO's 57.9mm (j1→j3) and 93.3mm
  (j1→tip), and which fits better flips between participants. Thumb anatomy also
  differs (CMC/MCP/IP, two phalanges).
- The Nexus 4-marker convention (`RIDX1-4`) is untouched and still maps
  1/2/3/4 = mcp/pip/dip/tip. The two protocols use distinct strings under
  `_normalise_label` — `"index2"` does not contain `"ind2"` — so they don't collide.
- 3 regression tests added in `tests/core/test_correspondence.py` (263 total).

### 2. `vicon2mano/core/fitter.py` (MODIFIED)
`_optimise_pose` gained a `target_override` kwarg so a caller can substitute
offset-corrected targets without duplicating the optimisation loop (temporal
smoothness, pose prior, accel penalty, scheduler stay shared). Backward
compatible — default behaviour unchanged.

### 3. `scripts/mano/export_joint_angles.py` (NEW)
Decodes a fit's PCA `hand_pose` into per-finger joint angles. `--plot` writes one
PNG per finger.

| Output | Meaning |
|---|---|
| `<out>.csv` | `frame`, `time_s`, then 15 `<finger>_<mcp\|pip\|dip>_deg` columns |
| `<out>_<finger>.png` | 3 stacked subplots (MCP/PIP/DIP) over time |

- "Joint angle" = rotation **magnitude** in degrees (`‖axis-angle‖`), not a
  flexion/abduction decomposition — MANO's 3 axis-angle components per joint are
  not individually labelled anatomically.
- `--fps 200` is real, read from the CSV's `Trajectories` header block, not assumed.

### 4. `scripts/mano/animate_mesh.py` (NEW)
MANO mesh surface + Vicon marker overlay as a self-contained HTML player.
Frames are **pre-baked to JPEG offline** and driven by a fixed-rate JS timer, so
playback speed is decoupled from render cost. `Inspect (rotate)` pauses and swaps
in a live Plotly Mesh3d of the current frame for free orbiting; pressing Play
exits Inspect so it never competes with the performance-sensitive path.

### 5. `scripts/mano/fit_single_frame.py` (NEW)
Fits a single frame or a short window and plots it. Built to test whether a
whole-trial fit's residual at one frame is a temporal-smoothing artifact — it is not.

### 6. `scripts/mano/refit_with_marker_offsets.py` (NEW)
Two-pass calibrate-then-refit using `strategies/mano/relabel.py`'s existing
bone-segment-frame offset calibration (reused, not reimplemented). Reports three
residuals because "lower joint-to-marker residual" stops being the right metric
once a real offset exists.

### 7. `scripts/mano/calibrate_shape_on_static.py` (NEW)
Fits `betas` on a static trial, then refits pose with them frozen. Has its own
shape-stage hyperparameters because the stock one is a no-op (see Gotchas).
`--report-only` screens a static segment-by-segment against a verified movement
trial. `--static-out` plots the fit **on the static itself**; `--out` plots a
*movement* frame fitted with those betas (different thing — don't confuse them).

### 8. `vicon2mano/strategies/gmm/labeler.py` (MODIFIED)
`theta_min` was a clamp, not a filter (CLAUDE.md open item #2). Added
`_decline_augmented_cost`: one dedicated "decline" column per marker at cost
`-theta_min`, so a marker with no plausible match opts out instead of being forced
into a tie-broken pairing. `top_n_assignments` needed no changes.

### 9. `.vscode/launch.json`, `PROGRESS.md`, `CLAUDE.md` (MODIFIED / NEW)
- launch.json: added a debug config for `export_joint_angles.py`.
- `PROGRESS.md` (NEW): deliberately minimal one-page tracker with a single open
  "next decision" checkbox. User asked for this to stay ~20 lines — keep it short.
- `CLAUDE.md`: added the 2026-10-07 session section and **corrected a prior claim**
  (see Key findings).

---

## Pipeline order (re-run from scratch)

```
1. vicon2mano.core.cli <trial.csv> results/mano/<trial>/mano_fit_right.npz --side right
2. scripts/mano/export_joint_angles.py --npz <...>.npz --out results/mano/<trial>/joint_angles --fps 200 --plot
3. scripts/mano/animate_mesh.py --npz <...>.npz --csv <trial.csv> --out .../eval/mesh_animation.html --n-out 1500 --fps 200
   (optional diagnostics)
4. scripts/mano/calibrate_shape_on_static.py --static <static.csv> --movement <trial.csv> --report-only
5. scripts/mano/refit_with_marker_offsets.py --csv <trial.csv> --start A --end B --highlight-frame F --out <...>.html
```

**Step 1 feeds 2 and 3 — the mapping fix changed step 1's output, so 2 and 3 must
be re-run for every trial.** See Gotchas for what is currently stale.

---

## Key findings

- **P10/Trial2 full-trial human ground truth** (`Trial2_handsonly_manuallylabelled_filled.csv`,
  all 42,238 frames, not a sample): 738,839 judgeable marker-frame slots,
  **97.6% of Vicon labels correct**, 17,658 wrong. Far better than the 500-frame
  stratified sample's 75.9%, which was deliberately biased toward hard cases.
- **strategyGMM vs that ground truth — first real evaluation.** Before the
  `theta_min` fix: 62 real errors fixed, 279 correct labels broken. After: **90
  fixed, 152 broken**. Both moved the right way simultaneously, but it is **still
  net harmful** (1:1.7 against) with 0.5% recall. The existing injected-swap
  protocol's 100%/100%/100% is not evidence of real-world performance.
- **Mapping fix effect on P10:** residual 14.85mm → 12.01mm (dropping the bogus
  wrist correspondence) → **9.11mm** (corrected mapping). `betas[0]` went
  −1.17 → −0.17: the model stopped distorting its shape to absorb the mismatch,
  which is the signature of a real fix rather than extra freedom soaking up error.
- **Static shape calibration is unnecessary for P10**: 12.33mm (betas=0) vs 12.35mm
  (calibrated). With the mapping corrected, this hand already matches MANO's mean
  (finger span ratios 0.98/1.06/1.04/1.04).
- **Static fit residual 5.41mm vs ~12mm on movement frames** — the remaining
  movement error is pose/soft-tissue, not shape. Further shape work has little left.
- **CORRECTION to a prior CLAUDE.md claim:** "P10's static is mislabelled,
  `Palm2-Thumb1`=129mm" is **wrong** — measured directly it is 53.85mm, and every
  segment agrees with the verified movement trial to ±4.4mm. It came from a buggy
  ad-hoc script. The **P9 half of that claim holds**: P9's *palm* markers really are
  mislabelled (`Palm1-Palm3` 28.0mm vs 47.6mm), though its finger chains are clean.
- **GPU enabled**: `.viconvenv` had `torch 2.14.0+cpu` while an RTX A2000 sat idle.
  Swapped to `torch 2.14.0+cu126` (same version, CUDA variant). 2000-frame chunk:
  28.2s, peak VRAM 0.32GB. `chunk_frames` 2000 vs 6000 made no difference (69.9 vs
  67.8s) — left at default.

---

## Things still worth doing

1. **(BLOCKING) Regenerate stale outputs.** See Gotchas — the joint angles currently
   on disk were produced with the wrong mapping.
2. **Resolve the thumb mapping.** Left unchanged because the chain test is ambiguous.
   Behavioural evidence now points at `Thumb3` also being a fingertip marker (the
   mesh thumb-tip overshoots `Thumb3` by ~13mm, and the pinch gap is mostly thumb).
   Needs a separate test before changing.
3. **Fix the `w_shape` pin** (optional). `betas` cannot move (see Gotchas). Nearly
   harmless for P10 since the mean hand fits, but will matter for subjects whose
   hands differ from MANO's mean.
4. **Fix the `Palm2`-as-wrist proxy** (optional but worth ~19% residual). MANO's
   wrist is 90mm from the index MCP; the `Palm2` marker is 59mm. Consider dropping
   the wrist correspondence entirely rather than remapping it.
5. **strategyGMM** is paused by user decision. The real bottleneck is anchor
   coverage: `44.5% anchor-valid, 0 frames with repaired anchors` on P10 — the
   55.5% of frames with no palm anchor are invisible to every downstream step.
   Bigger lever on recall than more hypothesis-scoring work.

---

## Gotchas to know about

- **STALE OUTPUTS — the main trap.** All four `results/mano/*/joint_angles.csv`
  (+ per-finger PNGs, timestamped 15:45–15:50) and all four
  `eval/mesh_animation.html` were generated **before** the mapping fix and are
  therefore wrong — **re-run pipeline steps 2 and 3 for every trial.**
  Fit state at time of writing (`ls -la results/mano/*/mano_fit_right.npz`):

  | Trial | npz | post-mapping-fix? |
  |---|---|---|
  | P10/Trial2 | 10-07 21:11 | yes |
  | P15/Trial2 | 10-07 21:28 | yes |
  | P5/Trial2  | 10-07 21:27 | yes |
  | **P3/Trial1** | **10-07 15:49** | **NO — refit still running** |

  Anything dated 10-07 15:xx or earlier predates the fix. Check timestamps
  before trusting anything under `results/mano/`.
- **`MANOFitter`'s shape stage is effectively a no-op.** `w_shape=1e-2` against a
  ~1e-3 joint loss, plus `lr=3e-3` over 100 iters (betas need ±1–3), leaves betas
  pinned at ~0 — verified: fitted betas come back `[0.005, 0, -0, ...]`. Same class
  of bug `CLAUDE.md` documents for `w_pose`, never fixed for `w_shape`. Any script
  needing real shape fitting must set its own hyperparameters.
- **A stale `.npz` fitted under the OLD mapping will silently corrupt offset
  calibration**, since it calibrates against joints that mean something different.
  `refit_with_marker_offsets.py --whole-trial-npz` is optional for this reason.
- **matplotlib `mplot3d` cannot depth-sort a mesh against scatter markers.** Markers
  geometrically in front can render hidden inside an opaque mesh. `zorder` does NOT
  fix it. `alpha<1` makes it **worse** — it also breaks sorting of the mesh's own
  overlapping triangles (visible dark hatching on the fingers). Use Plotly if correct
  depth matters.
- **MANO's mesh shading artifact is a lighting issue, not geometry.** Dark hatching
  appears in *both* matplotlib and Plotly; it vanishes with ambient-only lighting
  (`ambient=1.0, diffuse=0.15, specular=0.0`). The mesh winding is fine — it has 16
  boundary edges because it is an open surface at the wrist, not because it is
  non-manifold.
- **`--out` vs `--static-out`** in `calibrate_shape_on_static.py`: `--out` plots a
  MOVEMENT frame fitted with static-calibrated betas. Only `--static-out` plots the
  static itself. The previously written `frame31163_staticshape.html` is a movement
  frame despite the name.
- **Environment changed**: `.viconvenv` now has `torch 2.14.0+cu126`, not `+cpu`.
  Separate CUDA installs exist in conda envs `DEEPLABCUT` / `emg2pose`, but they are
  Python 3.10 / torch 2.5.1 and **cannot** be used with this Python 3.13 venv.
- **`chumpy` needs removed numpy aliases.** Every standalone script needs the
  `np.object`/`np.str`/`inspect.getargspec` shim before importing smplx (copy it
  from any script in `scripts/mano/`).
