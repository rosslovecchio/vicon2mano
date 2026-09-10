# vicon2mano — Claude session context

Fit MANO hand model parameters to Vicon 3D optical marker data.

## Environment

- Python env: `.viconvenv/` — use `.viconvenv/bin/python` and `.viconvenv/bin/pytest`
- Install: `pip install -e ".[dev,train]"` (train extras add h5py for dataset work)
- MANO weights: expected at `../clean_kinematics/mano_v1_2/models/` (sibling repo); override with `--mano-dir`

## Running tests

```bash
.viconvenv/bin/pytest tests/ -v
```

All 75 tests should pass. The three test files added in the deep-labeler work:
- `tests/test_correspondence_get_assignment.py` — `get_assignment` + `_majority_vote`
- `tests/test_deep_labeler.py` — voxelisation, U-Net forward pass, `DeepLabeler`
- `tests/test_synth_data.py` — augmentation, heatmap targets, `_SynthH5Dataset`

## Architecture

```
vicon2mano/
  cli.py              Entry point: parse args, load labeler, call MANOFitter
  fitter.py           MANOFitter: two-stage MANO optimisation
  correspondence.py   Marker↔joint assignment (get_assignment unified API)
  deep_labeler.py     CNN-based labeler (3-D U-Net, VOXEL_RES=64)
  synth_data.py       Synthetic data generation + HDF5 dataset
  real_data.py        Real Vicon data utilities
  loader.py           .c3d / CSV loaders
  export.py           save_npz / save_obj

scripts/
  train_labeler.py    CLI: generate synthetic data + train MarkerLabeler
  eval_labeler.py     CLI: evaluate labeler on held-out data
```

## Key constants (deep_labeler.py)

| Constant | Value | Meaning |
|---|---|---|
| `VOXEL_RES` | 64 | Grid side length (64³ voxels) |
| `SPATIAL_HALF` | 0.22 m | Grid half-extent |
| `GAUSSIAN_SIGMA_VOXELS` | 1.5 | Blob std for heatmap targets |
| `CONF_THRESHOLD` | 0.10 | Min sigmoid confidence to accept a match |

**Important:** tests must import `VOXEL_RES` from `vicon2mano.deep_labeler` rather than hardcoding the grid size.

## Assignment pipeline (correspondence.py — `get_assignment`)

Priority order:
1. **Label-seed heuristic** — if ≥15 Vicon labels are recognised. Labels are
   normalised first (lowercased, "left"/"right" and punctuation stripped, so
   `Index_Left1` matches the `index1` hint). Pass `side=` to exclude
   opposite-hand labels in two-hand exports.
2. **DeepLabeler** (`labeler.label_sequence`) — CNN heatmap + linear assignment
3. **Hungarian on mean pose** — always-available fallback (NaN-gap safe)

`_majority_vote` collapses (T, 21) per-frame assignments to (21,) via `np.bincount` mode; out-of-bounds indices and -1 are ignored.

**Warning:** the checked-in `vicon2mano/weights/deep_labeler.pt` is a
smoke-test checkpoint (trained with `max_samples=500` on one real recording);
its predictions are unreliable. The label-seed path is what makes real
recordings work. Retrain with the full synthetic pipeline before trusting the CNN.

## smplx MANO conventions (fitter.py)

smplx returns **16 joints** in MANO-native order (wrist, index, middle, pinky,
ring, thumb); fingertips are mesh vertices (ids 744, 320, 443, 554, 671).
`mano_output_to_joints21()` converts to the repo's 21-joint order — always use
it, never `out.joints[:, :21]`. The fitter optimises per-frame `transl`
(Vicon world coords vs MANO origin) and initialises global orientation with
per-frame Kabsch alignment. `FitResult` and the output `.npz` include `transl`.
Expected fit residual on real data: ~15–20 mm mean (markers sit on the skin,
not at joint centres).

## CLI

```bash
# Basic fit
vicon2mano recording.c3d output.npz --side right

# With trained CNN weights
vicon2mano recording.c3d output.npz --side right \
    --labeler-weights vicon2mano/weights/deep_labeler.pt

# Export meshes too
vicon2mano recording.c3d output.npz --side right --obj-dir meshes/
```

## Training the deep labeler

```bash
# Step 1: generate synthetic frames
python scripts/train_labeler.py generate \
    --mano-dir ../clean_kinematics/mano_v1_2/models \
    --n-samples 500000 \
    --out data/synth_labeler.h5

# Step 2: train
python scripts/train_labeler.py train \
    --data data/synth_labeler.h5 \
    --out-weights vicon2mano/weights/deep_labeler.pt \
    --epochs 20 --batch-size 2 --lr 3e-4
```

Weights are saved to `vicon2mano/weights/deep_labeler.pt` as `{"model": state_dict}`.

## Work completed in prior sessions

- `deep_labeler.py`: 3-D U-Net (`MarkerLabeler`), `voxelise_with_centroid`, `assign_from_heatmaps`, `DeepLabeler` inference wrapper
- `synth_data.py`: `augment_markers`, `build_target_heatmaps`, `_SynthH5Dataset`, `generate_dataset`
- `correspondence.py`: added `get_assignment` + `_majority_vote` (refactored fitter to use unified API)
- `fitter.py`: `MANOFitter.__init__` now accepts `labeler=` kwarg
- `cli.py`: added `--labeler-weights` flag with graceful fallback
- `scripts/train_labeler.py`, `scripts/eval_labeler.py`: full train/eval pipelines

## Work completed 2026-06-12 (pipeline repair)

- `loader.py`: `load_csv` auto-detects the single-header wide Nexus layout
  (`Marker_X,_Y,_Z`); blank cells → NaN; clear error for unknown layouts
- `fitter.py`: fixed stage-0 crash (45-dim pose fed to a PCA-6 model); fixed
  `_forward_np` final-minibatch betas expansion; added `mano_output_to_joints21`
  (16-joint reorder + fingertip vertices); added per-frame `transl` parameter
  and Kabsch init (`_init_global_pose`); `FitResult.transl`
- `correspondence.py`: label normalisation (`_normalise_label`), side filtering,
  label-seed now outranks the DeepLabeler, NaN-safe Hungarian/`nanmean`
- `deep_labeler.py`: inference skips non-finite markers, indices remapped
- `synth_data.py`: `generate_dataset` uses `mano_output_to_joints21`
- `export.py`: `.npz` / `to_dict` include `transl`
- Verified end-to-end on `data/Pxh8` (left hand): 16/21 joints matched by
  label, ~17 mm mean joint-to-marker residual
- All tests passing (75/75)

## Work completed 2026-06-12 (articulation fix)

The fit used to track global motion but kept the hand permanently extended
(flat mean pose). Three compounding causes, all in `fitter.py`:

1. **Mis-scaled pose prior (root cause).** The joint loss is squared metres
   (~6e-4 at a good fit) while grasps need PCA coefficients ±3, so with
   `w_pose=1e-3` the flat hand was the *global optimum* of the objective.
   Rescaled `w_pose` 1e-3→1e-5, `w_temp` 5e-2→5e-3. Keep this unit mismatch
   in mind when adding loss terms.
2. **Optimizer step budget.** Adam moves each param ≤ lr per step; one small
   lr (3e-3 × 200 iters) could never reach flexed pose space. Now per-group
   rates (`lr_pose=5e-2`, `lr_orient=1e-2`, transl `lr=3e-3`), cosine decay,
   `n_iters_pose=300`.
3. **Kabsch init on flexed frames.** Aligning the flat template to *all*
   markers tilted the palm toward curled fingers. `_init_global_pose` now
   anchors only to palm-rigid joints (wrist + 4 finger MCPs).

Result on full `data/Pxh8` recording: 18.7 mm mean residual, fitted
articulation correlates 0.978 with the raw marker flexion signal.

`scripts/animate_fit.py` renders a side-by-side raw-vs-fitted animation:
timelapse mode (default) or real-time clip mode (`--speed 1.5 --rate 100`,
auto-picks the most dynamic window; `.mp4` needs imageio-ffmpeg, `.gif` works
everywhere). The capture rate of the Pxh8 CSV is *assumed* 100 Hz (not
recorded anywhere) — `--rate` overrides.

## Work completed 2026-06-12 (both hands)

- Right hand fitted over the full recording with the fixed objective:
  `data/Pxh8/Hands_only_Left/mano_fit_right.npz` (18.0 mm mean residual;
  16/21 joints label-matched, all to `*_Right*` markers).
- `scripts/animate_fit.py` accepts multiple `--npz` files (one per hand) and
  draws every skeleton in the "after" panel; the camera follows the combined
  joints with smoothed per-axis adaptive zoom (`set_box_aspect`), and
  `.html` output writes a self-contained scrubbing player for phones that
  can't render GIF/mp4.

## Work completed 2026-06-12 (jitter / robustness fix)

Finger-angle analysis exposed 712 frames (124 short episodes) where the
fitted left wrist teleported up to 179 mm/frame while markers moved <1.5 mm —
per-frame solutions falling into wrong basins (axis-angle ±π wraparound +
weak temporal coupling after the articulation fix). Three-layer fix in
`fitter.py`, none of which suppresses real gestures:

1. `_init_global_pose` unwraps the Kabsch rotvec sequence (picks the 2π-
   equivalent representative closest to the previous frame).
2. Joint-space **acceleration** penalty `w_accel=2.0` in the pose stage:
   gauge-invariant, wraparound-safe, constant velocity costs zero. Don't
   raise it past ~5: at 20 it visibly damped fast flexions (articulation
   corr 0.98→0.84).
3. `_refine_outliers` rescue stage: frames whose mean joint acceleration
   exceeds `outlier_accel_m` (5 mm/frame²) are re-initialised by slerp/lerp
   from clean flanking frames and locally re-optimised; accepted only if the
   marker residual doesn't degrade.

Validation: spike windows 25200–25800 / 27300–28100 went from 33/46 spikes
to 0 (max step 3 mm); articulation window 18165–21165 keeps corr 0.969 and
full 104–146 mm dip-wrist range; residuals unchanged or better.

`scripts/export_finger_angles.py` exports per-frame MCP/PIP/DIP flexion
angles (degrees, bone-to-bone, 0° = straight) for any number of fitted
hands to CSV.

## Work completed 2026-09-04 through 2026-09-07 (marker-quality cascade for raw Vicon exports)

New `scripts/label_marker_quality.py` — separate from the MANO-fitting
pipeline above. It labels every marker in every frame of a *raw* Vicon CSV
as `correct` / `incorrect` / `missing`, so bad frames/markers can be
screened out before fitting. `visualize_data.ipynb` / `visualize_data.py`
drive it interactively: `manual_frames.csv`'s own (participant, canonical
trial, session label) rows are now authoritative for which CSV belongs to
which trial slot (filename keyword-guessing is only a fallback for slots
it doesn't cover), and the notebook animates every trial in `results` in
one pass, plots % correct over time, and steps through the reasoning for
one specific frame.

**How the check works, in plain terms** — a chain of trust, most-rigid
part of the hand first, each step only trusting what the step before it
already confirmed:

1. **Forearm gate.** These markers sit on skin, not a rigid plate — their
   mutual distances legitimately drift several mm over a long recording
   as the forearm rotates and the muscle underneath moves. So instead of
   comparing every frame to a handful of spot-checked reference frames
   (indistinguishable from a real error at that drift scale), each
   pairwise distance is compared to its own *local* median over a
   nearby-frames rolling window — slow drift is absorbed as the new
   baseline, only an abrupt deviation from a marker's own recent
   neighbourhood gets flagged. That local baseline is itself sanity-capped
   against the original fixed reference, since a genuine tracking failure
   that lasts longer than about half the window otherwise looks just like
   a new stable pose from inside it (seen on real data: a plate distance
   jumped ~55mm→~150mm in one frame and held there for hundreds of frames
   after — the local median duly "learned" 150mm as normal until the cap
   was added). If 2 or more of the 4 markers look wrong, nothing else this
   frame can be trusted either (see "veto" below).
2. **Palm gate.** Same rigidity idea for the Palm-plate markers, plus each
   one's distance back to the forearm's center — and `Thumb1` is folded in
   as a de facto 4th member of this group (it sits close enough to anchor
   off the palm the same way, and the extra vote means one bad/occluded
   Palm marker doesn't as easily starve the whole gate). Needs a majority
   agreeing on presence, plate-rigidity, and forearm distance, or the
   frame's palm/fingers are all thrown out.
3. **Fingers**, one joint at a time out from the palm. Each joint is
   checked against the one before it; a bad link poisons everything
   further out on that same finger. `Thumb1` itself is resolved as part
   of the palm gate above, not re-derived here.
4. **Reference frames are trusted absolutely.** A handful of frames get
   manually eyeballed as "definitely good" ahead of time
   (`manual_frames.csv`) and are pinned as correct no matter what the
   automatic checks say about them.
5. **Stickiness — but only for a marker that was never actually caught red-
   handed.** A marker that barely moved from an already-confirmed-correct
   neighbouring frame gets to stay correct too, *unless* some check
   actually measured a real deviation for it (as opposed to just lacking
   the data to confirm it either way). This distinction matters: a
   persistent label swap between two fingers can hold both physically
   still for hundreds of frames afterward, each individual step under
   tolerance — a marker that was genuinely caught being wrong doesn't get
   a pass just because it then stopped moving. (Forearm pairwise failures
   are a special case: a pair can only implicate two markers together,
   never say which one moved, so only a marker implicated in *every one*
   of its pairs — the actual culprit's signature — counts as "caught";
   its innocent, partly-implicated neighbours stay eligible for the
   stickiness pass, since movement is the only way to tell them apart
   there.)

**Tuned deliberately toward "flag it" over "wave it through"** — a good
frame wrongly marked bad just gets thrown away (annoying but safe); a bad
frame wrongly marked good silently corrupts a downstream MANO fit. Checked
the anchor-check margin distribution across 4 real trials before settling
on current tolerances: the median failure is 20-30mm over tolerance
(overwhelmingly real errors), with only a small (~3-7%) slice of narrow,
sub-1mm misses — tolerances were nudged just enough to recover that slice
without touching the bulk of genuine failures.

**Real bugs found and fixed by using this tool on real data:**
- `manual_frames.csv` frame numbers are the ones Vicon Nexus shows on
  screen, which count from 1 — but the loaded data array counts from 0.
  Every reference frame was silently off by one, quietly using the wrong
  frame's data as "ground truth" everywhere. Fixed in
  `load_ref_ranges_csv`/`parse_frame_spec`.
- A same-plate marker swap (e.g. `Palm2` and `Palm3` physically trading
  labels) is invisible to a pure rigidity check — the plate's own
  distances look fine either way, since swapping two labels on a rigid
  body doesn't change the distances between them. Only the asymmetric
  "distance back to the forearm" check can catch it, so that check now
  breaks ties between two rigidity-linked markers by asking which one's
  forearm-distance is actually off, rather than blaming both by default.
- The Forearm markers aren't on a rigid plate at all — they're on skin,
  which drove the local-reference redesign in stage 1 above.
- Stickiness comparing only to the adjacent frame could resurrect
  "correct" status across an entire persistent label swap once both
  swapped markers stopped moving — fixed by the verified/unverified
  distinction in stage 5 above.

**Known open limitation (not yet fixed):** two anatomically adjacent
fingers whose base markers sit at a similar distance from the palm can
swap labels without tripping any distance-based check at all — confirmed
on real data (`Middle1`/`Ring1`, `Middle2`/`Index2`) where the genuine,
non-sticky checks simply never notice, with or without stickiness, because
nothing in the cascade compares fingers to *each other*. A prototype
projected-left-to-right-finger-order check (using the forearm/palm plane
to project each finger's base marker onto a "thumb-to-pinky" axis, and
flagging when the calibrated order is violated) does catch this signature
cleanly, but the naive version flags ~77% of frames (real finger
spreading/opposition also changes projected order) and needs real
refinement — a margin instead of any sign-flip, requiring persistence over
several frames, and/or corroborating it with the existing distance checks
— before it's usable.

`label_marker_quality.debug_frame_cascade(markers, labels, ref_frames, t)`
prints the full step-by-step reasoning for one frame/timestep — which
distances it checked (including the local vs. global forearm reference),
what they were compared against, and exactly where in the chain a marker
was accepted or thrown out. Reach for it whenever a frame's verdict looks
surprising.

## Work in progress 2026-09-10 (strategyGMM — spatial-GMM + Viterbi marker labeler)

New `vicon2mano/gmm_labeler.py`, on branch `feat/gmm-labeler`. Implements
Alexanderson, O'Sullivan & Beskow, "Real-time labeling of non-rigid motion
capture marker sets" (Computers & Graphics, 2017) as a **model-free**
alternative to `mano_relabel.py`'s MANO-based relabelling — no kinematic
fit involved, just per-marker spatial priors + temporal continuity. Aimed
at the cascade's documented "known open limitation": adjacent-finger label
swaps (`Middle1`/`Ring1`, `Middle2`/`Index2`) that no distance check catches
because nothing in the cascade compares fingers to each other.

**Pieces** (all hand-rolled, no new dependency — matches this repo's
existing preference, see `correspondence._cluster_two`):
- `rigid_frames`/`to_local`/`to_world`: per-frame rigid coordinate frame
  from 3 anchor markers (modelled on `mano_relabel._palm_frames`, built
  directly from raw marker columns — no MANO fit needed).
- `fit_gmm_diag`/`fit_marker_gmms`: diagonal-covariance GMM via EM, one
  per marker, trained on trusted/reference frames only.
- `top_n_assignments`: k-best linear assignment (Murty's algorithm
  reduction to repeated `scipy.optimize.linear_sum_assignment` calls).
- `MarkerKalman`/`viterbi_select`: constant-velocity Kalman filter bank +
  Viterbi selection of the most probable hypothesis sequence through time
  (paper's Section 3.2 algorithm).
- `relabel_sequence`/`GMMLabeler`: sequence-level driver, `{slot: source}`
  output compatible with `mano_relabel.apply_relabel`.
- `mask_untrustworthy_frames` / `anchor_valid=` (threaded through
  `GMMLabeler.fit`/`.relabel` and `relabel_sequence`): NaNs out every
  marker's local coordinate on frames where the anchor markers themselves
  aren't trustworthy — see real-data finding below for why this exists.

**Scope note vs. the original plan:** `get_assignment`'s `labeler=` slot
(`correspondence.py`) assumes a fully order-agnostic raw point cloud (like
`DeepLabeler`, which voxelises everything with no anchor dependency).
strategyGMM needs 3 already-trustworthy anchor markers every frame, so it
doesn't fit that contract — it's a targeted repair driver (parallel to
`relabel_with_mano.py`), not a `get_assignment` strategy.

**Tests:** `tests/test_gmm_labeler.py`, 17 tests — rigid-frame round-trip
and rigid-motion invariance, GMM EM recovery on synthetic clusters, top-N
assignment validated against brute-force enumeration, Kalman convergence,
a synthetic identity-swap Viterbi must resolve via temporal continuity
alone, and an end-to-end synthetic two-finger recording with an injected
column swap (recovered correctly, zero false positives outside the swap
window). Full suite: 105 passed.

**Real-data validation (P7/Trial1_handsonly, `Ring1`↔`Pinky1`,
`scripts/validate_gmm_labeler.py` + `scripts/animate_gmm_relabel.py`,
sharing loader/fit logic via `scripts/_gmm_validation_common.py`):**
started from the swap `relabel_with_mano.py` already documents fixing at
frame 42789. Found two real, useful things before getting a clean result:

1. Frame 42789 sits inside a much longer cascade-flagged run —
   `[41195, 42829]`, 1635 frames — and at its start the raw data shows a
   **rolling mislabelling cascade across `Palm1`/`Palm2`/`Thumb1`/`Ring1`/
   `Pinky1` simultaneously**, not a clean pairwise swap. The anchor
   markers (`Palm2`/`Palm3`/`Thumb1`) were themselves corrupted for most
   of that run (1558/1635 frames), which silently wrecked every other
   marker's local coordinate — exactly the failure `mask_untrustworthy_frames`
   now guards against, gating on the cascade's own CORRECT verdict for
   the anchors.
2. Even after anchor-gating, `Palm2`/`Palm3`/`Thumb1` turned out to be a
   poor anchor choice for this trial specifically: the cascade calls them
   untrustworthy on ~83% of the *entire* recording (`Thumb1` flagged
   often), leaving little valid data either to train on or to evaluate
   against, and the placeholder Kalman noise parameters
   (`process_var=25`, `obs_var=100`, unmeasured mm² guesses) produced a
   noisy, flip-flopping verdict rather than a clean transition.

`scripts/animate_gmm_relabel.py` produces a before/after side-by-side 3-D
animation (`results/gmm_labeler/p7_ring1_pinky1_before_after.gif`) marking
swapped/corrected frames explicitly in the title of each panel — confirms
the mechanics (relabelling, colour-coding, anchor-trust flagging) work
end-to-end on real data, independent of whether this particular trial's
verdict is trustworthy yet.

**Both of those were then fixed:**

- **Anchors → `Palm1`/`Palm2`/`Palm3`.** The original `Palm2`/`Palm3`/
  `Thumb1` triple inherited the cascade's own de-facto palm-gate set, but
  the cascade distrusts `Thumb1` on ~78% of this recording, so joint
  anchor validity was ~12% of frames vs 21.4% for the plain palm plate.
  Mixing in a `Forearm` marker scores higher still (~59%) but is
  physically wrong — the forearm is a different, non-rigidly-attached
  segment, so the "rigid" frame would articulate at the wrist.
- **Kalman noise calibrated against real jitter.** `Ring1`/`Pinky1` move
  **~0.26 mm/frame at the median, ~0.9 mm at p90** in the palm-local
  frame. The first draft's `process_var=25` / `obs_var=100` implied
  5–10 mm/frame — 20–40× too loose, so the transition term barely
  penalised an incorrect swap and the spatial GMM term flip-flopped
  unopposed. Now `DEFAULT_PROCESS_VAR=0.5`, `DEFAULT_OBS_VAR=1.0`
  (mm², in `scripts/_gmm_validation_common.py`). The paper's Section 3.2
  explicitly calls for tuning these against training data; guessing them
  is what produced the noisy first result.

With both fixes, the exploratory P7 episode window reports **no swap and
no flip-flopping** — matching the cascade's own verdict on the clean part
of that window, and correctly declining to guess inside the flagged run
where only 109/1635 frames have trustworthy anchors.

**Ground-truth validation (`scripts/validate_gmm_injected_swap.py`):**
the P7 episode has no clean ground truth to score against, so instead:
take the longest cascade-clean run (anchors + both targets CORRECT →
frames [1428, 3093), 1665 frames), hold a 300-frame window out of GMM
training entirely, inject a known `Ring1`↔`Pinky1` swap into the middle
100 frames, and score the recovery. Real hand motion, real Vicon noise,
real occlusion gaps; only the swap is synthetic.

    Accuracy 100.0%  precision 100.0%  recall 100.0%
    TP 100  FP 0  FN 0  TN 200
    detected transitions [1528, 1628] — the exact injected boundaries

**Animations** (`scripts/animate_gmm_relabel.py`, `render_before_after`)
render side-by-side before/after 3-D players to both `.gif` and a
self-contained `.html` scrubbing player — same `_HTML_TEMPLATE` mechanism
as `scripts/animate_fit.py` (base64 JPEG frames, play/pause, scrub,
goto-frame, arrow-key/space shortcuts, no external assets). Panel titles
mark each frame explicitly: `[swap injected]` on the before panel,
`[corrected — OK]` / `[MISS]` / `[FALSE ALARM]` colour-coded on the after
panel when ground truth is available. Output in `results/gmm_labeler/`
(gitignored, regenerable).

### Whole-trial run, with no cascade at all (`scripts/relabel_trial_gmm.py`)

strategyGMM no longer consults the quality cascade for anything. Every
input it needs is derived from the marker geometry itself, which matters
because the cascade is deliberately tuned to over-flag (see its section
above) and that made it a poor gate.

**Anchor plate: learn it, then find it.** `learn_anchor_triangle` takes
the *modal* pairwise distances of `Palm1`/`Palm2`/`Palm3` (the rigid
plate's true geometry is the value that repeats; the median lands between
the right and wrong configurations and matches neither).
`locate_anchor_triangle` then searches each frame's whole point cloud for
a triple matching that triangle — so a frame whose palm labels were
swapped still yields a usable frame, with the anchors *repaired* rather
than merely rejected. Coverage on P7/Trial1_handsonly: **20% → 72.8% of
frames**, of which 25,447 needed repaired anchors. The modal triangle
(32.57/40.36/24.38 mm) was later confirmed independently by consensus.

**Reference bone lengths need consensus, not marginal modes.** The first
whole-trial run had *every* anchor-valid frame (34,688/34,688) proposing a
reassignment — the signature of a model trained on bad data. Cause: the
per-bone modal reference is only valid where that bone is correct in most
frames, and on this trial `Ring1`/`Pinky1` are mislabelled in the
*majority* of frames, so their modes locked onto the wrong configuration
(`Palm2-Ring1` = 34 mm, shorter than `Palm2-Pinky1` = 54 mm, which is
anatomically impossible). `consensus_bone_lengths` fixes this with RANSAC
over frames: each frame's whole bone-length vector is a hypothesis, scored
by how many other frames agree on *every* bone at once. Wrong
configurations do not agree with each other — each swap produces a
different distance vector — so only correct frames pile into one large
consensus. `Palm2-Ring1` → 50 mm, `Palm2-Middle1` 74 → 51 mm. The
inliers double as the GMM training set: geometrically verified clean, no
external labels involved.

**Model-free veto.** A frame's reassignment is applied only if it does not
worsen that frame's total bone-length error against the consensus
reference — an independent check on the model that proposed it, in the
spirit of `relabel_with_mano.py`'s accept-only-if-verification-agrees
contract. It earns its keep immediately: P7 frame 1427 proposes a
`Thumb1`↔`Thumb2` swap that would send the touched bones from 0.9 mm to
34.3 mm of error (they are adjacent joints ~34 mm apart, so swapping them
is exactly one bone-length wrong), and it is rejected.

**Whole-trial result, P7/Trial1_handsonly** (47,656 frames, 2 passes,
~20 min):

    anchor frame located          34,688 frames (72.8%), 25,447 repaired
    consensus-clean frames         3,260 (6.8%)  <- training set
    marker-instances reassigned  137,163 (26.4% of usable slots)
    frames rejected by the veto   20,356
    bone error, all bones         14.39mm -> 12.95mm  (17.2% improved)
    bones touching a reassignment 21.00mm -> 15.40mm  (66.9% improved)

Read that honestly: the accepted repairs measurably improve geometry where
they act, but **only 6.8% of this trial is fully self-consistent to begin
with** — P7/Trial1_handsonly is severely mislabelled throughout, not just
around frame 42789, which matches the cascade independently rating these
markers 10–30% correct. A recording this broken cannot be fully repaired
by relabelling alone, and the veto having to reject 20k frames is the
method correctly declining to guess.

**Animation.** `scripts/relabel_trial_gmm.py` writes a Plotly player in the
same style as `results/mano_relabelling_pass1` (finger-coloured markers
with digits, skeleton lines, per-marker status rings, Play/Pause + frame
slider) to `results/gmm_relabelling/`. Rings: green = label kept, blue =
reassigned, none = no anchor frame that frame. Titles carry the per-frame
reassignment count and bone error before/after.
`scripts/animate_gmm_relabel.py` covers the smaller before/after
comparison case as `.gif` + a self-contained `.html` scrubbing player
(the `animate_fit.py` `_HTML_TEMPLATE` mechanism).

**Next steps (not yet done):** run the injected-swap protocol across all
finger-marker pairs and several participants for a per-pair confusion
matrix (adjacent pairs like `Middle1`/`Ring1` should be the hard ones —
the cascade's documented open limitation this strategy targets); and try a
trial that is *not* majority-corrupted, where the consensus training set
would be far larger than 6.8% of frames, to see what the method's ceiling
actually is.
