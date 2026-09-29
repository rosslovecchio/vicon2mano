# Strategy: strategyGMM (spatial GMM + Viterbi)

*Split out of CLAUDE.md; see `../../CLAUDE.md` for the index.*

## Work in progress 2026-09-10 (strategyGMM — spatial-GMM + Viterbi marker labeler)

New `vicon2mano/strategies/gmm/labeler.py`, on branch `feat/gmm-labeler`. Implements
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
`scripts/gmm/validate.py` + `scripts/gmm/animate.py`,
sharing loader/fit logic via `scripts/gmm/_common.py`):**
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

`scripts/gmm/animate.py` produces a before/after side-by-side 3-D
animation (`results/gmm/validation/p7_ring1_pinky1_before_after.gif`) marking
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
  (mm², in `scripts/gmm/_common.py`). The paper's Section 3.2
  explicitly calls for tuning these against training data; guessing them
  is what produced the noisy first result.

With both fixes, the exploratory P7 episode window reports **no swap and
no flip-flopping** — matching the cascade's own verdict on the clean part
of that window, and correctly declining to guess inside the flagged run
where only 109/1635 frames have trustworthy anchors.

**Ground-truth validation (`scripts/gmm/validate_injected_swap.py`):**
the P7 episode has no clean ground truth to score against, so instead:
take the longest cascade-clean run (anchors + both targets CORRECT →
frames [1428, 3093), 1665 frames), hold a 300-frame window out of GMM
training entirely, inject a known `Ring1`↔`Pinky1` swap into the middle
100 frames, and score the recovery. Real hand motion, real Vicon noise,
real occlusion gaps; only the swap is synthetic.

    Accuracy 100.0%  precision 100.0%  recall 100.0%
    TP 100  FP 0  FN 0  TN 200
    detected transitions [1528, 1628] — the exact injected boundaries

**Animations** (`scripts/gmm/animate.py`, `render_before_after`)
render side-by-side before/after 3-D players to both `.gif` and a
self-contained `.html` scrubbing player — same `_HTML_TEMPLATE` mechanism
as `scripts/shared/animate_fit.py` (base64 JPEG frames, play/pause, scrub,
goto-frame, arrow-key/space shortcuts, no external assets). Panel titles
mark each frame explicitly: `[swap injected]` on the before panel,
`[corrected — OK]` / `[MISS]` / `[FALSE ALARM]` colour-coded on the after
panel when ground truth is available. Output in `results/gmm/validation/`
(gitignored, regenerable).

### Whole-trial run, with no cascade at all (`scripts/gmm/relabel_trial.py`)

strategyGMM derives everything from marker geometry -- no cascade, no manual
labels. Reference bone lengths come from `consensus_bone_lengths` (RANSAC
over frames: each frame's whole bone-length vector is a hypothesis scored by
how many others agree on *every* bone. Wrong configurations don't agree with
*each other*, so only correct frames form a large consensus. Needed because
`Ring1`/`Pinky1` are mislabelled in the *majority* of P7 frames, so their
marginal modes lock onto the wrong geometry -- `Palm2-Ring1` reads 34mm,
shorter than `Palm2-Pinky1` at 54mm, anatomically impossible; consensus
gives 50mm).

**There is no rigid plate** -- the palm markers are taped to skin. But the
recording is not drifting either: frame to frame the palm distances move
**0.06mm** (median, p99 1.2mm) while occasionally jumping up to **119mm**
and then *holding*. It is **piecewise stable across labelling regimes**,
max std 0.46mm inside a regime. The assumption is local rigidity within a
regime, not global rigidity.

`segment_by_jumps` -> `learn_reference_geometry` -> `solve_segment_anchors`
implement that: split at the jumps (165 regimes on P7), take the geometry
recurring across the most *independent* regimes (frame count is the wrong
criterion -- one 1930-frame corrupted regime outlasts the good ones), then
solve each regime's anchor **once by pooling over all its frames**. Pooling
is what makes it decisive: the true triple matches ~100% of a regime's
frames while the best coincidental rival reaches ~20%, a gap no single
frame can show.

**Why the single-frame search was replaced.** `locate_anchor_triangle`
matched 3 distances in one frame, which is not a unique signature among 22
markers: only **23% of located triangles were the real palm markers, 43%
contained none of them**. Tightening did not help (46% correct even at
1.5mm), nor did temporal persistence of a global reference (a spurious
triple held for >1700 consecutive frames). **Correction to earlier notes in
this file: the "coverage 20% -> 72.8% -> 97.3%" figures were inflated --
most of that coverage was coincidental matches, which is also why those runs
proposed changes on 83-94% of frames and the veto rejected ~28k of them.**

**Three verification bugs found by spot-checking single frames** (13410,
10057, 38315 -- worth re-reading when a verdict looks surprising):

1. *The animation title reported the proposal, not the outcome* -- vetoed
   frames read "(anchors repaired) reassigned 0" with identical before/after
   error. Now reports `(repair proposed but VETOED -- unchanged)`.
2. *The veto was circular.* An anchor repair forces `Palm1-Palm2`,
   `Palm2-Palm3`, `Palm1-Palm3` to match by construction, so scoring them
   handed every repair a free ~67mm "improvement". Frame 38315: 529->478
   ACCEPT with them, 461->476 REJECT without. The veto now excludes the
   anchor's own bones.
3. *Persistence was length-blind.* Frame 38315 anchored at "100%" over a
   **35-frame** regime and wrecked the hand; accepted regimes had median
   length 39. A *repair* now requires `min_repair_len=200` frames; keeping
   the labelled triple needs no such evidence.

Plus an absolute floor (`--max-residual-mm`, default 8): a frame still that
far from anatomically valid has not been repaired, however much it improved.

**Whole-trial result, P7/Trial1_handsonly** (47,656 frames, 2 passes, ~5min):

    anchor solved                  7,730 frames (16.2%), 1,150 repaired
    regimes refused                162 of 165
    consensus-clean (training)     3,260 frames (6.8%)
    marker-instances reassigned    3,453 (3.0% of usable slots)
    frames rejected by the veto    3,797
    non-anchor bones touching a reassignment  17.96mm -> 4.43mm (81.2% better)

High precision, low recall: it acts on ~500 of 47,656 frames, but where it
acts it takes the touched bones to 4.4mm -- near anatomically valid. The
spot-checked frames now resolve correctly: 13410 refused (no anchor
findable), 38315 refused (regime too short), 10057 anchored correctly but
left untouched by the residual floor, since repairing it still left ~19.7mm.

**The remaining bottleneck is the training set, not the anchor.** The 3,260
consensus-clean frames have **<1mm distance-std between every marker pair
including fingertips** -- they are all one near-static pose. Same dead-prior
condition measured directly for static training (99.3% of likelihood cells
clamped at the floor; correct-pair loglik -27.7 vs -4.5 for range-of-motion
training). Frame 10057 is the consequence: a correct anchor whose fingers
still do not resolve.

**Cohort context** (consensus-clean % per trial): nothing exceeds 15%.
P7 6.8/6.8/2.7/5.3, P9 7.9/0.9/3.1/0.1, P10 14.2/9.1/15.1/6.4,
P11 10.5/0.5/13.5/2.4. "Hands only" trials are 2-10x cleaner than HOI.
Occlusion is *not* the problem -- median visibility is 22/22 markers and no
frame was lost for want of markers.

**Statics do not help train the prior.** A static is one pose (0.1mm spread)
where markers actually vary by 9.1mm; and both statics checked are
themselves mislabelled (P10: `Palm2-Thumb1`=129mm; P9's palm plate sits
under `Index1`/`Middle1`/`Palm3`).

**Animation.** Plotly player in the same style as
`results/mano` -> `results/gmm/`. Rings: green
= label kept, blue = reassigned, none = no anchor that frame.

**Next steps:** give the prior real pose coverage -- a few hundred
hand-labelled frames spanning poses would break the ceiling directly. Label
where strategyGMM disagrees with the on-disk labels, starting with
P10/Trial2 Hands only (15.1%, the cleanest trial in the cohort).

## Work completed 2026-09-16 (external ground-truth prior, veto opt-out, dual-skeleton animation)

Follow-up to the "next steps" above: a manually-labelled-and-filled export
of P10/Trial2 Hands only (`Trial2_handsonly_manuallylabelled_filled.csv`,
assumed fully correct throughout) became available. `scripts/gmm/relabel_trial.py`
now has three additions, all opt-in -- the default (no flags) behaviour is
unchanged.

**`--prior-csv PATH` (`load_prior_gmms`).** Fits `fit_marker_gmms` directly
on the external labelled trial instead of this trial's own
consensus-bootstrapped frames for pass 1. Unlike the in-trial bootstrap, no
self-consistency filtering is needed -- a ground-truth trial's rigid frame
comes straight from its own labelled Palm1/2/3 (`rigid_frames`, not the
repair search in `rigid_frames_from_triangle`), and every finite frame is a
trustworthy training sample. The fitted GMMs are keyed by marker *name*,
then remapped onto the target trial's own column indices before use --
needed because the two trials' marker columns are not guaranteed to be in
the same order. Later passes (`--passes > 1`) still refit on the target
trial's own consensus-clean frames as before; only pass 1 changes.

Whole-trial result, P10/Trial2 Hands only (42,238 frames, prior trained on
all 42,238 ground-truth frames, target trial's own labels used as-is for
everything but the GMM prior):

    marker-instances reassigned        2,025 (0.7% of usable slots)
    frames rejected by the veto        748
    bone-length error, all bones       1.85mm -> 1.84mm (0.1% improved)
    bones touching a reassigned marker 10.64mm -> 6.81mm (84.7% improved)

The whole-trial error barely moves (expected -- most of the trial was
already fine), but the bones the repair actually touched improve sharply,
same signature as the in-trial-bootstrap results above.

**`--no-bone-veto`.** Skips step 3's independent bone-length check,
accepting every proposed reassignment unconditionally. Exists to see the
model's *raw* output when judging an external prior's proposals on their
own merits, separate from the veto's judgement. Confirms the veto is doing
real work here too: with it off on the same trial, 3,918 instances get
reassigned (vs. 2,025) and the touched-bone improvement drops to 31.4%
(vs. 84.7%) -- most of the veto's rejections were correct rejections.

**Dual-skeleton animation.** `build_figure` now takes both `corrected`
(`relabelled`) and `original` (raw `markers`) position arrays and renders
both skeletons overlaid, so a repair's effect on the geometry is visible
directly rather than only inferred from the status ring:
- corrected: solid, width 6, one trace per finger (`_bone_finger` groups
  bones by whichever endpoint names a finger, so e.g. the Palm2-Thumb1 base
  bone is coloured as thumb, not palm), plus the existing marker dots.
- original: dashed, width 2, same finger colours, plus a new faded
  (`opacity=0.35`) marker-dot trace at the raw positions.

Two independent Show/Hide button pairs act as checkboxes (Plotly's
`updatemenus` has no native checkbox widget), each toggling its set's
markers+bones together via `restyle`. They are stacked in their own row
above the plot (not placed side-by-side) after a first attempt with two
menus at the same height and long labels turned out to silently overlap and
steal each other's clicks -- worth remembering if a future button/menu
addition here "does nothing" when clicked: check for menu-region overlap
before assuming the restyle wiring is wrong.

Marker colour/label text needed no change -- a column's colour was already
its *corrected* identity: a repair only ever swaps which raw column's
*position* fills a given identity-slot, never the slot-to-identity mapping
itself, so `labels[m]`'s colour has always meant "this is column m's
(corrected) identity," not "this is what column m was originally labelled."

`.vscode/launch.json` gained two configs: "GMM (relabel_trial, external
prior)" (P10/Trial2, `--prior-csv` pointed at the manually-labelled export,
output to `results/gmm/external_prior`) and the same with `--no-bone-veto`
added (output to `results/gmm/external_prior_no_veto`).
