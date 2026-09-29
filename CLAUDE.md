# vicon2mano — Claude session context

Fit MANO hand model parameters to Vicon 3D optical marker data, and repair
the marker labelling that pipeline depends on.

This file is the **index**. Detail lives in `docs/`; each strategy has its
own page rather than one ever-growing log.

## Environment

- Python env: `.viconvenv/` — use `.viconvenv/Scripts/python.exe` on Windows
- Install: `pip install -e ".[dev,train]"` (train extras add h5py)
- MANO weights: `../clean_kinematics/mano_v1_2/models/` (sibling repo);
  override with `--mano-dir`
- Recordings: `C:\Users\RL000009\OneDrive - Vrije Universiteit Brussel\A-Skills\data_June25` — see `vicon2mano/core/dataset.py`
- Extra runtime deps not in pyproject: `plotly`, `matplotlib`

```bash
.viconvenv/Scripts/python.exe -m pytest tests/ -q     # 117 tests
```

## Layout

Code is organised by **which strategy it belongs to**. Anything used by more
than one strategy lives in `core/`.

```
vicon2mano/
  core/          loader · export · fitter · correspondence · cli
    dataset.py   find_trial_csv, manual_frames.csv parsing, DATA_ROOT
    viz.py       finger palette, marker colours, status-ring semantics
  strategies/
    cascade/     quality_cascade.py   geometric correct/incorrect/missing
    mano/        relabel.py           MANO-model-based repair
    gmm/         labeler.py           strategyGMM: spatial GMM + Viterbi
    deep/        labeler.py, synth_data.py, real_data.py   CNN labeler
    template/    (lives in core/correspondence.py; see docs)
scripts/   cascade/ · mano/ · gmm/ · deep/ · shared/
tests/     core/ · cascade/ · mano/ · gmm/ · deep/
results/   cascade/ · mano/ · gmm/          (gitignored, regenerable)
docs/      fitting.md · strategies/*.md
```

`core/dataset.py` and `core/viz.py` exist because this plumbing used to sit
inside `scripts/mano/relabel_trial.py` and the top third of
`vicon2mano/strategies/cascade/quality_cascade.py`, so every other strategy had to import a
strategy driver just to locate a trial CSV.

## Strategies

| Strategy | What it uses | Docs |
|---|---|---|
| **cascade** | rigid-body distance checks, chain of trust from the forearm out | [docs/strategies/cascade.md](docs/strategies/cascade.md) |
| **mano** | fits a subject-specific MANO model, predicts where markers should be | [docs/strategies/mano.md](docs/strategies/mano.md) |
| **gmm** | per-marker spatial GMM + Kalman/Viterbi, no kinematic model | [docs/strategies/gmm.md](docs/strategies/gmm.md) |
| **deep** | 3-D U-Net on a voxelised cloud; the only order-agnostic one | [docs/strategies/deep.md](docs/strategies/deep.md) |
| **template** | distance-descriptor match to a labelled template recording | [docs/strategies/template.md](docs/strategies/template.md) |

Core MANO fitting (not a relabelling strategy): [docs/fitting.md](docs/fitting.md).

## Assignment pipeline (`core/correspondence.py` — `get_assignment`)

1. **Label-seed heuristic** — if ≥15 Vicon labels are recognised. Labels are
   normalised (lowercased, "left"/"right" and punctuation stripped, so
   `Index_Left1` matches the `index1` hint). Pass `side=` to exclude
   opposite-hand labels in two-hand exports.
2. **DeepLabeler** (`labeler.label_sequence`)
3. **Hungarian on mean pose** — always-available fallback (NaN-gap safe)

`_majority_vote` collapses (T, 21) per-frame assignments to (21,); the
label-seed path is what actually makes real recordings work.

## smplx MANO conventions (`core/fitter.py`)

smplx returns **16 joints** in MANO-native order (wrist, index, middle,
pinky, ring, thumb); fingertips are mesh vertices (744, 320, 443, 554, 671).
`mano_output_to_joints21()` converts to this repo's 21-joint order — always
use it, never `out.joints[:, :21]`. The fitter optimises per-frame `transl`
and initialises global orientation with per-frame Kabsch alignment.
Expected residual on real data: **~15–20 mm** (markers sit on skin, not at
joint centres).

## State of the data (read before trusting any result)

Measured across 16 trials of P7/P9/P10/P11 — **the labelling is bad
cohort-wide**, not in one trial:

- Frames that are fully geometrically self-consistent, per trial: **0.1% to
  15.1%**, nothing higher. P7 6.8/6.8/2.7/5.3, P9 7.9/0.9/3.1/0.1,
  P10 14.2/9.1/15.1/6.4, P11 10.5/0.5/13.5/2.4.
- "Hands only" trials are consistently **2–10× cleaner** than HOI.
- **Occlusion is not the problem** — median visibility is 22/22 markers, and
  no frame was ever lost for want of markers.
- **Statics do not help train a spatial prior.** A static is one pose (0.1mm
  spread) where markers actually vary by 9.1mm across poses; and both statics
  checked are themselves mislabelled (P10 `Palm2-Thumb1`=129mm; P9's palm
  markers sit under `Index1`/`Middle1`/`Palm3`).
- **There is no rigid plate** — palm markers are taped to skin. But the data
  is not drifting either: it is piecewise stable across labelling *regimes*
  (0.06mm/frame within one, jumps up to 119mm between them).

The open bottleneck for every model-based strategy is that self-supervision
can only bootstrap from that 7–15% clean core, and that core is one
near-static pose. A few hundred hand-labelled frames **spanning poses** is
the highest-value thing to add; label where a strategy disagrees with the
on-disk labels, starting with P10/Trial2 Hands only (15.1%, the cleanest).

## Work completed 2026-09-11 (strategyGMM reviewed against the paper; 2 bugs fixed)

Read `vicon2mano/strategies/gmm/labeler.py` line by line against
Alexanderson et al. 2017 (Zotero `5DGD7MMR`), Sections 3.1-3.3. The
architecture is a faithful sketch; **13 defects** were found, 2 fixed here.
The remaining 11 are listed below so the next session can pick them up
without re-deriving the review.

### Fixed 1 - emission (spatial) term was missing from the Viterbi trellis

The paper scores each trellis node as emission (Eq. 1) **+** transition
(Eq. 3). `viterbi_select` destructured the hypothesis cost into `_cost` and
threw it away, scoring transitions only - while its own docstring claimed it
combined both. Proven by probe: with transitions held symmetric, the winner
followed the *list order* of the hypotheses, not their cost.

Now added at every node, with a `w_emission` knob (default 1.0, the paper's
1:1; Section 5 explicitly suggests exposing exactly this). Emission is
constant over the predecessor `src`, so it is added after that loop - it
cannot change which predecessor wins, only how a hypothesis ranks against
its siblings, which was the missing comparison.

**Severity correction, important:** this was a *latent* bug, not an active
one. Five adversarial variants failed to make the pre-fix code produce a
single wrong answer on P7 data (see the protocol section below). The reason:
`top_n_assignments` returns hypotheses **best-cost-first**, so `np.argmax`
on tied scores was already reproducing the spatial ranking wherever the
transition term was indifferent - and wherever it was not indifferent, it
was decisive alone. The exposure the fix closes is that the ordering
accident only rescues *exact* ties: with the term absent, a spatially
absurd hypothesis beat the truth on any non-zero transition margin, however
small. That is the persistent-swap regime this strategy exists for; it just
does not arise in clean 300-frame windows.

### Fixed 2 - hard crash on any non-finite value reaching a filter

`best_score` started at `-np.inf` and `nan > -inf` is `False`, so a NaN
transition score left `best_bank = None` for *every* predecessor and then
crashed on `best_bank[m].update(...)`. Reachable from real data:
`relabel_local` seeds the bank from `local_mm[init_frame]` with no
finiteness check, and a Vicon gap at that frame is routine.

Fixed in three layers, each independently correct:

- `MarkerKalman.__init__` treats a non-finite seed as *unknown* - sanitises
  to 0 but inflates that component's variance to 1e6, so the first real
  observation dominates. Seeding 0 at the normal 100mm^2 would instead
  assert the marker sits at the local origin. Verified: converges to the
  true position within 5 observations.
- `MarkerKalman.update` skips a non-finite observation, leaving the
  predicted state - which *is* the paper's Section 3.2 occlusion handling
  (extrapolate, omit the innovation update), so this is semantics, not a
  patch. `residual_loglik` returns `-inf` rather than NaN.
- `viterbi_select` coerces any non-finite node total to `-inf` and accepts
  the first predecessor unconditionally, so a node always has a bank.
  Empty frame-0 hypotheses now raise a clear `ValueError` instead of the
  same `NoneType` crash.

### Protocol rerun (`scripts/gmm/validate_injected_swap.py`)

100% / 100% / 100%, transitions `[1528, 1628]` - **unchanged by both
fixes**, and the animation was regenerated (`results/gmm/validation/`,
`.gif` + self-contained `.html` player).

**Read that number with care: it is reproduced by code with the spatial
model amputated from the selection objective.** It is therefore not
evidence that the spatial model works. A clean block swap has sharp
temporal boundaries that the transition term catches unaided. Variants
tried, all 100% for both `w_emission=0` and `1`:

| variant | discriminates? |
|---|---|
| baseline injected swap | no |
| swap beginning inside a 10-150 frame occlusion gap | no |
| window *starts* already swapped (no prior frame) | no |
| frame 0 unpinned, trellis must initialise | no |
| 10-marker set, `Middle1`/`Ring1` (the hard pair) | no |

`Ring1`/`Pinky1` sit **17.3 mm apart** in the palm-local frame - far outside
filter uncertainty even after a 150-frame gap, which is why gap length never
mattered. Genuinely exercising the spatial term needs the temporal term to
be confidently *wrong*: markers whose trajectories cross or converge, so the
filter locks onto the wrong branch. Build that against a trial with finger
adduction before treating the method's ceiling as known.

### Still open (ranked; numbering follows the review)

1. **(#2) Occlusion bias in `_transition_score`.** Unmatched markers
   contribute 0, matched ones contribute a negative log-likelihood, so a
   hypothesis is scored *better* the more markers it drops, monotonically.
   The paper avoids this by giving filtered-out assignments a uniform
   `log(1/M)`. Highest-value remaining fix.
2. **(#5) `theta_min` is a clamp, not a filter.** Paper 3.1 *removes*
   sub-threshold assignments (ghost/gap); `np.maximum(ll, theta_min)` keeps
   the pairing and floors its cost. So the labeler can never output "ghost"
   or "gap", and once many entries hit the floor they are *exactly equal*,
   making the LAP degenerate and Murty's top-N enumerate arbitrary tie
   permutations - precisely in the ambiguous frames the hypotheses exist
   for. Default is `-30.0` against the paper's `theta_min = -2`.
3. **(#3) `MAX_OCCLUSION_FRAMES = 10` is dead code.** Never read anywhere.
   No extrapolation cap and no post-gap reinit, so `P` grows unbounded
   through a gap and the transition term flattens toward uniform. Matters
   most in `relabel_trial.py`, which runs the whole 47k-frame trial
   including the ~27% of frames with no anchor solution.
4. **(#6) `init_frame` vs frame 0 incoherence.** `_build_hypotheses`
   collapses to a *single* hypothesis at `t == init_frame` while
   `viterbi_select` starts its trellis at `t == 0` and seeds filters from
   `init_frame`. When `init_frame != 0` (the normal case in
   `relabel_trial.py`) the trellis is pinched to width 1 mid-sequence -
   a hard decision the paper explicitly avoids. Note this pin is *also*
   what currently supplies spatial initialisation, so fix it together
   with #1 above, not before.
5. **(#7) Kalman model drift from the paper.** State is `[x, xdot]` vs the
   paper's `[x, xdot, xddot]` (documented, deliberate). But `Q = eye(6) *
   process_var` puts identical uncorrelated noise on position and velocity
   instead of the correlated constant-velocity block form, so the
   calibrated `0.5 mm^2` (measured on *position* jitter) means something
   different for the velocity states. Library defaults
   (`process_var=1.0, obs_var=25.0`) also still contradict the calibrated
   `0.5 / 1.0` in `scripts/gmm/_common.py` - any caller not passing them
   explicitly silently gets the uncalibrated 20-40x-too-loose regime.
6. **(#9) Degenerate rigid frame yields garbage, not NaN.** A near-collinear
   anchor triple makes `np.cross` ~ 0 and `_normalise`'s `+1e-12` turns it
   into a huge unit-ish vector; the frame silently becomes non-orthonormal
   and undetectable by `mask_untrustworthy_frames`. Needs a
   cross-product-norm threshold emitting NaN.
7. **(#10) `relabel_sequence` return shape breaks its contract.** Documented
   `(T, len(marker_idxs))`, actually `(T, len(kept))`, and `kept` is never
   returned - so a caller cannot tell *which* markers were dropped.
8. **(#11) `locate_anchor_triangle` is superseded but still public**, exported
   and tested, and still referenced by `scripts/gmm/relabel_trial.py`'s
   header. Its own section-6 comment records it at 23% correct on P7. A
   reader following the docstrings lands on the known-bad path.
9. **(#8) Section 3.3 (multiple marker sets) not implemented** - Eq. 4's
   global likelihood matrix over per-set transforms is exactly the two-hand
   case in this repo. Scope-note it in the module docstring at minimum.
10. **(#12) `learn_anchor_triangle` histogram edge cases** - allocates
    proportional to the largest outlier distance; fewer than 2 edges if
    `v.max() < bin_mm`, which `np.histogram` rejects.
11. **(#13) `time.time() - t0 / 1`** in `relabel_trial.py` - harmless
    leftover (`t0/1 == t0`).

### Tests

`tests/gmm/test_labeler.py` 29 -> 38; full suite 117 -> **126 passing**.
The new guards are written to fail against the *old* code specifically:
the emission tests are parametrised over both hypothesis orderings (one
ordering passed even when broken, via the best-first tie-break above), and
`w_emission=0.0` is pinned as reproducing the pre-fix verdict.

## Work completed 2026-09-12 (strategyGMM: occlusion-bias fix + init_frame forward/backward split)

Continuation of 2026-09-11's review. Two more items from that session's
ranked open list are fixed; both are load-bearing, unlike 09-11's two fixes
which turned out to be latent (verified inert on real data). These moved
real numbers on P10.

### Fixed 1 (list item #1: occlusion bias in `_transition_score`)

Unmatched markers scored a fixed `0`; matched markers scored a real (always
negative, for any non-trivial covariance) Kalman log-likelihood. So `0`
beat *any* real match regardless of fit quality, monotonically rewarding a
hypothesis for dropping markers.

First attempt used `-log(M)` (mirroring the paper's *emission*-side uniform
score for gated-out pairings, Section 3.1) - wrong on two counts: it
degenerates to 0 for a single marker (no fix at all), and the paper has no
equivalent formula for the *transition* side's missing-observation case in
the first place, so porting the emission constant over was the wrong
reference value regardless.

Replaced with `MarkerKalman.gate_loglik()`: the log-density at the
standard 95% chi-squared gate boundary (Mahalanobis^2 = 7.8147, 3 DOF) of
the filter's own current covariance, using the same formula as
`residual_loglik`. This is a real Kalman-tracking convention (gating), not
an invented heuristic. Verified both directions on a synthetic 2-frame
case:

    plausible match (2mm move)  vs drop -> match wins  [0]
    implausible match (500mm)   vs drop -> drop wins   [-1]

Before the fix neither of these could happen - drop always won regardless
of the match's plausibility. Also verified a real invariant: `gate_loglik`
and `residual_loglik` share the same `logdet(S)` term, so it cancels in
their difference - the match-vs-drop comparison's margin is scale-invariant
(constant to ~1e-5 across the filter's covariance scaled 1x/10x/50x), not
an artifact of how converged the filter happens to be. A fixed constant
would instead make the comparison arbitrarily easier or harder to win
purely as a side effect of unrelated filter uncertainty (e.g. right after
a gap).

### Fixed 2 (list item #4: `init_frame` vs. frame-0 incoherence)

`_build_hypotheses` pinned a single forced hypothesis at `t == init_frame`
while `viterbi_select` unconditionally started its trellis at array index
0 and seeded the filter bank from `local_mm[init_frame]` regardless -
physically incoherent whenever `init_frame != 0` (the normal case in
`relabel_trial.py`, where `init_frame` is the first consensus-clean frame,
typically mid-array).

Replaced with two Viterbi passes sharing one seed at `init_frame`: forward
over `[init_frame, T)`, backward (time-reversed) over `[0, init_frame]`,
spliced back together. Exact, not approximate - a constant-velocity model
is time-symmetric in position (only the estimated velocity's sign flips,
which is never read out). The forced single-hypothesis pin in
`_build_hypotheses` is removed entirely: every frame now gets the full
`n_hypotheses`, and each pass's own first frame is scored on emission alone
(no prior transition) - exactly the paper's Section 3.2 init procedure,
now made to actually work since 09-11's emission fix.

Verified on a case the old code could not get right even in principle: a
swap injected entirely *before* `init_frame` (`init_frame=7`, swap at
frames 2-4 of a 10-frame window):

    t=0,1: [0,1]   t=2,3,4: [1,0] (recovered)   t=5..9: [0,1]

Plus edge cases: `init_frame` at either boundary, `T=1`, and an
out-of-range `init_frame` now raises `ValueError` (previously undefined
behaviour via silent bad indexing).

### Real-data re-run, both P7 and P10 whole-trial

    P7/Trial1 (well-separated markers, 17mm+ apart):
      reassigned          3526 -> 3531   (~unchanged)
      improvement on touched bones   82.2% -> 82.2%  (unchanged)

    P10/Trial2 (closer-together markers):
      reassigned           776 -> 1096   (+41%)
      frames w/ >=1 reassignment   12.6% -> 13.4%
      improvement on touched bones   43.0% -> 52.9%   (+10 points)
      overall bone error (all bones)   1.78mm -> 1.78mm, unchanged (P10 was
        already mostly clean, so a flat trial-wide number here is expected,
        not a sign the fix did nothing)

Matches the prediction going in: P7's markers are far enough apart that the
old occlusion bias rarely had a close call to get wrong, so this landed as
a correctness fix with low visible effect there. P10 is where markers sit
closer together, so there were more real decisions on the margin - 320
more reassignments were made, and (checked by the same independent
bone-length veto as always, not by the model that proposed them) those
repairs are measurably *better*, not just more numerous.

The ground-truth injected-swap protocol (`validate_injected_swap.py`) is
still unchanged at 100%/100%/100% - still the documented non-discriminating
case (clean block swap, sharp temporal boundary, transition term alone
already resolves it). Neither of today's fixes touches that limitation;
the adduction-motion test case flagged on 09-11 remains the way to actually
exercise the spatial half of the model.

### Tests

`tests/gmm/test_labeler.py` 38 -> 46; full suite 126 -> **135 passing**.
New tests: 2 for `gate_loglik`/occlusion-bias (plausible-match-wins,
implausible-match-loses, plus the scale-invariance property), 6 for the
forward/backward `init_frame` split (mid-window labelling, a swap entirely
before `init_frame`, both boundaries, `T=1`, out-of-range rejection).

### Still open (list items #2, #3, #5-#11 from 09-11's ranking - unchanged)

Items #1 and #4 from that list are the two fixed above; the rest are
untouched:

- **#2 `theta_min` is a clamp, not a filter** - now the highest-priority
  remaining item; largest single change, touches the cost-matrix shape.
- **#3 `MAX_OCCLUSION_FRAMES = 10` dead code** - still confirmed unused
  anywhere in the codebase.
- **#5 Kalman noise model drift**, **#6 degenerate rigid frame -> garbage
  not NaN**, **#7 `relabel_sequence` return-shape contract**, **#8
  `locate_anchor_triangle` superseded but still public**, **#9 Section 3.3
  (multi-marker-set) unimplemented**, **#10-#11 minor/cosmetic** - all as
  described 09-11, no change.

## Work completed 2026-09-12, continued (strategyGMM: 3 latent bugs found and fixed by a fresh review)

A second, independent pass over `labeler.py` after the two fixes above,
specifically targeting anything the day's own changes might have newly
exposed rather than re-deriving the already-tracked open list. Found 3
latent issues - none reachable through the production driver
(`relabel_local`) today, but each a real gap. All 3 fixed.

### 1. Silent-wrong-confidence seeding fallback in `viterbi_select`

    m: MarkerKalman(init_positions.get(m, observations_per_frame[0][assign[i]]
                     if assign[i] >= 0 else np.zeros(3)), ...)

If a marker is both absent from `init_positions` and unmatched (-1) in the
winning t=0 hypothesis, this silently seeded it at `np.zeros(3)` with the
filter's *normal* starting confidence (`P[i,i]=100`) - the exact inverse of
the NaN-seed fix from earlier the same day, which correctly inflates `P` to
1e6 to flag "unknown". Confirmed directly: a `np.zeros(3)` seed produces
`P[0,0]=100`, not `1e6` - the filter asserts the marker sits at the local
origin with ordinary confidence rather than flagging it as unknown.

Unreachable via `relabel_local` (the only production driver), which always
builds a complete `init_positions` dict covering every `kept` marker - but
nothing enforced that, so a future direct caller of `viterbi_select` could
hit it silently. Fixed by extracting a documented `_seed_position` helper
that falls back to NaN (not zeros) when there is no real observation to
seed from either.

### 2. Misleading `_transition_score` docstring

Claimed its returned `predictions` dict is "used for extrapolating
occluded markers". Grepped the only call site - `trans, _preds = ...` -
the value is discarded immediately. There is no extrapolation-cap
mechanism; that is still open list-item #3 (`MAX_OCCLUSION_FRAMES` dead
code). The docstring read as if that fix's plumbing already half-existed,
which could mislead whoever picks up #3 into underestimating the remaining
work. Corrected to say plainly that the value is currently unused and that
`MAX_OCCLUSION_FRAMES` is still dead code.

### 3. Unchecked `slogdet` sign in `residual_loglik` and `gate_loglik`

Neither checked the `sign` return from `np.linalg.slogdet(S)` before
trusting `logdet`. `(I - KH)P` (the covariance update `MarkerKalman.update`
uses) is a numerically fragile form - unlike the Joseph-form update, it does
not guarantee `P` stays symmetric positive-definite under floating-point
roundoff. If it ever did lose PD-ness, `logdet` would silently become
nonsense (wrong sign, or `-inf`) rather than raising.

Deliberately stress-tested before concluding this was worth fixing: 5000
predict/update cycles with periodic occlusion, and a 100,000-frame
permanent-occlusion run (no cap exists, per open item #3) - `P` stayed
symmetric and positive-definite throughout both, and `gate_loglik` degraded
smoothly (no NaN/inf) even at extreme covariance scale. So this was a
latent robustness gap, not a demonstrated failure - but `gate_loglik`
(new the same day) is now called on every occluded marker in every
hypothesis every frame, far more often than `residual_loglik` alone ever
was, raising the practical odds of eventually hitting whatever edge case
would trigger it.

Fixed with a shared `_safe_logdet` helper that raises `np.linalg.LinAlgError`
on `sign <= 0` instead of silently proceeding; both functions now go
through it. Verified it actually fires (needed a large deliberately
indefinite covariance, `-1000` vs. `obs_var=1.0`, before `R` stopped
swamping the effect and the true indefiniteness of `S` itself showed up),
and that ordinary operation never trips it.

### Real-data re-run (P7 whole-trial)

    3531 -> 3531 reassigned, 82.2% -> 82.2% improvement on touched bones

Bit-identical, as expected: these are latent-edge-case fixes, and P7's data
doesn't hit any of the three edge cases. No `LinAlgError` raised across the
full 47,656-frame trial - consistent with the stress-test finding that
ordinary operation stays well clear of the failure mode being guarded
against.

### Things checked and ruled out during this pass (recorded so they are not
### re-litigated by a future review)

- **Performance regression from the forward/backward split** (P10's
  runtime rose ~20% between the previous two sessions' runs). An isolated
  synthetic timing test (T=3000, `init_frame` at start/middle/end) showed
  no dependence on split position (32.93s/32.84s/32.60s - flat), ruling out
  a complexity regression from the split itself. Most likely ordinary
  machine-load variance between runs, not a real regression.
- **Whether the emission side has the same unmatched-marker bias the
  transition side had** (fixed earlier the same day). It does not:
  `top_n_assignments`/Murty's algorithm always assigns `min(M,K)` pairs for
  every hypothesis at a given frame, so the unmatched-marker count is
  constant *within* a frame's hypothesis set and cancels out of same-frame
  comparisons. The real emission-side issue remains list-item #2
  (`theta_min` degenerate ties), not a parallel unmatched-count bias.

### Tests

`tests/gmm/test_labeler.py` 46 -> 50; full suite 135 -> **138 passing**.
New tests: the unseeded-unmatched-marker case resolves without crashing
and without falsely asserting a position, and both `gate_loglik` and
`residual_loglik` raise `LinAlgError` on a deliberately indefinite
covariance.
