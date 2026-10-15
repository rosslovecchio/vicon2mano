# `fit_surface_markers.py` — design rationale

Why this script exists, why each piece is the way it is, and what was tried and
rejected. The numbers here are measurements on P10/Trial2, not estimates; where
something is unverified it says so.

All line references are to [`fit_surface_markers.py`](fit_surface_markers.py).

---

## The problem

Every fit in this repo before this script minimised `||J_j(θ,β) − marker||²` —
the *joint centre* pulled onto the marker. A joint centre is inside the hand, so
a perfect fit by that objective necessarily buries each marker 10–25 mm below
the skin. That was the "markers are inside the mesh" symptom: not a rendering
fault, not a labelling fault, but the objective asking for the wrong thing.

This script fits a point on the **mesh surface** to each marker instead — the
standard SMPL/MANO "virtual marker" approach.

[`refit_with_marker_offsets.py`](refit_with_marker_offsets.py) attacks the same
problem with a simpler model (a rigid per-marker offset in a bone-segment
frame). `--offset-mode` implements that model inside this script so the two can
be compared with everything else held fixed. **That ablation has not been run to
a conclusion yet** — until it is, the claim that the mesh representation earns
its complexity is unproven.

---

## Input data

Use `P10/Trial2_handsonly_manuallylabelled_filled.csv`, **not** the raw
`Trial2_handsonly.csv`.

| | raw | relabelled+filled |
|---|---|---|
| frames with all 22 markers | 42.7 % | 100 % |
| worst markers | Thumb3 76 %, Palm1 85 % | — |

Measured difference with an identical config: pinch window fingers 5.82 → 5.81 mm,
fist window 18.86 → 18.29 mm. So clean input is **not** what fixes the
high-flexion problem — that is a model limitation (see Open problems).

Note 42.7 % is the probability that *no* marker anywhere is missing. Individual
markers are 76–99 % available, and a marker-derived joint angle needs only its
own 3–4 markers, so those are computable on 80–95 % of frames.

---

## Pipeline

### 0. Measured once, before any fitting — `joint_axis_basis()`

[L210](fit_surface_markers.py#L210). This is the least obvious part of the
script, so it is worth spelling out.

#### The problem it solves

MANO's pose is 45 numbers: 15 joints × a 3-component axis-angle vector. For
joint *j*, the vector `hp[3j : 3j+3]` says "rotate this joint about this axis by
this much" — its direction is the rotation axis, its length the angle.

We want to talk about **flexion, abduction and torsion** — to bound them
anatomically, to seed pose from marker-measured angles, and to report angles you
can sanity-check. The temptation is to assume the three components *are* those
three motions. They are not. MANO's axes come from how the model was built, not
from anatomy, and they are different per joint. Measured by perturbing each
component and watching the fingertip:

```
joint     axis 0 → flexion loading
index2         0.97
pinky1         0.94
ring2          0.90
middle1        0.81
```

So axis 0 — which an anatomical reading would call "twist" — carries most of the
*flexion* for several joints. An early version of this script assumed otherwise
and clamped axis 0 to ±0.10 rad as if it were torsion. That silently forbade
flexion and the fingers could not close. Hence: measure the axes, don't assume
them.

#### How the axes are recovered

For a small rotation, a point at offset `r` from the joint centre moves by

```
d = a × r
```

where `a` is the **world rotation axis**. So: nudge one pose component by a small
ε, read the displacement `d` of every mesh vertex skinned to that segment, and
solve that equation by least squares for `a`. Doing this for all three
components gives a 3×3 matrix `M` mapping pose-space → world rotation axis.

Then ask for the anatomical axes directly, by inverting `M`:

| axis | world direction we want | why |
|---|---|---|
| **twist** | `u` = the segment's own long axis (joint → child joint) | axial rotation of the bone |
| **abduction** | `p` = the palm normal | splaying sideways |
| **flexion** | `cross(u, p)` | rotation about this sends the tip palmward |

`B = M⁻¹ · [flex, abd, twist]`, normalised. Projecting any pose onto `B`'s
columns then gives genuine flexion/abduction/torsion in radians.

#### Two traps, both of which this code hit

**Rotation axes, not tip displacement.** An earlier version derived flexion from
the fingertip Jacobian using `Jᵀ` — the direction of steepest increase in
palmward tip motion. That is only the right answer when the Jacobian's columns
are orthonormal, which they are not, so it returned a blend of flexion and
abduction and the fingers swung sideways. Working in rotation axes avoids the
question entirely, and the pseudo-inverse `M⁻¹` (not `Mᵀ`) is what maps a
*desired* motion back into pose space.

**Twist must be defined, not left over.** Defining twist as `cross(flex, abd)` —
whatever remains after the other two — is only meaningful if flexion and
abduction are cleanly separated. For the middle finger they are not, so its
"twist" contained real flexion, and pinning torsion to zero dragged the finger
straight (middle PIP 9.5° → −3.4°). Defining twist as the segment's own long
axis fixes this, because every axis is then specified from anatomy rather than
inherited from the others' residue.

**The sign must be pinned globally.** A cross product's direction depends on
operand order, so the palm normal's sign is arbitrary. Left unpinned, flexion
inverts and the fingers bend backwards. It is resolved once, from an anatomical
fact: in MANO's rest pose the thumb sits on the palmar side of the plane through
the MCPs.

#### The verification that must pass

Apply +0.5 rad along each joint's flexion axis and check the fingertip moves
**palmward**, not sideways:

```
measured: 0.98–1.00 palmward, 0.00–0.06 sideways, all fingers
(thumb CMC is 0.52/0.58 — correct, as its primary motion is opposition
 across the palm rather than palmar flexion)
```

Earlier, weaker versions of this check — "flexion shortens tip-to-wrist" —
passed with both the sideways bug and the sign bug present, because sideways
bending also shortens that distance. The check must compare the palmward
component *against* the sideways one.

### 1. Static stage — calibrate shape and attachments

**Stage 1** [L850](fit_surface_markers.py#L850): a conventional joint-centre fit
on ~30 static frames, Kabsch-initialised. Exists only to produce a plausible
starting pose; its objective is the one this script is replacing.

**Stage 2** [L946](fit_surface_markers.py#L946), 5 ICP rounds:

1. **Attach** — `closest_point_on_faces()` [L574](fit_surface_markers.py#L574)
   picks the nearest barycentric point on an allowed face, subject to two
   filters: the marker must be on the face's **outward** side, and the face must
   face **dorsally** (`segment_dorsals()` [L537](fit_surface_markers.py#L537),
   which rotates with each finger — a single palm normal is wrong the moment a
   finger flexes).
2. **Pin overrides** — hand-picked vertices from `--attach`, with a warning if
   the vertex is poorly skinned to its own joint.
3. **Optimise** betas + pose against `||P + standoff·n − marker||²` plus a
   one-sided penalty for any marker ending up inside the mesh.

**Output: betas and the (face, barycentric) attachment per marker — frozen from
here.** Both are physical facts about the session; re-deriving them per frame
would let a marker slide to different skin whenever the fit drifted, rewarding a
wrong pose with a flattering residual.

### 2. Movement stage — pose only

[L1110](fit_surface_markers.py#L1110), per chunk:

1. **Seed** — `angles_from_markers()` [L360](fit_surface_markers.py#L360) reads
   MCP and PIP flexion straight off marker geometry, per frame.
2. **Root** — only `global_orient` and `transl`, against palm + MCP markers.
3. **Thumb**, then **per-chain** stages — each chain alone against its own
   markers, gradients masked, root frozen.
4. **Joint polish** — all markers, root still frozen.

Loss per stage: surface-point-to-marker + inward penalty + temporal
second-difference smoothness + `basis_limit_penalty()`
[L341](fit_surface_markers.py#L341), with the attachment displaced by
`stretch_offsets()` [L424](fit_surface_markers.py#L424).

---

## Design decisions, with the evidence

### Judge `FINGERS`, not the overall mean

7 of 18 markers (Palm1/2/3 + four MCPs) are satisfied by the root transform
alone, so the overall mean rewards a hand that never articulates. **A flat,
non-articulating hand scores 9.37 mm overall.** This metric choice cost several
hours of wrong conclusions before it was noticed; the fitted-vs-marker flexion
comparison is the independent cross-check.

### Pose must be seeded from marker-derived angles

From a flat start the fit converges to a near-rigid hand — index PIP at −17.1°
where the markers measure +34.7° — because flexing initially moves the
attachment *away* from its marker before it gets closer. Seeding from measured
angles starts on the correct side of that barrier.

| | fingers |
|---|---|
| flat start | 10.43 mm, hand barely articulates |
| marker-angle seed | 13.20 mm, every joint flexed correctly |

The seeded fit scores *worse* and is *right*. This is the clearest case of the
metric trap above.

### Full 45-DOF pose, not 6 PCA components

The repo default `n_pca_comps=6` is a grasp basis; abduction lives in later
components, so a 6-component fit cannot adduct the fingers. Measured: lateral
marker spans were 0.85–0.89× the mesh's, unreachable under that basis.

### Zero finger torsion — load-bearing, not cosmetic

With torsion free the optimiser satisfies the nailbed marker by **twisting**
rather than flexing, leaving the PIP slack and the MCP over-rotated. Pinning it
fixed the index split: PIP 26.6° → 34.8° against a measured 34.7°.

This only worked once the twist axis was defined correctly (see Bugs).

### Bounded abduction, unbounded flexion

Abduction p99 is 22–24° (anatomical) but maxima reached 66–100° — the optimiser
splaying a finger sideways *through* its neighbour. Fingertips came out in the
wrong order in 8.8 % of frames.

| | pinch fingers | fist fingers |
|---|---|---|
| unbounded | 6.65 mm | 20.11 mm |
| abduction bounded | **5.82 mm** | **18.86 mm** |

Removing a DOF the hand does not have *improves* the fit. Flexion is the
exception and must stay free — the markers show MCP flexion to 128°, past
textbook, and bounding it collapses the fingers.

### Root frozen after its stage

Palm + MCP markers alone place the hand to 2.6–6.4 mm, but letting the fingers
vote on `global_orient`/`transl` afterwards pushes those same markers to
11–28 mm. The fingers drag the root.

### Per-chain staging

With the root frozen each finger is an independent kinematic chain, so fitting
them separately is mathematically equivalent — but differs in optimisation
dynamics, and that matters: fingers 9.54 → 7.95 (thumb-first) → 6.65
(per-finger). Unlike every other gain, this one did **not** drift the joint
angles away from measured values.

### Flexion-dependent attachment slide

Markers are on the dorsum, and the dorsal surface over a flexed knuckle is
longer than over an extended one. Measured across the trial:

| segment | static | at <30° | at >80° | correlation |
|---|---|---|---|---|
| Index1–Index2 | 47.6 | 44.3 | 54.0 | +0.85 |
| Middle1–Middle2 | 49.3 | 49.0 | 58.5 | +0.80 |
| Palm1–Palm2 (control) | 32.2 | 32.9 | 34.2 | +0.22 |

Bone cannot lengthen; this is skin, and MANO's linear blend skinning does not
reproduce it. Modelled as a slide along the segment axis, differentiable in the
pose. Two things had to be right:

- Measured **relative to the static's own flexion** (19–28° at the MCP — the
  static is not an extended hand). From zero it double-counts and makes the fit
  worse: fingers 6.65 → 8.36 mm.
- **Gated above 45°**. Below that the correction is ~1 mm and loses to noise
  (6.65 → 7.14); near a fist it is worth 0.9 mm (21.01 → 20.11).

Net: a small win, only at high flexion. A single linear term in one joint's
angle explains R² 0.64–0.73 of the finger-2 variance and only 0.06–0.43 of
finger-3 — so it is recorded as a partial model, not a solution.

---

## Tried and rejected

| Change | Result | Why |
|---|---|---|
| Bound flexion to textbook range | fingers 13.20 → 21.69 | real flexion exceeds it; fingers collapse to the bound |
| Attach only to the segment *distal* to the joint | 17.74 → 20.28 mm | observability improves but the target becomes unreachable; middle PIP/DIP stopped articulating |
| `--min-own-weight > 0` with staging | palm+MCP 8.07 → 10.38 | moves MCP markers onto finger-dependent skin, so the root stage chases moving targets |
| Converge the root fully (`--iters-root 1500`) | fingers 9.54 → 10.74 | helps palm+MCP but over-rotates knuckles (middle MCP 38.4° → 57.5° vs measured 30.2°) |
| L2 pose prior (`w_pose`) | monotonically worse | unweighted L2 in PCA space suppresses abduction, which lives in high-coefficient components |
| Exclude finger-2 markers | palm+MCP −31 %, fingers neutral | confirms they corrupt the root, but discards real information |
| `--mv-chunk 6000` for speed | ~8.5 min/chunk vs 2.9 min at 1500 | work is compute-bound, not launch-latency-bound; no speedup |

---

## Bugs that shaped the design

Recorded because each produced a plausible-looking wrong answer:

- **NaN gradients from masked markers.** `0 × NaN = NaN` in the inward term
  poisoned every gradient. Finite loss, dead parameters. Fixed with
  `nan_to_num` under the mask.
- **Chunk warm-start overwrote the marker seed.** `hp0[:] = prev_hp` replaced a
  whole chunk's per-frame seed with one stale pose, so only chunk 1 was ever
  seeded. A one-chunk window scored 6.65 mm while the full trial scored 19.28.
  Found by spot-checking a single frame where the markers showed a 90° fist and
  the fit showed 0°.
- **`Jᵀ` instead of `J⁺`** when mapping desired tip motion into pose space —
  fingers bent sideways.
- **Unpinned sign on the palm normal** — fingers bent backwards.

The last two both passed a verification I had written too weakly ("flexion
shortens tip-to-wrist" is also satisfied by sideways bending). The current check
compares the palmward component against the sideways component.

---

## Open problems

1. **High-flexion poses fit ~3× worse** (fist 18.29 mm vs pinch 5.81 mm) and it
   is *not* input quality, attachments, joint limits, or the stretch model —
   all four were tested. Unexplained.
2. **Thumb1 at ~17 mm on the static**, attached to `palm`. Worst marker in the
   set, and since attachments are frozen this propagates to all 42 238 frames.
3. **Betas are weakly identified.** `w_shape = 1e-2` dominates a data term of
   ~5.7e-4, so shape stays near the mean hand. Measured scale ratio is 0.998, so
   this is approximately right, but it is not a strong calibration.
4. **No automatic angle-agreement check.** Residuals are reported per marker but
   fitted angles are never compared to marker-derived angles across the trial —
   the absence of which is why the warm-start bug survived.
5. **`--offset-mode` ablation unfinished.** Until it reports, the mesh
   representation is unjustified relative to the simpler
   `refit_with_marker_offsets.py` model.

---

## Reproducing

```bash
.viconvenv/Scripts/python.exe scripts/mano/fit_surface_markers.py \
  --static ".../P10/static_trial.csv" \
  --movement ".../P10/Trial2_handsonly_manuallylabelled_filled.csv" \
  --attach  configs/surface_fit/P10_attach.json \
  --stretch configs/surface_fit/P10_stretch.json --stretch-gate 45 \
  --staged --iters-thumb 800 --iters-finger 800 \
  --min-own-weight 0 --marker-init \
  --biomech 1.0 --free-flexion --twist-weight 10 \
  --standoff 3 --icp-rounds 5 --w-pose 0 --w-smooth 1.0 --iters-mv 1500
```

Parameters and their justifications are also recorded in
[`configs/surface_fit/P10.json`](../../configs/surface_fit/P10.json).

Progression on the pinch window: **13.20 → 9.54 → 7.95 → 6.65 → 5.82 mm**
(seeded → zero torsion → thumb-first → per-finger → bounded abduction).
