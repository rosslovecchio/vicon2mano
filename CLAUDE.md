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
- Recordings: `D:\ExperimentsJune25` — see `vicon2mano/core/dataset.py`
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
