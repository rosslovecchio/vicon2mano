# Marker-label swap detection

Catches small label errors in recordings that pass triage as "clean": two
markers whose **names are swapped** (adjacent fingers, or two markers on one
finger). All markers are present and named for the set, so presence-based
triage misses them, but they corrupt the joint angles. Tool:
`scripts/detect_swaps.py`.

## Method — two complementary geometric signals

1. **Descriptor swap-test (primary).** A consensus per-label distance
   descriptor is built as the *median over many clean recordings* of each
   marker's sorted pairwise-distance signature (rigid- and scale-invariant;
   the median makes the reference robust to the occasional swapped recording).
   For each anatomically plausible pair (the k nearest labels by descriptor),
   we test whether exchanging the two labels fits the consensus better than
   the current labelling, by more than a margin. Catches both cross-finger
   and same-finger swaps.
2. **Bone-length confirmation.** Whether the swap also brings the affected
   within-finger bone lengths closer to their canonical (consensus) values.
   This is decisive for **cross-finger** swaps but **blind to same-finger**
   ones (two markers on one finger barely change the bone-length set).

### Why not require both to agree
The first design *required* both signals — high precision, but recall
collapsed to **38%** because the bone-length veto discards every same-finger
swap. So bone-length is used to **rank confidence**, not to veto:

| confidence | meaning |
|---|---|
| high | descriptor + bone-length agree (cross-finger, strong) |
| medium | descriptor only — review before trusting |

## Validation (inject one known swap per held-out recording)

At margin 0.25 on `Hands_only_Left`: **88% recall (7/8)**, ~1 detection on
as-is recordings over 8 (which may itself be a real swap). A margin sweep
showed 0.2–0.3 is the sweet spot; higher margins lose recall without much
false-positive benefit.

## Findings on the cohort's clean single-hand sets

| set | recordings w/ ≥1 flag | swaps | notes |
|---|---|---|---|
| Hands_only_Left  | 2 / 21 (10%) | 3 (all medium) | 1 cross-finger (Index3↔Middle3) + 2 thumb (noisiest marker) |
| Hands_only_Right | 6 / 19 (32%) | 6 (all medium) | **all forearm** (Forearm3↔4 etc.) |

- **No high-confidence swaps on real clean data** → these recordings are
  genuinely well-labelled; confirmed finger swaps are rare.
- Right-hand flags are dominated by **forearm** markers, which sit
  near-symmetrically on a rigid plate — ambiguous to the geometry and
  **irrelevant to finger joint angles**. (A follow-up restricts detection to
  finger markers and adds a motion-correlation signal for same-finger cases.)

## Fixing

`--fix` corrects high-confidence swaps; `--fix-medium` includes the rest.
Labels are exchanged and the recording re-written in wide CSV — **marker
positions are never modified**.

```bash
python scripts/detect_swaps.py --h5 data/all_trajectories_synced.h5 \
    --session-type Hands_only_Left --side left --validate   # precision/recall
python scripts/detect_swaps.py --h5 data/all_trajectories_synced.h5 \
    --session-type Hands_only_Left --side left               # scan + report
```
