# Marker-label swap detection

Catches small label errors in recordings that pass triage as "clean": two
markers whose **names are swapped** (adjacent fingers, or two markers on one
finger). All markers are present and named for the set, so presence-based
triage misses them, but they corrupt the joint angles. Tool:
`scripts/detect_swaps.py`.

Detection is restricted to **finger markers**. Forearm markers are excluded:
the four sit near-symmetrically on a rigid plate, are ambiguous to swap, and
do not affect finger joint angles (without this they dominated the flags).

## Method — three signals

1. **Descriptor swap-test (primary).** A consensus per-label distance
   descriptor is built as the *median over many clean recordings* of each
   marker's sorted pairwise-distance signature (rigid- and scale-invariant;
   the median makes the reference robust to the occasional swapped recording).
   For each plausible pair (k nearest finger labels by descriptor), test
   whether exchanging the two labels fits the consensus better than the
   current labelling, by more than a margin. Catches both swap types.
2. **Bone-length confirmation.** Whether swapping brings within-finger bone
   lengths closer to canonical. Decisive for cross-finger swaps.
3. **Motion confirmation (independent, kinematic).**
   - *Cross-finger*: a marker's speed time-series should correlate with its
     labelled finger's other markers; if it instead matches the swap
     partner's finger, the swap is confirmed.
   - *Same-finger*: motion amplitude grows distally (DIP > PIP > MCP), so an
     amplitude ordering that is violated and repaired by the swap confirms it.

### Confidence (not a veto)
A "require both signals to agree" gate was tried first and rejected — it
dropped recall to 38% by discarding every same-finger swap (bone-length is
blind to them). Instead the descriptor flags candidates and the other two
signals **rank confidence**:

| confidence | meaning |
|---|---|
| high | descriptor + (bone-length OR motion) agree |
| medium | descriptor only — review before trusting |

## Validation (inject every plausible pair into each held-out recording)

`Hands_only_Left`, 304 injected-swap trials, margin 0.25:

| swap type | recall | of detected, % high-confidence |
|---|---|---|
| cross-finger | 81% | 95% |
| same-finger | 90% | 100% |
| **overall** | **83%** | — |

Adding the motion signal is what lifts true detections to high confidence
(both types); detections on as-is recordings stay ~1 per 8.

## Findings on the cohort's clean single-hand sets

- **Hands_only_Left**: 3 candidates in 2/21 recordings, all *medium*
  (descriptor-only, neither bone nor motion confirmed) — e.g. Index3↔Middle3,
  a thumb same-finger pair. Uncertain; flagged for review, not auto-fixed.
- **Hands_only_Right**: **0** after excluding forearm markers (previously 6,
  all forearm).
- No high-confidence swaps on real clean data → these recordings are
  genuinely well-labelled. The detector's value is the validated ability to
  catch swaps with high confidence (per the injection study) while staying
  appropriately skeptical of ambiguous real-data candidates.

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
