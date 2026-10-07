# Progress tracker

## Goal
Get accurate joint angles from Vicon marker data. Blocker: marker labels are
wrong in ~2-20% of frames depending on trial, so relabeling has to work first.

## Status (2026-10-07)
- MANO fitting pipeline works. Fit on manually-clean trial (P10/Trial2 Hands
  only) gives ~15mm residual, physically plausible joint angles. **Done, not blocking.**
- GMM-Viterbi relabeler: fixed one real bug (theta_min), verified against
  full-trial human ground truth for the first time. Result: it still breaks
  more labels than it fixes (90 fixed vs 152 broken on P10/Trial2). **Not
  deployable yet.**

## Decision made (2026-10-07)
Priority is a clean set of joint angles, not a better GMM-Viterbi. GMM-Viterbi
work is paused — not abandoned, just not the thing to improve right now.

## Done (2026-10-07)
Joint angles delivered for P10/Trial2 Hands only:
- `scripts/mano/export_joint_angles.py --npz <fit.npz> --out <path> --fps 200 --plot`
- `results/mano/P10_Trial2_handsonly/joint_angles.csv` + 5 per-finger PNGs
- Caveat: angle = rotation magnitude in degrees (MCP/PIP/DIP), not a true
  flexion/abduction decomposition. Good enough to see bend amount, not clinical-grade.

## Status (2026-10-07, done)
4 trials now have joint angles (P10 + 3 more picked by automated clean-ness
proxy, NOT human-verified like P10 — flagging that honestly):

| Trial | Result |
|---|---|
| P10/Trial2 Hands only | Clean (human-verified, 97.6% accurate) |
| P15/Trial2 Hands only | Clean |
| P5/Trial2 Hands only  | 0.04% of frames spike to ~168deg, 2 brief windows (t=123s, 207s) |
| P3/Trial1 Hands only  | ~16s noisy region at t=242-257s, real mislabelling likely |

All in `results/mano/<trial>/joint_angles.csv` + 5 per-finger PNGs each.
Mesh+marker HTML animation also done for P10.
Bug fixed: plot titles were hardcoded to "P10" regardless of input trial.
Everything committed and pushed to origin/feat/gmm-labeler.

## Next decision (pick one, nothing proceeds until you do)
- [ ] Leave P5/P3's bad windows as-is (just documented) — done for now.
- [ ] Dig into why P3's t=242-257s window is bad (relabel just that window?).
- [ ] Something else — say what.

## Rules for this file
- Max ~20 lines. If it's longer, delete the old stuff.
- Only one "next decision" open at a time.
- Claude edits the Status/Next-decision sections when asked; you edit anything.
