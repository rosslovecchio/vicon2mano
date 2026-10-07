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

## In progress (2026-10-07)
- Fitting 3 more trials picked by automated clean-ness proxy (not human-verified
  like P10): P15/Trial2, P5/Trial2, P3/Trial1, all Hands only. Running in background.
- Mesh+marker HTML animation: done for P10.
  `results/mano/P10_Trial2_handsonly/eval/mesh_animation.html` (script:
  `scripts/mano/animate_mesh.py`).

## Next decision (pick one, nothing proceeds until you do)
- [ ] (nothing blocked right now — both items above are mid-flight)

## Rules for this file
- Max ~20 lines. If it's longer, delete the old stuff.
- Only one "next decision" open at a time.
- Claude edits the Status/Next-decision sections when asked; you edit anything.
