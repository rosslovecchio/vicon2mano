# Progress tracker

## Goal
Get accurate joint angles from Vicon marker data. Blocker: marker labels are
wrong in ~2-20% of frames depending on trial, so relabeling has to work first.

## Status (2026-10-07)
- Joint angles delivered for 4 trials (P10, P15/Trial2 clean; P5/Trial2 and
  P3/Trial1 have small flagged bad windows — see git log for detail).
  `results/mano/<trial>/joint_angles.csv` + per-finger PNGs + mesh animations.
- GMM-Viterbi relabeler: fixed theta_min bug, still not deployable (net
  harmful on raw labels). Paused, not priority right now.
- Mesh animation has 2 known rendering bugs, not yet fixed: (1) markers can
  render hidden inside the opaque mesh (mplot3d depth-sort limitation,
  zorder doesn't fix it), (2) dark hatching on the mesh surface (a lighting
  artifact — fixed by using ambient-only lighting, not yet applied to the
  production animations). Real fix for both = switch frame baking from
  matplotlib to Plotly+Kaleido (confirmed in testing: correct depth +
  ambient-only lighting = clean render, just slower to bake, ~4s/frame).
- Found a real, fixable bug in `MANOFitter`: it fits joint position directly
  to marker position, ignoring the systematic skin-surface offset (MCP/wrist
  17-28mm, PIP/DIP 5-16mm). Added `target_override` hook + a two-pass
  calibrate-then-refit script (`scripts/mano/refit_with_marker_offsets.py`).
  Confirmed real improvement on one tested frame: 14.5mm -> 8.6mm marker-
  prediction accuracy. Not yet applied to the full-trial fits.
- Confirmed structural (not fixable by any fitting strategy): no Vicon
  marker exists at any fingertip, so thumb/index-touching gestures never
  look fully closed in the mesh (~50mm gap, unchanged across every fit tried).

## Next decision (pick one, nothing proceeds until you do)
- [ ] Apply the offset-corrected refit to all 4 full-trial fits (bigger job,
      re-fits every frame, not just one window).
- [ ] Switch mesh animation rendering to Plotly+Kaleido to fix both display bugs.
- [ ] Something else — say what.

## Rules for this file
- Max ~20 lines. If it's longer, delete the old stuff.
- Only one "next decision" open at a time.
- Claude edits the Status/Next-decision sections when asked; you edit anything.
