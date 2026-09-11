# Strategy: marker-quality cascade

*Split out of CLAUDE.md; see `../../CLAUDE.md` for the index.*

## Work completed 2026-09-04 through 2026-09-07 (marker-quality cascade for raw Vicon exports)

New `vicon2mano/strategies/cascade/quality_cascade.py` — separate from the MANO-fitting
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

`quality_cascade.debug_frame_cascade(markers, labels, ref_frames, t)`
prints the full step-by-step reasoning for one frame/timestep — which
distances it checked (including the local vs. global forearm reference),
what they were compared against, and exactly where in the chain a marker
was accepted or thrown out. Reach for it whenever a frame's verdict looks
surprising.
