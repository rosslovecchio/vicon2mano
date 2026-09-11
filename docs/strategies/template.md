# Strategy: template / distance-descriptor matching

*See `../../CLAUDE.md` for the index.*

Labels an unlabelled but *complete* marker cloud by matching it to a
labelled template recording, using a rigid- and scale-invariant per-marker
distance descriptor plus the Hungarian algorithm. Marker positions are never
modified -- this only names columns.

- Core: `vicon2mano/core/correspondence.py`
  (`relabel_by_template`, `relabel_two_hands`, `_distance_descriptor`)
- Scripts: `scripts/shared/relabel_markers.py`, `scripts/shared/detect_swaps.py`

**Limitation -- chirality.** Pairwise-distance descriptors are
reflection-invariant, so a left and a right hand (near mirror images) have
nearly identical signatures; on a two-hand cloud the matcher swaps markers
between hands (validated: 22/22 single-hand, 16/44 two-hand).
`relabel_two_hands` works around it by splitting the cloud spatially and
relabelling each hand against a single-hand template.
