"""Shared visualisation vocabulary for relabelling animations.

Every strategy's animation should read the same way -- same finger colours,
same digit labels, same status-ring semantics -- so a reviewer can compare
a MANO run against a GMM run without relearning the encoding. These lived
in ``scripts/relabel_with_mano.py``; the GMM driver imported that module
purely for the palette.
"""

from __future__ import annotations

FINGER_PALETTE = {
    "thumb": "#e74c3c", "index": "#27ae60", "middle": "#2980b9",
    "ring": "#e67e22", "pinky": "#8e44ad", "palm": "#444444",
    "forearm": "#95a5a6",
}

# Ring colours. Each strategy maps its own status codes onto these, but the
# *meaning* is fixed: green = we left this label alone, blue = we moved it,
# red = we moved it and it still does not verify, nothing = untouched /
# no information this frame.
RING_KEPT = "#2ecc71"        # green
RING_MOVED_OK = "#2e86ff"    # blue
RING_MOVED_BAD = "#e74c3c"   # red
RING_NONE = "rgba(0,0,0,0)"  # no ring


def marker_color(label: str) -> str:
    low = label.lower()
    for key, color in FINGER_PALETTE.items():
        if key in low:
            return color
    return "#000000"


def marker_digit(label: str) -> str:
    for ch in reversed(label):
        if ch.isdigit():
            return ch
    return ""


# Segments of the MARKER SET itself -- the chains the Vicon labels describe,
# drawn so the markers read as a hand rather than a cloud of dots. This is
# raw measured geometry, independent of the fitted model: where a bone and
# the mesh disagree, that is the fit's error made visible.
MARKER_BONES = [
    ("Palm1", "Palm2"), ("Palm2", "Palm3"), ("Palm3", "Palm1"),
    ("Palm2", "Thumb1"), ("Thumb1", "Thumb2"), ("Thumb2", "Thumb3"),
    ("Palm2", "Index1"), ("Index1", "Index2"), ("Index2", "Index3"),
    ("Palm2", "Middle1"), ("Middle1", "Middle2"), ("Middle2", "Middle3"),
    ("Palm2", "Ring1"), ("Ring1", "Ring2"), ("Ring2", "Ring3"),
    ("Palm2", "Pinky1"), ("Pinky1", "Pinky2"), ("Pinky2", "Pinky3"),
    ("Index1", "Middle1"), ("Middle1", "Ring1"), ("Ring1", "Pinky1"),
]


def marker_bone_pairs(labels):
    """(i, j) index pairs for MARKER_BONES present in this label set (subject prefixes like "P15:" ignored)."""
    idx = {l.split(":")[-1]: i for i, l in enumerate(labels)}
    return [(idx[a], idx[b]) for a, b in MARKER_BONES
            if a in idx and b in idx]
