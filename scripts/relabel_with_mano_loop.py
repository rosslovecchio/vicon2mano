#!/usr/bin/env python3
"""Label-agnostic MANO relabelling — thin preset over ``relabel_with_mano.py``.

Same pipeline, same CLI, same output format as ``relabel_with_mano.py``, with
one difference: ``--fit-on`` defaults to ``label-agnostic`` instead of
``correct``. In that mode the pose fit still seeds from cascade-CORRECT
markers (preserving known-good correspondences for the first iteration), but
every further iteration reassigns among *all present markers*, not just the
ones the cascade flagged — see ``mano_relabel.relabel_frame_iterative`` and
``relabel_with_mano._iterate_label_agnostic`` for the algorithm.

Usage
-----
python scripts/relabel_with_mano_loop.py --participant P7 --trial "Trial 1 Hands only"

Everything else (``--min-correct-pct``, ``--max-dist-mm``, ``--passes`` as
the iteration cap, ``--all``, etc.) is identical to ``relabel_with_mano.py``;
run ``python scripts/relabel_with_mano.py --help`` for the full flag list, or
pass ``--fit-on correct``/``--fit-on all`` here to fall back to the original
behaviour for a direct A/B comparison without switching scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import relabel_with_mano as rwm   # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--fit-on" or a.startswith("--fit-on=") for a in argv):
        argv = ["--fit-on", "label-agnostic", *argv]
    return rwm.main(argv)


if __name__ == "__main__":
    main()
