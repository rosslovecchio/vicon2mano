#!/usr/bin/env python3
"""CLI for the marker-quality cascade.

The implementation now lives in
``vicon2mano.strategies.cascade.quality_cascade`` -- it is library code that
the MANO driver and the notebook both import, so it no longer sits in
``scripts/``. This file is the command-line entry point only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vicon2mano.strategies.cascade.quality_cascade import main  # noqa: E402

if __name__ == "__main__":
    main()
