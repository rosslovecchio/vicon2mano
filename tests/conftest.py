"""Put the repo root on sys.path for every test directory.

Tests are grouped by strategy (tests/gmm, tests/mano, ...) and pytest's
default import mode only adds each test file's own directory, so without
this the package would only be importable via an editable install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
