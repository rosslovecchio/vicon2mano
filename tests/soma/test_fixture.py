"""Regression tests for the real-data tutorial fixture (scripts/soma_all/make_fixture.py).

The fixture is participant data and is not in the repository: every test skips unless results/soma/fixture_P3 exists
(tests that compare with the source recording also need the data root).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "results/soma/fixture_P3"
pytestmark = pytest.mark.skipif(not (FIX / "MANIFEST.json").exists(), reason="local fixture not built")
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma_all")]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((FIX / "MANIFEST.json").read_text())


def test_manifest_hashes_match(manifest):
    for name, h in manifest["sha256"].items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == h, name


@pytest.mark.parametrize("excerpt", ["movement", "train"])
def test_frame_map_matches_excerpt(manifest, excerpt):
    from vicon2mano.core.loader import csv_frame_numbers
    m = pd.read_csv(FIX / f"frame_map_{excerpt}.csv")
    src = manifest["sources"][excerpt]
    assert len(m) == src["rows"]
    assert (m.row == np.arange(len(m))).all()
    assert (m.source_row == src["first_row"] + m.row).all()
    assert (csv_frame_numbers(str(FIX / f"{excerpt}.csv")) == m.source_frame).all()


def test_reviewed_rows_map_to_source(manifest):
    r, m = pd.read_csv(FIX / "reviewed_labels.csv"), pd.read_csv(FIX / "frame_map_movement.csv")
    assert len(r)
    assert (m.set_index("row").loc[r.row, "source_row"].to_numpy() == r.source_row.to_numpy()).all()
    assert sorted(r.source_row.unique()) == manifest["reviewed_source_rows"]


@pytest.mark.parametrize("excerpt", ["movement", "train"])
def test_excerpt_equals_source(manifest, excerpt):
    from vicon2mano.core.loader import load_csv
    src = Path(manifest["sources"][excerpt]["source"])
    if not src.exists():
        pytest.skip("data root not available")
    m = pd.read_csv(FIX / f"frame_map_{excerpt}.csv")
    X, names = load_csv(str(FIX / f"{excerpt}.csv"))[:2]
    S, sn = load_csv(str(src))[:2]
    S = S[m.source_row][:, [[n.split(":")[-1] for n in sn].index(n) for n in names]]
    np.testing.assert_allclose(X, S, atol=1e-4, equal_nan=True)


def test_gap_cuts_are_observed():
    from vicon2mano.core.loader import load_csv
    X, names = load_csv(str(FIX / "movement.csv"))[:2]
    for c in json.loads((FIX / "gap_mask.json").read_text()):
        assert np.isfinite(X[c["first"]:c["first"] + c["frames"], names.index(c["marker"])]).all(), c


def test_rigid_fill_regression(tmp_path):
    import make_fixture
    exp = json.loads((FIX / "expected.json").read_text())
    got = make_fixture.rigid_check(FIX, tmp_path)
    assert got.keys() == exp["rigid_fill_mm"].keys()
    for k, e in exp["rigid_fill_mm"].items():
        assert got[k]["filled"] == e["filled"], k
        if e["median"] is not None:
            assert abs(got[k]["median"] - e["median"]) <= exp["tolerance_mm"], k
