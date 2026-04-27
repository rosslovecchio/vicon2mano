import io
import textwrap

import numpy as np
import pytest

from vicon2mano.loader import load_csv


def _make_csv(n_markers=22, n_frames=5) -> str:
    """Generate a minimal Nexus-style CSV string."""
    names_row = ["", ""] + [f"MARK{i}" for i in range(n_markers) for _ in range(3)]
    axis_row  = ["Frame", "Sub Frame"] + ["X", "Y", "Z"] * n_markers
    rows = [",".join(names_row), ",".join(axis_row)]
    for f in range(n_frames):
        vals = [str(f), "0"] + [f"{f*i:.2f}" for i in range(n_markers * 3)]
        rows.append(",".join(vals))
    return "\n".join(rows)


def test_load_csv_shape(tmp_path):
    csv_text = _make_csv(n_markers=22, n_frames=10)
    p = tmp_path / "test.csv"
    p.write_text(csv_text)
    markers, labels = load_csv(str(p))
    assert markers.shape == (10, 22, 3)
    assert len(labels) == 22


def test_load_csv_labels(tmp_path):
    csv_text = _make_csv(n_markers=3, n_frames=2)
    p = tmp_path / "labels.csv"
    p.write_text(csv_text)
    _, labels = load_csv(str(p))
    assert labels == ["MARK0", "MARK1", "MARK2"]
