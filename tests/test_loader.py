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


def _make_wide_csv(n_frames=4) -> str:
    """Single-header wide layout: _Frame,_Sub Frame,Name_X,_Y,_Z,..."""
    names = ["Palm_Left2", "Index_Left1", "Thumb_Left1"]
    header = ["_Frame", "_Sub Frame"]
    for n in names:
        header += [f"{n}_X", "_Y", "_Z"]
    rows = [",".join(header)]
    for f in range(1, n_frames + 1):
        vals = [str(f), "0"]
        for m in range(len(names)):
            vals += [f"{f * 10 + m}.0", f"{f * 10 + m}.5", f"{f * 10 + m}.25"]
        rows.append(",".join(vals))
    return "\n".join(rows)


def test_load_wide_csv_shape_and_labels(tmp_path):
    p = tmp_path / "wide.csv"
    p.write_text(_make_wide_csv(n_frames=4))
    markers, labels = load_csv(str(p))
    assert markers.shape == (4, 3, 3)
    assert labels == ["Palm_Left2", "Index_Left1", "Thumb_Left1"]
    assert markers[0, 0, 0] == 10.0
    assert markers[3, 2, 1] == 42.5


def test_load_wide_csv_blank_cells_become_nan(tmp_path):
    text = _make_wide_csv(n_frames=2)
    lines = text.split("\n")
    # blank out the second marker's coords in frame 1
    cells = lines[1].split(",")
    cells[5] = cells[6] = cells[7] = ""
    lines[1] = ",".join(cells)
    p = tmp_path / "gaps.csv"
    p.write_text("\n".join(lines))
    markers, _ = load_csv(str(p))
    assert np.isnan(markers[0, 1]).all()
    assert np.isfinite(markers[1, 1]).all()
