"""Load Vicon marker data from .c3d or .csv files.

Returns arrays of shape (T, N, 3) — T frames, N markers, XYZ in millimetres.
"""

from __future__ import annotations

import numpy as np


def load_c3d(path: str) -> tuple[np.ndarray, list[str], float]:
    """Read a Vicon .c3d file.

    Returns:
        markers: (T, N, 3) float32, millimetres
        labels:  list of N marker names
        fps:     capture frame rate
    """
    import c3d

    with open(path, "rb") as fh:
        reader = c3d.Reader(fh)
        fps = reader.header.frame_rate
        labels = [s.strip() for s in reader.point_labels]
        frames = []
        for _, points, _ in reader.read_frames():
            # points: (N, 5) — x y z err cam; take xyz
            frames.append(points[:, :3])

    markers = np.stack(frames, axis=0).astype(np.float32)  # (T, N, 3)
    return markers, labels, fps


def load_csv(path: str, *, mm: bool = True) -> tuple[np.ndarray, list[str]]:
    """Read a Vicon-exported CSV (Nexus format).

    Two layouts are auto-detected:

    Two-row header (device names + axis labels):
        Frame, Sub Frame, Marker0_X, Marker0_Y, Marker0_Z, Marker1_X, ...

    Single-row wide header (trajectories export):
        _Frame, _Sub Frame, Marker0_X, _Y, _Z, Marker1_X, _Y, _Z, ...

    Missing coordinates (blank cells) become NaN.

    Args:
        path: path to CSV file
        mm:   True if coordinates are already in millimetres; False → metres→mm

    Returns:
        markers: (T, N, 3) float32
        labels:  list of N marker names
    """
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    # Wide format: a single header row with "<Marker>_X" cells
    wide_idx = next(
        (i for i, r in enumerate(rows[:20])
         if any(c.strip().endswith("_X") and len(c.strip()) > 2 for c in r)),
        None,
    )
    if wide_idx is not None:
        return _parse_wide_rows(rows, wide_idx, mm=mm)

    # Two-row format: find the axis-label row (cells X/Y/Z)
    try:
        header_row = next(
            i for i, r in enumerate(rows)
            if any(c.strip() in ("X", "Y", "Z") for c in r)
        )
    except StopIteration:
        raise ValueError(
            f"Unrecognised CSV layout in {path}: no '<Marker>_X' header cells "
            "and no X/Y/Z axis-label row found."
        ) from None
    name_row = rows[header_row - 1]

    # Build label list from the name row (every 3 columns after Frame/SubFrame)
    labels: list[str] = []
    col_start = 2  # skip Frame, SubFrame
    raw_cols = name_row[col_start:]
    for i in range(0, len(raw_cols), 3):
        labels.append(raw_cols[i].strip())

    data_rows = rows[header_row + 1 :]
    frames = []
    for row in data_rows:
        if not row or not row[0].strip().isdigit():
            continue
        vals = [float(v) if v.strip() else np.nan for v in row[col_start:]]
        xyz = np.array(vals, dtype=np.float32).reshape(-1, 3)
        frames.append(xyz)

    markers = np.stack(frames, axis=0)  # (T, N, 3)
    if not mm:
        markers *= 1000.0
    return markers, labels


def _parse_wide_rows(
    rows: list[list[str]],
    header_idx: int,
    *,
    mm: bool,
) -> tuple[np.ndarray, list[str]]:
    """Parse the single-header wide layout (Marker_X, _Y, _Z triplets)."""
    header = rows[header_idx]
    cols = [
        (c.strip()[:-2], j)
        for j, c in enumerate(header)
        if c.strip().endswith("_X") and len(c.strip()) > 2
    ]
    labels = [name for name, _ in cols]

    frames = []
    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip().lstrip("-").isdigit():
            continue
        frame = np.full((len(cols), 3), np.nan, dtype=np.float32)
        for k, (_, j) in enumerate(cols):
            for axis in range(3):
                if j + axis < len(row):
                    v = row[j + axis].strip()
                    if v:
                        try:
                            frame[k, axis] = float(v)
                        except ValueError:
                            pass
        frames.append(frame)

    if not frames:
        raise ValueError("Wide CSV contained no data rows.")

    markers = np.stack(frames, axis=0)  # (T, N, 3)
    if not mm:
        markers *= 1000.0
    return markers, labels
