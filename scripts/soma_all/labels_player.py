#!/usr/bin/env python3
"""Player comparing one trial's labellings beside cam1: Vicon labels, SOMA labels (no fill), learned gap fill.

Uses only existing outputs (nothing is re-fitted or re-encoded):
  cam1          results/video/<P>/<Trial>_<cond>/cam1.mp4 + cam1_frames.csv + sync.json
  Vicon labels  the raw Vicon CSV named in sync.json
  SOMA labels   results/soma/fullset/<P>/<slug>/soma_labelled.csv, flags from its flags.csv (flag_frames.py)
  learned fill  results/soma/real_fill_w01/movement/<P>/<slug>/soma_labelled_filled.csv (+ .npz fill status),
                flags from the fit's flags.json (frame_diagnostics.py; fitted every 10th frame, each frame
                takes the grade of the nearest fitted one)
The marker panes share one view (drag to rotate). Filled markers are hollow, markers on a flagged
finger get an orange ring, and a yellow/red border means partly/not usable (grades as in
scripts/soma_all/mesh_video.py).

    python scripts/soma_all/labels_player.py results/video/P16/Trial2_HOI
    -> labels_player.html: cam1 + the three marker panes, with the sync strip and offset controls

    python scripts/soma_all/labels_player.py results/video/P16/Trial2_HOI --mesh
    -> labels_player_mesh.html: 2x2 grid (cam1, Vicon labels, SOMA + its fitted mesh, learned fill + its
       fitted mesh), no sync controls. The meshes are drawn live from the fits' vertices at the fits'
       frames (every 10th Vicon frame, 20 Hz), so the page is ~66 MB.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "soma_all"), str(REPO / "scripts" / "mano"), str(REPO / "scripts" / "shared")]

import animate_mesh as am  # noqa: E402
from mesh_video import load_fit  # noqa: E402
from realsense_player import cam_summary, markers_pane  # noqa: E402
from vicon2mano.core.dataset import MANO_DIR, soma_fit_dir  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.core.video_sync import hand_speed, load_trajectories  # noqa: E402


def mesh_pane(model, fit: dict, rows: np.ndarray, keep: np.ndarray) -> dict:
    """Mesh vertices (int16 mm, (n, 778, 3)) of the fit's frames `keep`, with their Vicon rows, for the player."""
    v = np.round(am.fit_vertices_mm(model, fit, keep)).astype(np.int16)
    return {"frames": rows[keep].tolist(), "verts": base64.b64encode(v.tobytes()).decode("ascii")}


def soma_pane(soma: Path, rate: float, mesh: bool):
    """(marker pane, fit, rows) of the SOMA-labelled, unfilled trial; graded by flag_frames.py, one fit row per frame."""
    fit, rows, _, _, _, grade, note = load_fit(soma)
    X, ls = load_csv(str(soma / "soma_labelled.csv"))
    g, n = np.zeros(len(X), int), [""] * len(X)
    for r, gg, nn in zip(rows, grade, note):
        g[r], n[r] = gg, nn
    pane = markers_pane(X, ls, rate, title="SOMA labels, no fill" + (" + fitted mesh" if mesh else ""),
                        status=np.where(np.isfinite(X).all(-1), 0, 2), grade=g, note=n)
    return pane, fit, rows


def fill_pane(fill: Path, rate: float, mesh: bool):
    """(marker pane, fit, rows) of the learned-fill trial; graded by the fit's flags.json (fitted every 10th frame)."""
    fit, rows, _, names, st_fit, grade, note = load_fit(fill)
    X, ls = load_csv(str(fill / "soma_labelled_filled.csv"))
    ls = [x.split(":")[-1] for x in ls]
    how = np.load(fill / "soma_labelled_filled.npz")
    how = how["how"][:, [list(how["names"]).index(x) for x in ls]]
    status = np.where(np.isfinite(X).all(-1), np.where(how == 0, 0, 1), 2).astype(np.uint8)
    near = np.clip(np.searchsorted(rows, np.arange(len(X)) - 5), 0, len(rows) - 1)  # nearest fitted frame
    cols = [ls.index(x) for x in names]
    status[:, cols] = np.where(st_fit[near] == 3, 3, status[:, cols])
    pane = markers_pane(X, ls, rate, title="Learned gap fill" + (" + fitted mesh" if mesh else " (fit flags)"),
                        status=status, grade=grade[near], note=[note[k] for k in near])
    return pane, fit, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path, help="results/video/<P>/<Trial>_<cond>, already synced")
    ap.add_argument("--mesh", action="store_true", help="2x2 page with the fitted meshes, without sync controls")
    a = ap.parse_args()
    vdir = a.video_dir.resolve()
    p, trial = vdir.parent.name, vdir.name
    soma, fill = soma_fit_dir(p, trial), soma_fit_dir(p, trial, "real_fill_w01/movement")
    soma = soma if soma and (soma / "fit.npz").exists() else None
    if not (soma or fill):
        sys.exit(f"no SOMA fit for {vdir}")
    sync = json.loads((vdir / "sync.json").read_text())
    with open(vdir / "cam1_frames.csv") as f:
        cam = cam_summary("cam1", [(int(r["frame_number"]), float(r["timestamp_ms"]), int(r["pts_ms"]), float(r["motion"]))
                                   for r in csv.DictReader(f)])

    X, labels, rate = load_trajectories(Path(sync["vicon_csv"]))
    panes, model = [markers_pane(X, labels, rate, title="Vicon labels")], None
    for fit_dir, build, stride in ((soma, soma_pane, 10), (fill, fill_pane, 1)):  # SOMA mesh: 20 Hz, like the fill fit
        if not fit_dir:
            continue
        pane, fit, rows = build(fit_dir, rate, a.mesh)
        if a.mesh:
            model = model or am.load_model(str(MANO_DIR), "right", int(fit["hand_pose"].shape[1]))
            pane["mesh"] = mesh_pane(model, fit, rows, np.arange(0, len(rows), stride))
        panes.append(pane)

    t, speed = hand_speed(X, labels, rate)
    step = int(round(rate / 50))
    meta = {"title": f"{p} {trial.replace('_', ' ')} - labellings", "trigger_ms": sync.get("trigger_ms"), "cams": [cam],
            "sync": sync, "emg": None, "markers": panes,
            **({"hide_sync": True, "cols": 2, "faces": base64.b64encode(np.asarray(model.faces, np.uint16).tobytes()).decode("ascii")}
               if a.mesh else {}),
            "speed50": np.round(speed[: len(speed) // step * step].reshape(-1, step).mean(axis=1), 1).tolist()}
    html = (REPO / "scripts/shared/realsense_player.html.in").read_text(encoding="utf-8").replace("/*META*/null", json.dumps(meta))
    out = vdir / ("labels_player_mesh.html" if a.mesh else "labels_player.html")
    out.write_text(html, encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
