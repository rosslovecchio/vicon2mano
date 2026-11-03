#!/usr/bin/env python3
"""Player comparing one trial's labellings beside cam1: Vicon labels, SOMA labels (no fill), learned gap fill.

Uses only existing outputs (nothing is re-fitted or re-encoded):
  cam1          results/video/<P>/<Trial>_<cond>/cam1.mp4 + cam1_frames.csv + sync.json
  Vicon labels  the raw Vicon CSV named in sync.json
  SOMA labels   results/soma/fullset/<P>/<slug>/soma_labelled.csv, flags from its flags.csv (flag_frames.py)
  learned fill  results/soma/real_fill_w01/movement/<P>/<slug>/soma_labelled_filled.csv (+ .npz fill status),
                flags from the fit's flags.json (frame_diagnostics.py; fitted every 10th frame, each frame
                takes the grade of the nearest fitted one)
2x2 grid: cam1, Vicon labels, SOMA labels with their fitted mesh, learned fill with its fitted mesh.
The marker panes share one view (drag to rotate) and the meshes are drawn live from the fits' vertices,
at the fits' frames (every 10th Vicon frame, 20 Hz). Filled markers are hollow, markers on a flagged
finger get an orange ring, and a yellow/red border means partly/not usable (grades as in
scripts/soma/mesh_video.py). The sync strip and offset controls are left out.

    python scripts/soma/labels_player.py results/video/P16/Trial2_HOI
    -> results/video/P16/Trial2_HOI/labels_player.html
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
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "soma"), str(REPO / "scripts" / "mano"), str(REPO / "scripts" / "shared")]

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path, help="results/video/<P>/<Trial>_<cond>, already synced")
    a = ap.parse_args()
    vdir = a.video_dir.resolve()
    p, trial = vdir.parent.name, vdir.name
    sync = json.loads((vdir / "sync.json").read_text())
    with open(vdir / "cam1_frames.csv") as f:
        cam = cam_summary("cam1", [(int(r["frame_number"]), float(r["timestamp_ms"]), int(r["pts_ms"]), float(r["motion"]))
                                   for r in csv.DictReader(f)])

    X, labels, rate = load_trajectories(Path(sync["vicon_csv"]))
    panes = [markers_pane(X, labels, rate, title="Vicon labels")]

    soma = soma_fit_dir(p, trial)
    soma_fit, rows, _, _, _, grade, note = load_fit(soma)  # fullset: every frame, flag_frames.py grades
    model = am.load_model(str(MANO_DIR), "right", int(soma_fit["hand_pose"].shape[1]))
    Xs, ls = load_csv(str(soma / "soma_labelled.csv"))
    g, n = np.zeros(len(Xs), int), [""] * len(Xs)
    for r, gg, nn in zip(rows, grade, note):
        g[r], n[r] = gg, nn
    panes.append(markers_pane(Xs, ls, rate, title="SOMA labels, no fill + fitted mesh",
                              status=np.where(np.isfinite(Xs).all(-1), 0, 2), grade=g, note=n))
    panes[-1]["mesh"] = mesh_pane(model, soma_fit, rows, np.arange(0, len(rows), 10))  # 20 Hz, like the learned-fill fit

    fill = soma_fit_dir(p, trial, "real_fill_w01/movement")
    fill_fit, rows, _, names, st_fit, grade, note = load_fit(fill)  # fitted every 10th frame
    Xf, lf = load_csv(str(fill / "soma_labelled_filled.csv"))
    lf = [x.split(":")[-1] for x in lf]
    how = np.load(fill / "soma_labelled_filled.npz")
    how = how["how"][:, [list(how["names"]).index(x) for x in lf]]
    status = np.where(np.isfinite(Xf).all(-1), np.where(how == 0, 0, 1), 2).astype(np.uint8)
    near = np.clip(np.searchsorted(rows, np.arange(len(Xf)) - 5), 0, len(rows) - 1)  # nearest fitted frame
    cols = [lf.index(x) for x in names]
    status[:, cols] = np.where(st_fit[near] == 3, 3, status[:, cols])
    panes.append(markers_pane(Xf, lf, rate, title="Learned gap fill + fitted mesh", status=status,
                              grade=grade[near], note=[note[k] for k in near]))
    panes[-1]["mesh"] = mesh_pane(model, fill_fit, rows, np.arange(len(rows)))

    t, speed = hand_speed(X, labels, rate)
    step = int(round(rate / 50))
    meta = {"title": f"{p} {trial.replace('_', ' ')} - labellings", "trigger_ms": sync.get("trigger_ms"), "cams": [cam],
            "sync": sync, "emg": None, "markers": panes,
            "hide_sync": True, "cols": 2, "faces": base64.b64encode(np.asarray(model.faces, np.uint16).tobytes()).decode("ascii"),
            "speed50": np.round(speed[: len(speed) // step * step].reshape(-1, step).mean(axis=1), 1).tolist()}
    html = (REPO / "scripts/shared/realsense_player.html.in").read_text(encoding="utf-8").replace("/*META*/null", json.dumps(meta))
    (vdir / "labels_player.html").write_text(html, encoding="utf-8")
    print("->", vdir / "labels_player.html")


if __name__ == "__main__":
    main()
