#!/usr/bin/env python3
"""Render a window of a trial's fitted MANO mesh to MP4 and add it to that trial's RealSense player.

Same frames as scripts/mano/animate_mesh.py (mesh + SOMA-labelled markers), but
every Vicon frame of the window, encoded as video on the Vicon clock instead of
JPEGs baked into HTML (10k frames would be ~500 MB of HTML). The player places
it beside the cameras via the trial's sync.json (docs/video_sync.md).

    python scripts/soma/mesh_video.py results/video/P16/Trial2_HOI --step 4     # whole trial at 50 Hz
    python scripts/soma/mesh_video.py results/video/P16/Trial2_HOI --n 10000   # 10k consecutive frames
    -> results/video/P16/Trial2_HOI/{mesh.mp4, mesh_frames.csv, player.html}

The fit is the learned-gap-fill run (results/soma/real_fill_w01/movement, every 10th frame,
filled values weighted 0.1) where it exists, else results/soma/fullset (every frame, no fill).
Each rendered frame gets a grade in mesh_frames.csv, which the player shows as a yellow or
red border and timeline strip:
  2 (red, unusable)  the hand turned too fast in one step, or >= half the markers are missing
                     or on a flagged finger (frame_diagnostics.py; fullset: flag_frames.py's 2)
  1 (yellow, partial) some finger flagged (pose past the natural limits, too-fast joint) or a
                     marker missing (fullset: flag 1)
Markers are drawn dark when observed, blue when gap-filled, orange when on a flagged finger.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "mano"), str(REPO / "scripts" / "shared")]

import animate_mesh as am  # noqa: E402
from realsense_player import rebuild  # noqa: E402
from vicon2mano.core.dataset import MANO_DIR, soma_fit_dir  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.core.viz import marker_bone_pairs  # noqa: E402


STATUS_COLORS = np.array(["#2c3e50", "#0ea5e9", "#2c3e50", "#f59e0b"])  # observed, filled, missing (not drawn), suspect


def _render(args):
    return am.render_mesh_frames(*args)


def load_fit(fit_dir: Path):
    """(fit arrays, fitted CSV rows, markers (F, N, 3), labels, status (F, N), grade (F,), note (F,)) per fitted frame.

    status: 0 observed, 1 gap-filled, 2 missing, 3 on a flagged finger.
    """
    batches = sorted(glob.glob(str(fit_dir / "*_every*" / "fit.npz")), key=lambda f: int(Path(f).parent.name.split("-")[0]))
    if batches:  # real_fill_w01: 10k-frame batches, each with frame_diagnostics.py's flags.json
        fits = [np.load(f) for f in batches]
        fit = {k: np.concatenate([f[k] for f in fits]) for k in ("global_orient", "hand_pose", "transl", "frames_fitted")}
        fit.update(betas=fits[0]["betas"], scale=fits[0]["scale"])
        names = [str(n) for n in fits[0]["names"]]
        flags = [json.loads((Path(f).parent / "flags.json").read_text()) for f in batches]
        status = np.concatenate([np.array(fl["status"], np.uint8) for fl in flags])
        reasons = [r for fl in flags for r in fl["reasons"]]
        X, labels = load_csv(str(fit_dir / "soma_labelled_filled.csv"))
    else:        # fullset: one fit of every frame, flag_frames.py's flags.csv
        fit = dict(np.load(fit_dir / "fit.npz"))
        X, labels = load_csv(str(fit_dir / "soma_labelled.csv"))
        names = [n.split(":")[-1] for n in labels]
        with open(fit_dir / "flags.csv") as f:
            fl = {int(r["frame"]): r for r in csv.DictReader(f)}
    labels = [n.split(":")[-1] for n in labels]
    rows = fit["frames_fitted"].astype(int)
    X = X[rows][:, [labels.index(n) for n in names]]
    if batches:
        bad = (status >= 2).sum(1)
        grade = np.where([any("hand turned" in x for x in r) for r in reasons] | (bad >= len(names) / 2), 2,
                         np.where((bad > 0) | np.array([bool(r) for r in reasons]), 1, 0))
        note = ["; ".join(r + ([f"{(status[t] == 2).sum()} markers missing"] if (status[t] == 2).any() else []))
                for t, r in enumerate(reasons)]
    else:
        status = np.where(np.isfinite(X).all(-1), 0, 2).astype(np.uint8)
        grade = np.array([int(fl[r]["flag"]) if r in fl else 0 for r in rows])
        note = [", ".join([k.split("_")[0] + " doubt" for k in ("label_signal", "fit_signal", "angle_signal")
                           if fl.get(r, {}).get(k) == "True"] or [f"only {fl[r]['n_markers']} markers observed"] * (r in fl))
                for r in rows]
    return fit, rows, X, names, status, grade, note


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path, help="results/video/<P>/<Trial>_<cond>, already synced")
    ap.add_argument("--fit-dir", type=Path)
    ap.add_argument("--start", type=int, default=1, help="first Vicon frame (1-based)")
    ap.add_argument("--n", type=int, help="Vicon frames to cover (default: the rest of the trial)")
    ap.add_argument("--step", type=int, default=1, help="render every step-th Vicon frame (4 = 50 Hz at 200 Hz)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=85)
    ap.add_argument("--side", choices=["right", "left"], default="right")
    a = ap.parse_args()
    vdir = a.video_dir.resolve()
    sync = json.loads((vdir / "sync.json").read_text())
    fit_dir = a.fit_dir or next((d for d in (soma_fit_dir(vdir.parent.name, vdir.name, "real_fill_w01/movement"),
                                             soma_fit_dir(vdir.parent.name, vdir.name)) if d), None)
    if fit_dir is None:
        sys.exit(f"no SOMA fit for {vdir} - pass --fit-dir")
    fit, rows, markers, labels, status, grade, note = load_fit(fit_dir)
    keep = np.flatnonzero((rows >= a.start - 1) & (rows < a.start - 1 + (a.n or rows[-1] + 1)))[::a.step]
    idx = rows[keep]  # CSV row k = Vicon frame k+1
    print(f"Vicon frames {idx[0] + 1}-{idx[-1] + 1} ({len(idx)}) from {fit_dir}; "
          f"grade 0/1/2: {[int((grade[keep] == g).sum()) for g in (0, 1, 2)]}")

    model = am.load_model(str(MANO_DIR), a.side, int(fit["hand_pose"].shape[1]))
    verts = am.fit_vertices_mm(model, fit, keep)
    center = verts.mean(axis=1)  # follow the hand; the cameras show where it is in the room
    half = 1.15 * np.linalg.norm(verts - center[:, None], axis=-1).max()
    bones = marker_bone_pairs(labels)
    chunks = np.array_split(np.arange(len(idx)), a.workers * 4)
    colors = STATUS_COLORS[status[keep]]
    jobs = [(verts[c], markers[keep[c]], model.faces, bones, center[c], half, idx[c] + 1, a.dpi, colors[c]) for c in chunks]

    rate = sync["vicon_rate"]
    out_rows = []
    with av.open(str(vdir / "mesh.mp4"), "w") as out, ProcessPoolExecutor(a.workers) as pool:
        st = None
        for c, jpegs in zip(chunks, pool.map(_render, jobs)):
            for k, jpg in zip(c, jpegs):
                img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
                img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]  # yuv420p needs even sizes
                if st is None:
                    st = out.add_stream("libx264", rate=int(rate), options={"crf": "23"})
                    st.width, st.height, st.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                    st.time_base = st.codec_context.time_base = Fraction(1, 1000)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = round(1000 * (idx[k] - idx[0]) / rate)
                out.mux(st.encode(frame))
                out_rows.append((idx[k] + 1, frame.pts, grade[keep[k]], note[keep[k]]))
            print(f"  {len(out_rows)}/{len(idx)}", flush=True)
        out.mux(st.encode())
    with open(vdir / "mesh_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vicon_frame", "pts_ms", "grade", "note"])
        w.writerows(out_rows)

    rebuild(vdir)  # the player picks up the mesh pane


if __name__ == "__main__":
    main()
