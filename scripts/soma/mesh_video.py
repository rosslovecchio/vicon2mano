#!/usr/bin/env python3
"""Render a window of a trial's fitted MANO mesh to MP4 and add it to that trial's RealSense player.

Same frames as scripts/mano/animate_mesh.py (mesh + SOMA-labelled markers), but
every Vicon frame of the window, encoded as video on the Vicon clock instead of
JPEGs baked into HTML (10k frames would be ~500 MB of HTML). The player places
it beside the cameras via the trial's sync.json (docs/video_sync.md).

    python scripts/soma/mesh_video.py results/video/P16/Trial2_HOI --step 4     # whole trial at 50 Hz
    python scripts/soma/mesh_video.py results/video/P16/Trial2_HOI --n 10000   # 10k consecutive frames
    -> results/video/P16/Trial2_HOI/{mesh.mp4, mesh_frames.csv, player.html}

--fit-dir defaults to the matching results/soma/fullset/<P>/<slug>.
"""
from __future__ import annotations

import argparse
import csv
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
from vicon2mano.core.dataset import MANO_DIR, _normalise_name  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.core.viz import marker_bone_pairs  # noqa: E402


def _render(args):
    return am.render_mesh_frames(*args)


def fit_dir_of(video_dir: Path) -> Path | None:
    """results/soma/fullset/<P>/<slug> of a video folder (fit slugs spell "Trial 1" as Trial_1)."""
    root = REPO / "results" / "soma" / "fullset" / video_dir.parent.name
    return next((d for d in root.glob("*") if _normalise_name(d.name) == _normalise_name(video_dir.name)), None)


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
    fit_dir = a.fit_dir or fit_dir_of(vdir)
    if fit_dir is None or not (fit_dir / "fit.npz").exists():
        sys.exit(f"no SOMA fit for {vdir} - pass --fit-dir")

    fit = np.load(fit_dir / "fit.npz")
    markers, labels = load_csv(str(fit_dir / "soma_labelled.csv"))
    labels = [n.split(":")[-1] for n in labels]
    end = min(len(fit["global_orient"]), len(markers), a.start - 1 + (a.n or len(markers)))
    idx = np.arange(a.start - 1, end, a.step)  # row k = Vicon frame k+1
    print(f"Vicon frames {idx[0] + 1}-{idx[-1] + 1} ({len(idx)}) from {fit_dir}")

    model = am.load_model(str(MANO_DIR), a.side, int(fit["hand_pose"].shape[1]))
    verts = am.fit_vertices_mm(model, fit, idx)
    center = verts.mean(axis=1)  # follow the hand; the cameras show where it is in the room
    half = 1.15 * np.linalg.norm(verts - center[:, None], axis=-1).max()
    bones = marker_bone_pairs(labels)
    chunks = np.array_split(np.arange(len(idx)), a.workers * 4)
    jobs = [(verts[c], markers[idx[c]], model.faces, bones, center[c], half, idx[c] + 1, a.dpi) for c in chunks]

    rate = sync["vicon_rate"]
    rows = []
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
                rows.append((idx[k] + 1, frame.pts))
            print(f"  {len(rows)}/{len(idx)}", flush=True)
        out.mux(st.encode())
    with open(vdir / "mesh_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vicon_frame", "pts_ms"])
        w.writerows(rows)

    rebuild(vdir)  # the player picks up the mesh pane


if __name__ == "__main__":
    main()
