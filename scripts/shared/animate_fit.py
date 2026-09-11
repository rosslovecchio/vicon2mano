#!/usr/bin/env python3
"""Side-by-side animation: raw Vicon recording vs fitted/corrected MANO hand.

Left panel  — the recording *before* processing: the raw, unlabelled marker
cloud exactly as loaded from the Nexus CSV (gaps appear as missing dots).
Right panel — the recording *after* fitting and correction: the fitted MANO
skeleton, the markers the assignment matched to each joint, and dashed
joint→marker links.

Both panels share the same (per-frame) camera framing, centred on the fitted
hand, so the two views are directly comparable.

Two modes:

* default — timelapse: ``--n-out`` frames evenly sampled over the whole
  recording (or ``--start``/``--end`` range).
* ``--speed S`` — real-time playback at S× speed of a ``--clip-s``-second
  clip; the clip with the most hand motion is chosen automatically unless
  ``--start`` is given. Requires the capture rate (``--rate``, default 100 Hz).

Output format follows the ``--out`` extension: ``.mp4`` (imageio-ffmpeg),
``.gif`` (Pillow), or ``.html`` (self-contained player — frames embedded as
base64 JPEGs with play/pause and a scrub slider; opens in any browser).

Usage
-----
python scripts/animate_fit.py \\
    --npz   data/Pxh8/Hands_only_Left/mano_fit_left.npz \\
    --csv   data/Pxh8/Hands_only_Left/trajectories_synced.csv \\
    --out   vicon2mano/eval/fit/fit_anim_before_after.mp4 \\
    --speed 1.5 [--rate 100] [--clip-s 30] [--fps 24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_fit import FINGER_COLORS, FINGER_LEGEND, FINGER_LEGEND_COLORS, SKELETON
from vicon2mano.core.loader import load_csv

import matplotlib.patches as mpatches


def _draw_before(ax, markers_t, ctr, span):
    """Raw, unlabelled marker cloud — what the optical capture gives us."""
    finite = np.isfinite(markers_t).all(axis=1)
    if finite.any():
        ax.scatter(*markers_t[finite].T, c="#34495e", s=24, depthshade=False)
    ax.set_title(f"Before — raw Vicon markers "
                 f"({finite.sum()}/{len(markers_t)} visible)", fontsize=10)
    _frame_axes(ax, ctr, span)


def _draw_after(ax, hands_t, markers_t, ctr, span):
    """Fitted MANO skeleton(s) + the markers the assignment matched.

    hands_t: list of (joints_t (21, 3), assignment (21,)) — one per hand.
    """
    finite = np.isfinite(markers_t).all(axis=1)
    matched = sorted({int(mi) for _, assignment in hands_t
                      for mi in assignment if mi >= 0 and finite[mi]})
    if matched:
        ax.scatter(*markers_t[matched].T, c="lightgray", s=24, depthshade=False)

    for joints_t, assignment in hands_t:
        for j0, j1 in SKELETON:
            ax.plot(*zip(joints_t[j0], joints_t[j1]),
                    color=FINGER_COLORS[j1], linewidth=2)
        for j in range(21):
            ax.scatter(*joints_t[j], c=FINGER_COLORS[j], s=40,
                       depthshade=False, edgecolors="white", linewidths=0.5)
        for j, mi in enumerate(assignment):
            if mi >= 0 and finite[mi]:
                ax.plot(*zip(joints_t[j], markers_t[mi]), color="gray",
                        linewidth=0.7, linestyle=":", alpha=0.6)

    n = len(hands_t)
    ax.set_title("After — fitted MANO skeleton%s + matched markers"
                 % ("s" if n > 1 else ""), fontsize=10)
    _frame_axes(ax, ctr, span)


def _frame_axes(ax, ctr, span):
    """span: (3,) per-axis half-extents.  The box aspect mirrors the spans so
    millimetres stay isotropic without forcing a cube around separated
    hands."""
    ax.set_xlim(ctr[0] - span[0], ctr[0] + span[0])
    ax.set_ylim(ctr[1] - span[1], ctr[1] + span[1])
    ax.set_zlim(ctr[2] - span[2], ctr[2] + span[2])
    ax.set_box_aspect(tuple(span))
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.set_zlabel("Z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)


def _most_dynamic_window(joints_mm: np.ndarray, win: int) -> int:
    """Start frame of the `win`-frame window with the highest mean joint speed."""
    speed = np.linalg.norm(np.diff(joints_mm, axis=0), axis=-1).mean(axis=1)
    cumsum = np.concatenate([[0.0], np.cumsum(speed)])
    sums = cumsum[win:] - cumsum[:-win]
    return int(np.argmax(sums))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vicon2mano — before vs after fit</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center; }}
  img {{ max-width:100vw; height:auto; }}
  #bar {{ display:flex; gap:10px; align-items:center; padding:8px;
          width:95vw; max-width:900px; }}
  input[type=range] {{ flex:1; }}
  button {{ font-size:16px; padding:4px 14px; }}
  #goto {{ width:70px; font-size:14px; }}
</style></head><body>
<img id="view">
<div id="bar">
  <button id="btn" onclick="toggle()">&#9208;</button>
  <button id="prev" onclick="step(-1)" title="previous frame">&#9198;</button>
  <button id="next" onclick="step(1)" title="next frame">&#9197;</button>
  <input id="slider" type="range" min="0" max="{nmax}" value="0"
         oninput="seek(this.value)">
  <span id="lbl"></span>
  <input id="goto" type="number" min="1" max="{nframes}" placeholder="go to #"
         onkeydown="if(event.key==='Enter') gotoFrame(this.value)">
  <button onclick="gotoFrame(document.getElementById('goto').value)">Go</button>
</div>
<script>
const frames = [{frames}];
const fps = {fps};
let i = 0, timer = null;
const img = document.getElementById("view");
const slider = document.getElementById("slider");
const lbl = document.getElementById("lbl");
function show(k) {{
  i = ((k % frames.length) + frames.length) % frames.length;
  img.src = "data:image/jpeg;base64," + frames[i];
  slider.value = i;
  lbl.textContent = (i + 1) + "/" + frames.length;
}}
function play() {{
  timer = setInterval(() => show(i + 1), 1000 / fps);
  document.getElementById("btn").innerHTML = "&#9208;";
}}
function pause() {{
  clearInterval(timer); timer = null;
  document.getElementById("btn").innerHTML = "&#9654;";
}}
function toggle() {{ timer ? pause() : play(); }}
function seek(v) {{ pause(); show(+v); }}
function step(d) {{ pause(); show(i + d); }}
function gotoFrame(v) {{
  const n = parseInt(v, 10);
  if (!isNaN(n)) {{ pause(); show(n - 1); }}
}}
document.addEventListener("keydown", (e) => {{
  if (document.activeElement === document.getElementById("goto")) return;
  if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowRight") step(1);
  else if (e.key === " ") {{ e.preventDefault(); toggle(); }}
}});
show(0); play();
</script></body></html>
"""


def _write_html(fig, update, n_frames, fps, dpi, out: Path):
    """Render every frame to an in-memory JPEG and embed in an HTML player."""
    import base64
    import io

    encoded = []
    for k in range(n_frames):
        update(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=dpi)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    print(f"[animate_fit] writing HTML player ({n_frames} frames) …")
    out.write_text(_HTML_TEMPLATE.format(
        nmax=n_frames - 1,
        nframes=n_frames,
        fps=fps,
        frames=",".join(f'"{e}"' for e in encoded),
    ))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, nargs="+",
                    help="FitResult .npz file(s) — pass two to render both hands")
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--out", required=True, help="output .mp4 or .gif path")
    ap.add_argument("--start", type=int, default=-1,
                    help="first source frame (-1 = 0 in timelapse mode, "
                         "most-dynamic clip in --speed mode)")
    ap.add_argument("--end", type=int, default=-1,
                    help="last source frame (-1 = end of recording)")
    ap.add_argument("--n-out", type=int, default=240,
                    help="timelapse mode: frames in the output animation")
    ap.add_argument("--fps", type=int, default=20, help="output frame rate")
    ap.add_argument("--dpi", type=int, default=90,
                    help="render resolution (lower = smaller file)")
    ap.add_argument("--speed", type=float, default=None,
                    help="real-time playback at this multiple of recording "
                         "speed (e.g. 1.5); switches from timelapse to clip mode")
    ap.add_argument("--rate", type=float, default=100.0,
                    help="capture rate of the recording in Hz (clip mode)")
    ap.add_argument("--clip-s", type=float, default=30.0,
                    help="clip mode: seconds of recording to animate")
    args = ap.parse_args()

    hands = []                               # [(joints (T,21,3) mm, assignment)]
    for path in args.npz:
        print(f"[animate_fit] loading {path}")
        data = np.load(path)
        hands.append((data["joints"] * 1000.0, data["assignment"]))

    print(f"[animate_fit] loading markers from {args.csv}")
    markers, _ = load_csv(args.csv)          # (T, N, 3) mm
    T = min(len(markers), *(len(j) for j, _ in hands))
    # all hands' joints side by side — drives window choice and camera
    joints_mm = np.concatenate([j[:T] for j, _ in hands], axis=1)
    end = T if args.end < 0 else min(args.end, T)

    if args.speed is not None:
        # Real-time clip: advance speed×rate source frames per output second.
        step = args.speed * args.rate / args.fps
        win = min(int(args.clip_s * args.rate), end)
        start = args.start
        if start < 0:
            start = _most_dynamic_window(joints_mm[:end], win)
            print(f"[animate_fit] most dynamic {args.clip_s:.0f} s window "
                  f"starts at frame {start}")
        win = min(win, end - start)
        frames = (start + np.arange(int(win / step)) * step).astype(int)
        print(f"[animate_fit] clip mode: {args.speed}×, {args.rate:.0f} Hz, "
              f"{len(frames)} output frames "
              f"({len(frames) / args.fps:.1f} s of video)")
    else:
        start = max(args.start, 0)
        frames = np.linspace(start, end - 1, min(args.n_out, end - start))
        frames = np.unique(frames.astype(int))
    print(f"[animate_fit] animating {len(frames)} frames "
          f"from source range [{frames[0]}, {frames[-1]}]")

    # Per-frame camera centre follows the fitted joints.  The zoom adapts
    # per frame (smoothed over ~2 s so it doesn't pulse) — with two hands a
    # fixed span would have to cover their worst-case separation, leaving
    # the hands tiny whenever they come together.
    centres = joints_mm[frames].mean(axis=1)                       # (F, 3)
    rel = joints_mm[frames] - centres[:, None, :]
    ext = np.nanmax(np.abs(rel), axis=1)                           # (F, 3)
    k = max(1, int(args.fps * 2)) | 1
    kernel = np.ones(k) / k
    smooth = np.stack([
        np.convolve(np.pad(ext[:, a], k // 2, mode="edge"),
                    kernel, mode="valid")
        for a in range(3)
    ], axis=1)
    spans = np.maximum(ext, smooth) * 1.25 + 10.0                  # (F, 3) mm

    fig = plt.figure(figsize=(11, 5.5))
    fig.suptitle("vicon2mano — recording before vs after fit & correction",
                 fontsize=12, fontweight="bold")
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    handles = [
        mpatches.Patch(color="#34495e", label="Raw marker"),
        mpatches.Patch(color="lightgray", label="Matched marker"),
    ] + [mpatches.Patch(color=FINGER_LEGEND_COLORS[i], label=FINGER_LEGEND[i])
         for i in range(len(FINGER_LEGEND))]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    def update(i):
        t = frames[i]
        ctr = centres[i]
        for ax in (ax_l, ax_r):
            ax.cla()
        _draw_before(ax_l, markers[t], ctr, spans[i])
        _draw_after(ax_r, [(j[t], a) for j, a in hands], markers[t], ctr,
                    spans[i])
        tag = (f"t = {t / args.rate:6.2f} s · {args.speed}× speed"
               if args.speed is not None else f"frame {t}/{T}")
        fig.suptitle(
            f"vicon2mano — before vs after fit & correction   ({tag})",
            fontsize=12, fontweight="bold")
        if i and i % 40 == 0:
            print(f"  rendered {i}/{len(frames)} frames")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".html":
        _write_html(fig, update, len(frames), args.fps, args.dpi, out)
        plt.close(fig)
        print(f"[animate_fit] done → {out} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
        return

    anim = FuncAnimation(fig, update, frames=len(frames))
    if out.suffix == ".mp4":
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = \
            imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=args.fps, bitrate=2500)
    else:
        writer = PillowWriter(fps=args.fps)
    print(f"[animate_fit] writing {out} at {args.fps} fps …")
    anim.save(out, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"[animate_fit] done → {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
