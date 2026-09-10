#!/usr/bin/env python3
"""Animate the before/after effect of strategyGMM's relabelling.

Side-by-side 3-D scatter animation of P7/Trial1_handsonly's marker cloud
across the Ring1<->Pinky1 investigation window (see
validate_gmm_labeler.py's module docstring for what that window actually
contains -- a rolling mislabelling cascade, anchor-gated here via
scripts/_gmm_validation_common.py). Left panel is the recording as
labelled on disk ("before"), right panel is after applying strategyGMM's
per-frame identity mapping ("after"). Ring1 is always drawn orange and
Pinky1 always drawn purple, and frames the cascade itself flags as having
untrustworthy anchors are marked -- a frame where "before" and "after"
look the same despite that mark means the GMM (correctly) declined to
guess from garbage-in local coordinates, not that nothing happened.

Usage
-----
python scripts/animate_gmm_relabel.py
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from _gmm_validation_common import (   # noqa: E402
    OUT_DIR, load_p7_ring1_pinky1, fit_and_relabel_window,
)

# Same self-contained scrubbing player as scripts/animate_fit.py's
# _HTML_TEMPLATE -- kept in sync deliberately so both tools' output "feels
# like the same product" (play/pause, prev/next, scrub, goto-frame,
# keyboard shortcuts, frames embedded as base64 JPEG, no external files).
_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>strategyGMM — before vs after relabelling</title>
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


def _write_html(fig, update, n_frames, fps, dpi, out):
    """Render every frame to an in-memory JPEG and embed in an HTML player.

    Identical technique to scripts/animate_fit.py's _write_html: no
    external assets, works offline, scrubs instantly since every frame is
    already decoded in the page.
    """
    encoded = []
    for k in range(n_frames):
        update(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=dpi)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    print(f"[animate_gmm_relabel] writing HTML player ({n_frames} frames) ...")
    out.write_text(_HTML_TEMPLATE.format(
        nmax=n_frames - 1,
        nframes=n_frames,
        fps=fps,
        frames=",".join(f'"{e}"' for e in encoded),
    ))


def render_before_after(
    ctx, before: np.ndarray, after: np.ndarray, lo: int,
    is_swapped: np.ndarray, out_stem, title: str,
    *, truth: np.ndarray | None = None, fps: int = 10, dpi: int = 90,
    max_frames: int = 150,
) -> None:
    """Side-by-side before/after 3-D animation -> ``.gif`` and ``.html``.

    ``before``/``after`` are ``(W, N, 3)`` windows of marker positions
    starting at global frame ``lo``; ``is_swapped`` is the per-frame
    verdict driving the panel labels. Pass ``truth`` (the known injected
    swap mask) to also label each frame as a hit/miss/false alarm --
    only meaningful for the injected-swap validation, where ground truth
    exists.
    """
    idx_ring, idx_pinky = ctx.marker_idxs
    anchor_bad = ~ctx.anchor_valid[lo:lo + len(before)]

    stride = max(1, len(before) // max_frames)
    frame_ids = np.arange(0, len(before), stride)

    all_pts = np.concatenate([before.reshape(-1, 3), after.reshape(-1, 3)])
    finite = np.isfinite(all_pts).all(axis=1)
    mins, maxs = all_pts[finite].min(0), all_pts[finite].max(0)
    pad = 0.05 * (maxs - mins).max()

    fig = plt.figure(figsize=(11, 6.5))
    ax_before = fig.add_subplot(121, projection="3d")
    ax_after = fig.add_subplot(122, projection="3d")

    def style(ax, panel_title, color="black"):
        ax.set_title(panel_title, fontsize=10, color=color)
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    def scatter_frame(ax, pts):
        ax.cla()
        others = np.delete(pts, ctx.marker_idxs, axis=0)
        finite_o = np.isfinite(others).all(axis=1)
        ax.scatter(*others[finite_o].T, color="#bbbbbb", s=12, depthshade=False)
        if np.isfinite(pts[idx_ring]).all():
            ax.scatter(*pts[idx_ring], color="#e67e22", s=70, depthshade=False)
        if np.isfinite(pts[idx_pinky]).all():
            ax.scatter(*pts[idx_pinky], color="#8e44ad", s=70, depthshade=False)

    def update(i):
        k = int(frame_ids[i])
        global_frame = lo + k
        tags = []
        if truth is not None and truth[k]:
            tags.append("swap injected")
        if anchor_bad[k]:
            tags.append("anchors untrustworthy")
        flag = f"  [{', '.join(tags)}]" if tags else ""
        scatter_frame(ax_before, before[k])
        style(ax_before, f"BEFORE (as labelled) — frame {global_frame}{flag}")

        if truth is not None:
            ok = bool(is_swapped[k]) == bool(truth[k])
            verdict = ("corrected" if is_swapped[k] else "left as-is")
            mark = "OK" if ok else ("MISS" if truth[k] else "FALSE ALARM")
            colour = "#1e8449" if ok else "#c0392b"
            after_title = (f"AFTER (strategyGMM) — frame {global_frame}"
                           f"  [{verdict} — {mark}]")
        else:
            colour = "black"
            after_title = (f"AFTER (strategyGMM) — frame {global_frame}"
                           + ("  [corrected]" if is_swapped[k] else ""))
        scatter_frame(ax_after, after[k])
        style(ax_after, after_title, color=colour)
        fig.suptitle(f"{title}   (Ring1 = orange, Pinky1 = purple)",
                     fontsize=12, fontweight="bold")
        return []

    out_gif = out_stem.with_suffix(".gif")
    print(f"Rendering {len(frame_ids)} frames to {out_gif} ...")
    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=80)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    print(f"Saved {out_gif}")

    out_html = out_stem.with_suffix(".html")
    _write_html(fig, update, len(frame_ids), fps=fps, dpi=dpi, out=out_html)
    print(f"Saved {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")
    plt.close(fig)


def main() -> None:
    ctx = load_p7_ring1_pinky1()
    idx_ring, idx_pinky = ctx.marker_idxs
    markers = ctx.markers

    lo, hi = max(0, ctx.run_lo - 150), min(markers.shape[0], ctx.run_lo + 150)
    is_swapped, init_frame_global = fit_and_relabel_window(ctx, lo, hi)
    print(f"GMM verdict: {is_swapped.sum()}/{len(is_swapped)} frames in [{lo},{hi}) "
          f"marked swapped (init_frame global={init_frame_global})")

    eval_markers = markers[lo:hi]
    after = eval_markers.copy()
    for k in np.flatnonzero(is_swapped):
        after[k, idx_ring], after[k, idx_pinky] = (
            eval_markers[k, idx_pinky].copy(), eval_markers[k, idx_ring].copy())

    render_before_after(
        ctx, before=eval_markers, after=after, lo=lo, is_swapped=is_swapped,
        out_stem=OUT_DIR / "p7_ring1_pinky1_before_after",
        title="strategyGMM — P7/Trial1_handsonly swap episode, before vs after",
    )


if __name__ == "__main__":
    main()
