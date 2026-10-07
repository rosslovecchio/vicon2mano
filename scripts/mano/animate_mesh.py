#!/usr/bin/env python3
"""HTML animation: fitted MANO mesh surface + Vicon marker overlay, at real
elapsed-time playback speed.

First version of this script used Plotly's live Mesh3d frame animation,
which looked right in principle but played back noticeably slower than
real time in the browser: redrawing a 778-vertex/1538-face mesh every
frame is too expensive for the browser to keep up with the requested
per-frame duration, so Plotly silently falls behind instead of honouring
it. Same lesson ``scripts/shared/animate_fit.py`` already learned for the
skeleton-only animation: render every frame to a static image *offline*
(no per-frame interactivity cost at playback time) and drive it with a
fixed-rate JS timer, so playback speed is decoupled from render cost
entirely. This script now follows that exact pattern -- ``_write_html``
imported directly from ``animate_fit.py`` rather than reimplemented.

Usage
-----
python scripts/mano/animate_mesh.py \\
    --npz results/mano/P10_Trial2_handsonly/mano_fit_right.npz \\
    --csv "<raw Vicon CSV>" \\
    --out results/mano/P10_Trial2_handsonly/eval/mesh_animation.html \\
    --n-out 1500 --fps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SHARED_DIR = REPO_ROOT / "scripts" / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

# chumpy (an smplx dependency) uses removed numpy type aliases -- same
# compat shim as vicon2mano.core.fitter / export_joint_angles.py.
import inspect as _inspect
import numpy as _np
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
for _attr, _builtin in {"int": int, "float": float, "bool": bool, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _builtin)

from vicon2mano.core.loader import load_csv

_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vicon2mano — MANO mesh animation</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center; }}
  img {{ max-width:100vw; height:auto; }}
  #plotlyMesh {{ width:min(100vw, 900px); height:700px; display:none; }}
  #bar {{ display:flex; gap:10px; align-items:center; padding:8px;
          width:95vw; max-width:900px; flex-wrap:wrap; }}
  input[type=range] {{ flex:1; }}
  button {{ font-size:16px; padding:4px 14px; }}
  #goto {{ width:70px; font-size:14px; }}
  #inspectBtn.active {{ background:#2c7; color:#111; }}
  #hint {{ font-size:12px; color:#999; max-width:900px; text-align:center; padding:0 8px; }}
</style></head><body>
<img id="view">
<div id="plotlyMesh"></div>
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
  <button id="inspectBtn" onclick="toggleInspect()">Inspect (rotate)</button>
</div>
<div id="hint">Playback (▶) uses pre-rendered frames for guaranteed real-time
speed. "Inspect" pauses and swaps in a live, rotatable 3-D view of the
current frame — drag to orbit, scroll to zoom. Pressing ▶ again exits
Inspect and resumes real-time playback.</div>
<script>{plotlyjs}</script>
<script>
const frames = [{frames}];
const fps = {fps};
let i = 0, timer = null, inspecting = false, plotlyInited = false;
const img = document.getElementById("view");
const meshDiv = document.getElementById("plotlyMesh");
const slider = document.getElementById("slider");
const lbl = document.getElementById("lbl");
const inspectBtn = document.getElementById("inspectBtn");

// Per-frame mesh vertices + marker positions, decoded once from base64
// float32 buffers (not shipped as per-frame JSON -- far more compact).
function decodeFloat32(b64) {{
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let j = 0; j < bin.length; j++) buf[j] = bin.charCodeAt(j);
  return new Float32Array(buf.buffer);
}}
const nVerts = {n_verts};
const nMarkers = {n_markers};
const vertsFlat = decodeFloat32("{verts_b64}");       // frames.length * nVerts * 3
const markersFlat = decodeFloat32("{markers_b64}");   // frames.length * nMarkers * 3
const facesI = {faces_i};
const facesJ = {faces_j};
const facesK = {faces_k};
const markerLabels = {marker_labels};
const center = {center};
const half = {half};

function frameSlice(flat, nPts, k) {{
  const off = k * nPts * 3;
  const x = new Array(nPts), y = new Array(nPts), z = new Array(nPts);
  for (let p = 0; p < nPts; p++) {{
    x[p] = flat[off + p * 3]; y[p] = flat[off + p * 3 + 1]; z[p] = flat[off + p * 3 + 2];
  }}
  return {{x, y, z}};
}}

function updatePlotly(k) {{
  const v = frameSlice(vertsFlat, nVerts, k);
  const m = frameSlice(markersFlat, nMarkers, k);
  if (!plotlyInited) {{
    const meshTrace = {{
      type: "mesh3d", x: v.x, y: v.y, z: v.z,
      i: facesI, j: facesJ, k: facesK,
      color: "#e8a0a0", opacity: 0.95, flatshading: false, hoverinfo: "skip",
      lighting: {{ambient: 0.5, diffuse: 0.8, specular: 0.2, roughness: 0.6}},
      lightposition: {{x: 200, y: 200, z: 400}},
    }};
    const markerTrace = {{
      type: "scatter3d", mode: "markers", x: m.x, y: m.y, z: m.z,
      marker: {{size: 4, color: "#2c3e50"}},
      text: markerLabels, hoverinfo: "text",
    }};
    Plotly.newPlot(meshDiv, [meshTrace, markerTrace], {{
      scene: {{
        xaxis: {{range: [center[0]-half, center[0]+half], title: "X (mm)"}},
        yaxis: {{range: [center[1]-half, center[1]+half], title: "Y (mm)"}},
        zaxis: {{range: [center[2]-half, center[2]+half], title: "Z (mm)"}},
        aspectmode: "cube",
      }},
      uirevision: "constant",  // preserve camera across updates
      margin: {{t: 10, b: 10}},
    }}, {{displaylogo: false}});
    plotlyInited = true;
  }} else {{
    Plotly.restyle(meshDiv, {{x: [v.x, m.x], y: [v.y, m.y], z: [v.z, m.z]}}, [0, 1]);
  }}
}}

function enterInspect() {{
  pause();
  inspecting = true;
  img.style.display = "none";
  meshDiv.style.display = "block";
  inspectBtn.classList.add("active");
  updatePlotly(i);
}}
function exitInspect() {{
  inspecting = false;
  meshDiv.style.display = "none";
  img.style.display = "block";
  inspectBtn.classList.remove("active");
}}
function toggleInspect() {{ inspecting ? exitInspect() : enterInspect(); }}

function show(k) {{
  i = ((k % frames.length) + frames.length) % frames.length;
  img.src = "data:image/jpeg;base64," + frames[i];
  slider.value = i;
  lbl.textContent = (i + 1) + "/" + frames.length;
  if (inspecting) updatePlotly(i);  // manual stepping/scrubbing also updates the 3-D view
}}
function play() {{
  if (inspecting) exitInspect();  // Inspect only applies while paused -- never during playback
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


def _write_html(frames_b64, n_frames, playback_fps, verts_mm, markers_mm, faces,
                 marker_labels, center, half, out: Path) -> None:
    """Write the hybrid player: baked JPEGs for real-time Play/scrub, plus a
    live Plotly Mesh3d of the current frame for free rotation while paused
    ("Inspect"). Embeds plotly.js inline (``get_plotlyjs()``) so the page
    stays self-contained, same as this repo's other Plotly-based outputs.
    """
    import base64
    import json
    import plotly.offline as pyo

    verts_b64 = base64.b64encode(verts_mm.astype(np.float32).tobytes()).decode("ascii")
    markers_b64 = base64.b64encode(markers_mm.astype(np.float32).tobytes()).decode("ascii")

    out.write_text(_HTML_TEMPLATE.format(
        nmax=n_frames - 1,
        nframes=n_frames,
        fps=playback_fps,
        frames=",".join(f'"{e}"' for e in frames_b64),
        plotlyjs=pyo.get_plotlyjs(),
        n_verts=verts_mm.shape[1],
        n_markers=markers_mm.shape[1],
        verts_b64=verts_b64,
        markers_b64=markers_b64,
        faces_i=json.dumps(faces[:, 0].tolist()),
        faces_j=json.dumps(faces[:, 1].tolist()),
        faces_k=json.dumps(faces[:, 2].tolist()),
        marker_labels=json.dumps(list(marker_labels)),
        center=json.dumps([float(c) for c in center]),
        half=float(half),
    ), encoding="utf-8")


def resolve_model_path(mano_dir: str, side: str) -> Path:
    model_path = Path(mano_dir)
    if model_path.is_dir() and not (model_path / "mano").exists():
        side_str = "RIGHT" if side == "right" else "LEFT"
        pkl = model_path / f"MANO_{side_str}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"MANO weights not found at {pkl}")
        model_path = pkl
    return model_path


def forward_vertices(model, global_orient, hand_pose, transl, betas) -> np.ndarray:
    """No-grad forward pass -> (n, 778, 3) mesh vertices in metres."""
    import torch
    n = global_orient.shape[0]
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            hand_pose=torch.tensor(hand_pose, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(n, -1),
            return_verts=True,
        )
    return out.vertices.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="FitResult .npz from vicon2mano")
    ap.add_argument("--csv", required=True, help="source Vicon CSV (markers, mm)")
    ap.add_argument("--out", required=True, help="output .html path")
    ap.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--n-out", type=int, default=1500,
                     help="frames sampled into the animation")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--fps", type=float, default=200.0,
                     help="capture rate in Hz (read from the CSV's Trajectories header, "
                          "not assumed) -- used only to derive playback speed")
    ap.add_argument("--dpi", type=int, default=85, help="render resolution (lower = smaller file)")
    args = ap.parse_args()

    import smplx
    model_path = resolve_model_path(args.mano_dir, args.side)
    model = smplx.create(
        str(model_path), model_type="mano",
        is_rhand=(args.side == "right"), use_pca=True, num_pca_comps=6,
        flat_hand_mean=True, batch_size=1,
    )
    faces = model.faces  # (nf, 3) int

    print(f"[animate_mesh] loading {args.npz}")
    fit = np.load(args.npz)
    print(f"[animate_mesh] loading markers from {args.csv}")
    markers_mm, labels = load_csv(args.csv)
    labels = [l.split(":")[-1] for l in labels]

    T = min(fit["global_orient"].shape[0], markers_mm.shape[0])
    end = T if args.end < 0 else min(args.end, T)
    frame_idx = np.unique(np.linspace(args.start, end - 1, min(args.n_out, end - args.start)).astype(int))
    print(f"[animate_mesh] animating {len(frame_idx)} frames from source range "
          f"[{frame_idx[0]}, {frame_idx[-1]}]")

    # Real elapsed time per animation step, from the actual (roughly uniform,
    # after np.unique) gap between sampled source frames -- the playback fps
    # passed to _write_html's fixed-rate JS timer, so total playback time
    # matches the trial's real recording duration regardless of render cost.
    avg_step_frames = (frame_idx[-1] - frame_idx[0]) / max(len(frame_idx) - 1, 1)
    playback_fps = args.fps / avg_step_frames
    print(f"[animate_mesh] {args.fps:.0f} Hz source, {avg_step_frames:.1f} frames/step "
          f"-> {playback_fps:.1f} fps playback (real-time)")

    print("[animate_mesh] running forward pass for sampled frames...")
    verts_m = forward_vertices(
        model,
        fit["global_orient"][frame_idx],
        fit["hand_pose"][frame_idx],
        fit["transl"][frame_idx],
        fit["betas"],
    )
    verts_mm = verts_m * 1000.0  # metres -> mm, to match raw marker scale
    sampled_markers = markers_mm[frame_idx]  # (n, N, 3) mm

    # Fixed camera range across the whole animation, computed once (same
    # pattern as scripts/gmm/relabel_trial.py's build_figure), so the hand
    # doesn't visually jump as the window re-centres frame to frame.
    flat_markers = sampled_markers.reshape(-1, 3)
    all_pts = np.concatenate([
        verts_mm.reshape(-1, 3),
        flat_markers[np.isfinite(flat_markers).all(-1)],
    ], axis=0)
    center = np.nanmean(all_pts, axis=0)
    half = np.nanmax(np.linalg.norm(all_pts - center, axis=-1)) * 1.1

    print("[animate_mesh] rendering frames...")
    import base64
    import io

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    frames_b64 = []
    for k in range(len(frame_idx)):
        ax.clear()
        v = verts_mm[k]
        # Known limitation, not yet fixed: mplot3d depth-sorts the mesh
        # surface (Poly3DCollection) and the marker scatter (PathCollection)
        # independently via separate per-artist heuristics, and regularly
        # gets the ordering wrong -- an opaque mesh can hide markers that are
        # geometrically in front of it. zorder does NOT fix this (mplot3d
        # doesn't use it for cross-artist depth). Tried alpha<1 as a
        # mitigation -- worse, not better: translucency also breaks
        # mplot3d's self-sorting of the mesh's OWN overlapping triangles
        # (fingers self-occlude heavily in 2D projection), producing visible
        # dark hatching across the surface. Reverted to opaque. Real fix
        # needs a renderer with actual depth-buffering (e.g. baking frames
        # via Plotly+Kaleido instead of matplotlib) -- not done here.
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=faces,
                         color="#e8a0a0", edgecolor="none", shade=True,
                         antialiased=False, alpha=0.95)
        m = sampled_markers[k]
        finite = np.isfinite(m).all(axis=-1)
        if finite.any():
            ax.scatter(*m[finite].T, c="#2c3e50", s=22, depthshade=False)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"frame {int(frame_idx[k])}", fontsize=10)
        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Y (mm)", fontsize=7)
        ax.set_zlabel("Z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=args.dpi)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close(fig)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[animate_mesh] writing hybrid HTML player ({len(frame_idx)} frames)...")
    _write_html(frames_b64, len(frame_idx), playback_fps, verts_mm, sampled_markers,
                faces, labels, center, half, out_path)
    print(f"[animate_mesh] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
