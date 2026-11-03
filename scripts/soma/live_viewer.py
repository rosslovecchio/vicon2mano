#!/usr/bin/env python3
"""Live 3-D player (three.js) for a movement fit: rotatable mesh + markers, played in real time.

Mesh vertices come from the fit (animate_mesh.forward_vertices), markers from the CSV
of the fitted frames; both are embedded as base64 float32, and the page picks the frame from the clock, so a fit of
every n-th frame of a ``--hz`` recording plays at real speed.
"""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/mano")]
import smplx
from animate_mesh import forward_vertices, resolve_model_path
from vicon2mano.core.loader import load_csv
from vicon2mano.core.dataset import MANO_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--csv", required=True, help="markers of the fitted frames (mm)")
ap.add_argument("--out", required=True)
ap.add_argument("--hz", type=float, default=200.0, help="capture rate of the source recording")
ap.add_argument("--title", default="")
ap.add_argument("--flags", default=None, help="diag.py's flags.json: flagged frames on the timeline, marker status colours")
a = ap.parse_args()

fit = np.load(a.fit)
model = smplx.create(str(resolve_model_path(str(MANO_DIR), "right")), model_type="mano", is_rhand=True, use_pca=False,
                     flat_hand_mean=True, batch_size=1)
V = forward_vertices(model, fit["global_orient"], fit["hand_pose"], fit["transl"], fit["betas"],
                     float(fit["scale"]) if "scale" in fit else 1.0) * 1000.0           # (T, 778, 3) mm
X, names = load_csv(a.csv)[:2]
names = [n.split(":")[-1] for n in names]
keep = [i for i, n in enumerate(names) if not n.startswith("Forearm")]
X, names = X[:, keep], [names[i] for i in keep]
T = min(len(V), len(X))
V, X = V[:T], X[:T]
c = np.nanmean(X.reshape(-1, 3), 0)
V, X = (V - c).astype(np.float32), (X - c).astype(np.float32)
fr = fit["frames_fitted"][:T]
dt = float(np.median(np.diff(fr))) / a.hz if T > 1 else 1 / a.hz                  # seconds per fitted frame
b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
att = {str(n): int(v) for n, v in zip(fit["names"], fit["att"])} if "att" in fit else {}
data = dict(T=T, nv=V.shape[1], nm=X.shape[1], dt=dt, frames=fr.tolist(), names=names, title=a.title, hz=a.hz,
            att=[att.get(n, -1) for n in names], faces=b64(np.asarray(model.faces, np.uint32)), V=b64(V), M=b64(X))
st, reasons = np.zeros((T, len(names)), np.uint8), [[] for _ in range(T)]
if a.flags:                                                                   # status: 0 observed 1 filled 2 missing 3 suspect
    fl = json.loads(Path(a.flags).read_text())
    order = [str(n) for n in fit["names"]]
    S_ = np.array(fl["status"], np.uint8)[:T]
    st = np.stack([S_[:, order.index(n)] if n in order else np.zeros(T, np.uint8) for n in names], 1)
    reasons = fl["reasons"][:T]
data.update(status=b64(st), reasons=reasons)

html = """<!doctype html><html><head><meta charset="utf-8"><title>TITLE</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#f4f5f7; --fg:#1f2933; --panel:#ffffffd9; --accent:#2563eb; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171c; --fg:#e5e7eb; --panel:#1f242cd9; --accent:#60a5fa; } }
html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; overflow:hidden; }
#bar { position:fixed; left:12px; right:12px; bottom:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;
       background:var(--panel); padding:8px 12px; border-radius:10px; box-shadow:0 2px 10px #0003; }
#bar input[type=range] { flex:1; min-width:160px; accent-color:var(--accent); }
button, select { font:inherit; padding:4px 10px; border-radius:6px; border:1px solid #8885; background:transparent; color:inherit; cursor:pointer; }
#info { position:fixed; left:12px; top:10px; background:var(--panel); padding:6px 10px; border-radius:8px; }
#flag { position:fixed; right:12px; top:10px; max-width:46%; background:#dc2626e6; color:#fff; padding:6px 10px; border-radius:8px; display:none; }
#strip { position:fixed; left:12px; right:12px; bottom:64px; height:10px; width:calc(100% - 24px); border-radius:4px; background:var(--panel); cursor:pointer; }
#legend { position:fixed; left:12px; top:44px; background:var(--panel); padding:4px 10px; border-radius:8px; font-size:12px; }
#legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 10px; vertical-align:-1px; }
label { display:flex; gap:4px; align-items:center; }
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script></head>
<body><div id="info"></div><div id="flag"></div><canvas id="strip"></canvas>
<div id="legend">markers:<i style="background:#1b3a5c"></i>observed<i style="background:#f59e0b"></i>gap-filled<i style="background:#dc2626"></i>suspect &nbsp;|&nbsp; red ticks: flagged frames</div>
<div id="bar"><button id="prev" title="previous flagged frame">&#9664; flag</button><button id="play">Play</button><button id="next" title="next flagged frame">flag &#9654;</button><input id="t" type="range" min="0" value="0">
<span id="time"></span><select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
<label><input id="lab" type="checkbox" checked>labels</label><label><input id="lines" type="checkbox" checked>attachments</label>
<label><input id="wire" type="checkbox">wire</label></div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = DATA;
const dec = (s, T) => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new T(b.buffer); };
const V = dec(D.V, Float32Array), M = dec(D.M, Float32Array), F = dec(D.faces, Uint32Array), ST = dec(D.status, Uint8Array);
const COL = [new THREE.Color(0x1b3a5c), new THREE.Color(0xf59e0b), new THREE.Color(0x888888), new THREE.Color(0xdc2626)];
const flagged = D.reasons.map((r, i) => r.length ? i : -1).filter(i => i >= 0);
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio); document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(dark ? 0x14171c : 0xf4f5f7);
const cam = new THREE.PerspectiveCamera(35, 1, 1, 5000); cam.position.set(0, -350, 250); cam.up.set(0, 0, 1);
const ctl = new OrbitControls(cam, renderer.domElement); ctl.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(200, -300, 400); scene.add(sun);
const geo = new THREE.BufferGeometry();
const pos = new THREE.BufferAttribute(new Float32Array(D.nv * 3), 3); geo.setAttribute("position", pos);
geo.setIndex(new THREE.BufferAttribute(F, 1));
const mat = new THREE.MeshStandardMaterial({ color: 0xe8a0a0, roughness: 0.75, side: THREE.DoubleSide, flatShading: true });
const mesh = new THREE.Mesh(geo, mat); scene.add(mesh);
const mk = new THREE.InstancedMesh(new THREE.SphereGeometry(3.2, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff }), D.nm);
scene.add(mk);
const lg = new THREE.BufferGeometry(); const lp = new THREE.BufferAttribute(new Float32Array(D.nm * 6), 3); lg.setAttribute("position", lp);
const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x16a34a })); scene.add(lines);
const tags = D.names.map(n => { const c = document.createElement("canvas"); c.width = 128; c.height = 32;
  const g = c.getContext("2d"); g.font = "bold 22px system-ui"; g.fillStyle = dark ? "#e5e7eb" : "#1b3a5c"; g.fillText(n, 2, 24);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(32, 8, 1); scene.add(s); return s; });
const m4 = new THREE.Matrix4(), slider = document.getElementById("t"); slider.max = D.T - 1;
let k = -1;
function show(i) {
  if (i === k) return; k = i;
  pos.array.set(V.subarray(i * D.nv * 3, (i + 1) * D.nv * 3)); pos.needsUpdate = true; geo.computeVertexNormals();
  for (let j = 0; j < D.nm; j++) {
    const o = (i * D.nm + j) * 3, x = M[o], y = M[o + 1], z = M[o + 2], ok = Number.isFinite(x);
    m4.makeTranslation(ok ? x : 1e5, ok ? y : 1e5, ok ? z : 1e5); mk.setMatrixAt(j, m4); mk.setColorAt(j, COL[ST[i * D.nm + j]]);
    tags[j].position.set(x, y, z + 7); tags[j].visible = ok && lab.checked;
    const a = D.att[j], q = a >= 0 ? (i * D.nv + a) * 3 : -1;
    lp.array.set(ok && q >= 0 ? [x, y, z, V[q], V[q + 1], V[q + 2]] : [0, 0, 0, 0, 0, 0], j * 6);
  }
  mk.instanceMatrix.needsUpdate = true; mk.instanceColor.needsUpdate = true; lp.needsUpdate = true;
  const fb = document.getElementById("flag"); fb.style.display = D.reasons[i].length ? "block" : "none";
  fb.innerHTML = D.reasons[i].map(r => "&bull; " + r).join("<br>");
  slider.value = i;
  document.getElementById("time").textContent = `${(i * D.dt).toFixed(2)} s  (frame ${D.frames[i]})`;
}
const lab = document.getElementById("lab");
lab.onchange = () => { const i = k; k = -1; show(i); };
document.getElementById("lines").onchange = e => lines.visible = e.target.checked;
document.getElementById("wire").onchange = e => mat.wireframe = e.target.checked;
let playing = false, t0 = 0, i0 = 0;
const btn = document.getElementById("play"), speed = document.getElementById("speed");
const start = () => { playing = true; t0 = performance.now(); i0 = k; btn.textContent = "Pause"; };
btn.onclick = () => playing ? (playing = false, btn.textContent = "Play") : start();
speed.onchange = () => { if (playing) start(); };
slider.oninput = () => { show(+slider.value); if (playing) start(); };
const jump = d => { const c = d > 0 ? flagged.find(f => f > k) : [...flagged].reverse().find(f => f < k);
  if (c !== undefined) { playing = false; btn.textContent = "Play"; show(c); } };
document.getElementById("prev").onclick = () => jump(-1); document.getElementById("next").onclick = () => jump(1);
const strip = document.getElementById("strip");
function drawStrip() { const w = strip.clientWidth; strip.width = w; strip.height = 10; const g = strip.getContext("2d");
  g.fillStyle = "#dc2626"; for (const f of flagged) g.fillRect(Math.floor(f / D.T * w), 0, Math.max(1, w / D.T), 10); }
strip.onclick = e => { show(Math.min(D.T - 1, Math.floor(e.offsetX / strip.clientWidth * D.T))); if (playing) start(); };
addEventListener("resize", drawStrip); drawStrip();
addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); btn.click(); } });
document.getElementById("info").textContent = `${D.title}  —  ${D.T} fitted frames, ${D.hz} Hz source, every ${Math.round(D.dt * D.hz)}th frame, real-time playback`;
function resize() { renderer.setSize(innerWidth, innerHeight); cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); }
addEventListener("resize", resize); resize(); show(0);
renderer.setAnimationLoop(now => {
  if (playing) { const i = i0 + Math.floor((now - t0) / 1000 * +speed.value / D.dt);
    if (i >= D.T) { playing = false; btn.textContent = "Play"; show(D.T - 1); } else show(i); }
  ctl.update(); renderer.render(scene, cam);
});
</script></body></html>"""
html = html.replace("TITLE", a.title or "Hand fit player").replace("DATA", json.dumps(data))
Path(a.out).write_text(html, encoding="utf-8")
print(f"saved {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB), {T} frames, {dt * 1000:.0f} ms each")
