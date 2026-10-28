#!/usr/bin/env python3
"""Fit a movement trial with a final static (static_final_fit.py): shape, scale and attachments held, per-frame pose.

Each frame starts from the static pose moved rigidly onto that frame's palm
markers (Kabsch), then chunks of frames are fitted together with the static's terms: marker on its attached vertex
(+standoff along the normal), never inside, joints inside the anatomical limits (finger twist ~0, PIP/DIP abduction,
DIP coupled to the PIP), plus frame-to-frame smoothness; --w-filled down-weights gap-filled values. Writes fit.npz / pred.npz / summary.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma"), str(REPO / "scripts/mano")]
import static_search as ss, static_pose_check as spc, fit_surface_markers as fsm
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.calibrate import static_cloud

ap = argparse.ArgumentParser()
ap.add_argument("--participant", default="P4")
ap.add_argument("--static", required=True, help="params.npz of static_final_fit.py")
ap.add_argument("--csv", required=True, help="labelled trial CSV")
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--first", type=int, default=0, help="first source frame")
ap.add_argument("--max-frames", type=int, default=0, help="source frame to stop before (0 = end)")
ap.add_argument("--step", type=int, default=1, help="fit every n-th frame; the fitted frames are also written as a CSV")
ap.add_argument("--standoff", type=float, default=3.0)
ap.add_argument("--margin", type=float, default=3.0)
ap.add_argument("--twist", type=float, default=4.0)
ap.add_argument("--hinge", type=float, default=2.0)
ap.add_argument("--couple", type=float, default=10.0)
ap.add_argument("--w-nat", type=float, default=1e-2)
ap.add_argument("--w-inside", type=float, default=100.0)
ap.add_argument("--cap", type=float, default=4.5, help="mm: hinge on |marker - attached vertex| above this")
ap.add_argument("--w-cap", type=float, default=30.0)
ap.add_argument("--w-smooth", type=float, default=1e-3)
ap.add_argument("--w-filled", type=float, default=1.0, help="weight of gap-filled values (gap_fill.py's .npz next to --csv)")
a = ap.parse_args()
p = a.participant

J = ss.Judge()
dev, model = J.dev, J.model
z = np.load(a.static)
mk = [str(n) for n in z["names"]]
att = torch.tensor(z["att"], device=dev)
sm = fsm.ScaledModel(model, dev)
with torch.no_grad():
    sm.log_scale.fill_(float(np.log(z["scale"])))
betas = torch.tensor(z["betas"], device=dev)
F = torch.tensor(np.asarray(model.faces, np.int64), device=dev)
B = torch.tensor(J.B, dtype=torch.float32, device=dev)

# limits as in the final static (fit bounds: margin inside, finger twist and PIP/DIP abduction tight)
lo, hi = (torch.tensor(np.radians(x), dtype=torch.float32, device=dev) for x in (J.lo, J.hi))
lo[:12, 2], hi[:12, 2] = -np.radians(a.twist), np.radians(a.twist)
HINGE = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] in "23"]
PIP = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] == "2"]
lo[HINGE, 1], hi[HINGE, 1] = -np.radians(a.hinge + a.margin - 0.5), np.radians(a.hinge + a.margin - 0.5)
lo, hi = lo + np.radians(a.margin), hi - np.radians(a.margin)
clo, chi = J.lo.copy(), J.hi.copy()                                           # the check: judge limits + the user's
clo[:12, 2], chi[:12, 2] = -5, 5
clo[HINGE, 1], chi[HINGE, 1] = -a.hinge, a.hinge


def fwd(go, hp, tr):
    o = sm(global_orient=go, hand_pose=hp, transl=tr, betas=betas[None].expand(len(go), -1), return_verts=True)
    V = o.vertices
    fn = torch.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]], dim=-1)
    N = torch.zeros_like(V).index_add_(1, F[:, 0], fn).index_add_(1, F[:, 1], fn).index_add_(1, F[:, 2], fn)
    return V, N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# static: rest wrist and the palm markers it was fitted to, for the rigid start of every frame
Xs, ns = load_csv(str(spc.sources(p)[p][0]))[:2]
cloud = dict(zip([n.split(":")[-1] for n in ns], static_cloud(Xs, 30)[0]))
go_s, hp_s, tr_s = (z[k][0] for k in ("global_orient", "hand_pose", "transl"))
with torch.no_grad():
    J0 = sm(global_orient=torch.zeros(1, 3, device=dev), hand_pose=torch.zeros(1, 45, device=dev), transl=torch.zeros(1, 3, device=dev),
            betas=betas[None]).joints[0, 0].cpu().numpy()
RIGID = [n for n in ("Palm1", "Palm2", "Palm3", "Index1", "Middle1", "Ring1", "Pinky1") if n in mk]
S_rigid = np.stack([cloud[n] for n in RIGID]) / 1000.0

X, names = load_csv(a.csv)[:2]
names = [n.split(":")[-1] for n in names]
X = X[:, [names.index(n) for n in mk]] / 1000.0                               # (T, 18, 3) metres, NaN = missing
frames = np.arange(len(X))[a.first:a.max_frames or None:a.step]
X = X[frames]
T = len(X)
ri = [mk.index(n) for n in RIGID]
GO, TR = np.zeros((T, 3), np.float32), np.zeros((T, 3), np.float32)
Rs = Rotation.from_rotvec(go_s)
last = (np.eye(3), np.zeros(3))
for t in range(T):
    ok = np.isfinite(X[t, ri]).all(1)
    if ok.sum() >= 3:                                                         # Kabsch: static palm -> this frame
        P0, P1 = S_rigid[ok], X[t, ri][ok]
        c0, c1 = P0.mean(0), P1.mean(0)
        U, _, Vt = np.linalg.svd((P0 - c0).T @ (P1 - c1))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        last = (R, c1 - R @ c0)
    R, t_ = last
    GO[t] = (Rotation.from_matrix(R) * Rs).as_rotvec()
    TR[t] = R @ (J0 + tr_s) + t_ - J0

Mt = torch.tensor(np.nan_to_num(X), dtype=torch.float32, device=dev)
seen = np.isfinite(X).all(-1).astype(np.float32)                              # (T, 18)
if a.w_filled != 1.0:
    filled = np.load(Path(a.csv).with_suffix(".npz"))["how"][:, [names.index(n) for n in mk]][frames] > 0
    seen[filled] *= a.w_filled
seen = torch.tensor(seen, device=dev)
HP, GOf, TRf = np.zeros((T, 45), np.float32), GO.copy(), TR.copy()
prev_hp = torch.tensor(hp_s, device=dev)
t0 = time.time()
for s in range(0, T, a.chunk):
    e = min(T, s + a.chunk)
    M, w = Mt[s:e], seen[s:e]
    hp = prev_hp[None].repeat(e - s, 1).clone().requires_grad_(True)
    go = torch.tensor(GO[s:e], device=dev, requires_grad=True)
    tr = torch.tensor(TR[s:e], device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [hp], "lr": 2e-2}, {"params": [go], "lr": 5e-3}, {"params": [tr], "lr": 2e-3}])
    for it in range(a.iters):
        opt.zero_grad()
        V, N = fwd(go, hp, tr)
        Va, Na = V[:, att], N[:, att]
        tgt = ((M - (Va + a.standoff * 1e-3 * Na)) ** 2).sum(-1)
        near = torch.cdist(M, V).argmin(-1).detach()
        Vn, Nn = torch.gather(V, 1, near[..., None].expand(-1, -1, 3)), torch.gather(N, 1, near[..., None].expand(-1, -1, 3))
        inside = torch.relu(-((M - Va) * Na).sum(-1)) + torch.relu(-((M - Vn) * Nn).sum(-1) + 5e-4)
        c = torch.einsum("tjk,jkc->tjc", hp.view(-1, 15, 3), B)
        nat = (torch.relu(c - hi) ** 2 + torch.relu(lo - c) ** 2).sum((1, 2))
        nat = nat + (torch.relu(c[:, [j + 1 for j in PIP], 0] - 2 / 3 * c[:, PIP, 0] - np.radians(a.couple)) ** 2).sum(1)
        smooth = (((hp[1:] - hp[:-1]) ** 2).sum(1) + ((go[1:] - go[:-1]) ** 2).sum(1)).mean() + ((tr[1:] - tr[:-1]) ** 2).sum(1).mean()
        cap = torch.relu((M - Va).norm(dim=-1) - a.cap * 1e-3) ** 2
        loss = ((tgt * w).sum() + a.w_inside * ((inside ** 2) * w).sum() + a.w_cap * (cap * w).sum()) / w.sum().clamp_min(1) \
            + a.w_nat * nat.mean() + a.w_smooth * smooth
        loss.backward()
        opt.step()
    HP[s:e], GOf[s:e], TRf[s:e] = (x.detach().cpu().numpy() for x in (hp, go, tr))
    prev_hp = hp.detach()[-1]
    print(f"frames {s}-{e} done ({time.time() - t0:.0f} s)", flush=True)

# per-frame check, same measures as the static
dist, excess, pred = np.full((T, len(mk)), np.nan, np.float32), np.zeros(T, np.float32), np.zeros((T, len(mk), 3), np.float32)
with torch.no_grad():
    for s in range(0, T, a.chunk):
        e = min(T, s + a.chunk)
        V, N = fwd(*(torch.tensor(x[s:e], device=dev) for x in (GOf, HP, TRf)))
        Va, Na = V[:, att], N[:, att]
        pred[s:e] = ((Va + a.standoff * 1e-3 * Na) * 1000).cpu().numpy()
        dist[s:e] = ((Mt[s:e] - Va).norm(dim=-1) * 1000).cpu().numpy()
c = np.degrees(np.einsum("tjk,jkc->tjc", HP.reshape(T, 15, 3), J.B))
ex = np.maximum(clo - c, c - chi).clip(0)
excess = ex.max((1, 2))
dist[~np.isfinite(X).all(-1)] = np.nan
mx = np.nanmax(dist, 1)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "fit.npz", global_orient=GOf, hand_pose=HP, transl=TRf, betas=z["betas"], scale=np.float32(z["scale"]),
         frames_fitted=frames, names=np.array(mk), att=z["att"])
np.savez(out / "pred.npz", pred_mm=pred, frames=frames, names=np.array(mk), dist_mm=dist, excess_deg=excess)
summ = dict(frames=T, seconds=round(time.time() - t0),
            max_dist_mm=dict(median=round(float(np.nanmedian(mx)), 2), p95=round(float(np.nanpercentile(mx, 95)), 2),
                             frames_under_5mm_pct=round(100 * float(np.mean(mx < 5)), 1)),
            per_marker_median_mm={n: round(float(np.nanmedian(dist[:, i])), 2) for i, n in enumerate(mk)},
            per_marker_p95_mm={n: round(float(np.nanpercentile(dist[:, i], 95)), 2) for i, n in enumerate(mk)},
            pose=dict(frames_within_limits_pct=round(100 * float(np.mean(excess <= 0.05)), 1),
                      excess_p95_deg=round(float(np.percentile(excess, 95)), 2), excess_max_deg=round(float(excess.max()), 2)))
lines = Path(a.csv).read_text().splitlines()                                  # the fitted frames as their own CSV
(out / "markers.csv").write_text("\n".join(lines[:1] + [lines[1 + f] for f in frames]) + "\n")
summ["source_frames"] = f"{frames[0]}..{frames[-1]} every {a.step}"
(out / "summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
