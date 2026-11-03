#!/usr/bin/env python3
"""Final static fit: every marker within 5 mm of its attached vertex, the pose natural.

Starts from a static_search arm's static_params.npz (or an earlier output of this script, to polish it) and
alternates (a) choosing each marker's vertex among the dorsal vertices of the same bone within --radius mm of its
pin (for markers 2 and 3 of every digit, only vertices on the segment's dorsal midline, within --axis-tol mm of
the plane through its bone axis and dorsal direction; with --along, only the pin or its mesh neighbour straight
up or down the segment) and (b) Adam on pose, shape, scale and placement. The loss
holds each marker 3 mm out along its vertex normal, penalises distances above --cap, markers inside the skin or above non-dorsal skin, and joints outside the
anatomical limits (with --margin inside them, finger twist within --twist, PIP/DIP abduction within --hinge, DIP
flexion at most 2/3 of the PIP's plus --couple). No segment lengths, so the Judge's pose recovery reproduces the
mesh and its pose check agrees with the fit's own angles.

P4's accepted static (docs: soma_transfer/static_final/P4) came from two runs, the second polishing the first:

    python scripts/soma_all/static_final_fit.py --participant P4 --start fingers_m3e --out OUT1 --fix Index3,Middle3,Ring3,Pinky3,Thumb3
    python scripts/soma_all/static_final_fit.py --participant P4 --start OUT1 --rounds 3 --out OUT2 --fix Index3,Middle3,Ring3,Pinky3,Thumb3

Writes static.html (static_search's view format), diag.json (the Judge), params.npz and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma_all"), str(REPO / "scripts/mano")]
import fit_surface_markers as fsm  # noqa: E402
import static_pose_check as spc  # noqa: E402
import static_search as ss  # noqa: E402
from vicon2mano.core.fitter import mano_output_to_joints21  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402
from vicon2mano.strategies.soma.calibrate import static_cloud  # noqa: E402

HAND = ["Palm1", "Palm2", "Palm3"] + [f"{f}{k}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for k in (1, 2, 3)]
_PIP = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] == "2"]
_HINGE = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] in "23"]


def check_limits(J, hinge):
    """(lo, hi) degrees used to judge a pose: anatomical limits, finger twist 5 deg, PIP/DIP abduction ``hinge``."""
    lo, hi = J.lo.copy(), J.hi.copy()
    lo[:12, 2], hi[:12, 2] = -5, 5
    lo[_HINGE, 1], hi[_HINGE, 1] = -hinge, hinge
    return lo, hi


def write_view(path, V, F, M, names, Pk, J21, title):
    """static.html in fit_surface_markers' layout, so static_search.read_view / Judge.judge read it."""
    import plotly.graph_objects as go
    line = lambda i: np.ravel([[a[i], b[i], None] for a, b in zip(M, Pk)])
    fig = go.Figure([
        go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color="#e8a0a0", flatshading=True,
                  lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05), name="MANO (surface-fitted)"),
        go.Scatter3d(x=M[:, 0], y=M[:, 1], z=M[:, 2], mode="markers+text", marker=dict(size=6, color="#1b3a5c"), text=names,
                     textposition="top center", hovertext=[f"{n}: {np.linalg.norm(m - p):.1f} mm to its vertex" for n, m, p in zip(names, M, Pk)],
                     name="Vicon markers (static)"),
        go.Scatter3d(x=line(0), y=line(1), z=line(2), mode="lines", line=dict(color="#16a34a", width=5), name="marker -> its attached vertex"),
        go.Scatter3d(x=Pk[:, 0], y=Pk[:, 1], z=Pk[:, 2], mode="markers", marker=dict(size=4, color="#16a34a", symbol="diamond"),
                     name="attached vertex"),
        go.Scatter3d(x=J21[:, 0], y=J21[:, 1], z=J21[:, 2], mode="markers", marker=dict(size=3, color="#555555"),
                     name="MANO joint centres", visible="legendonly")])
    fig.update_layout(title=title, scene=dict(aspectmode="data"), width=1000, height=850)
    fig.write_html(str(path), include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", required=True)
    ap.add_argument("--start", required=True, help="static_search arm of this participant, or a folder written by this script")
    ap.add_argument("--pins", default=str(REPO / "configs/surface_fit/static_final_pins.json"), help="JSON {marker: vertex}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=8.0, help="mm a pin may move (same bone, dorsal)")
    ap.add_argument("--fix", default="", help="markers held on their pin (comma-separated)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--standoff", type=float, default=3.0)
    ap.add_argument("--cap", type=float, default=4.5, help="mm: hinge on |marker - attached vertex| above this")
    ap.add_argument("--w-cap", type=float, default=30.0)
    ap.add_argument("--w-nat", type=float, default=1e-2)
    ap.add_argument("--twist", type=float, default=4.0, help="deg: finger twist the fit may use (before --margin)")
    ap.add_argument("--margin", type=float, default=3.0, help="deg kept inside every limit by the fit")
    ap.add_argument("--hinge", type=float, default=3.0, help="deg: PIP/DIP abduction allowed")
    ap.add_argument("--along", action="store_true",
                    help="markers 2 and 3 keep their pin or move to the neighbouring vertex straight up or down the segment")
    ap.add_argument("--axis-tol", type=float, default=2.0,
                    help="mm: markers 2 and 3 attach only to vertices this close to their segment's dorsal midline")
    ap.add_argument("--couple", type=float, default=10.0, help="deg: DIP flexion allowed beyond 2/3 of the PIP's")
    ap.add_argument("--w-shape", type=float, default=1e-7)
    ap.add_argument("--min-dorsal", type=float, default=0.7, help="candidate vertex normal . its segment's dorsal")
    ap.add_argument("--w-inside", type=float, default=100.0)
    ap.add_argument("--w-dorsal", type=float, default=1e-2, help="nearest skin facing the segment's dorsal")
    a = ap.parse_args()
    p, out = a.participant, Path(a.out)

    J = ss.Judge()
    dev, model = J.dev, J.model
    X, names = load_csv(str(spc.sources(p)[p][0]))[:2]
    cloud = {k: v for k, v in zip([n.split(":")[-1] for n in names], static_cloud(X, 30)[0]) if np.isfinite(v).all()}
    mk = [n for n in HAND if n in cloud]
    M = torch.tensor(np.stack([cloud[n] for n in mk]) / 1000.0, dtype=torch.float32, device=dev)
    start = Path(a.start) if Path(a.start).is_dir() else ss.OUT / p / a.start
    z = np.load(start / ("params.npz" if (start / "params.npz").exists() else "static_params.npz"))
    pins = json.loads(Path(a.pins).read_text())

    sm = fsm.ScaledModel(model, dev)
    P = dict(hp=torch.tensor(z["hand_pose"].mean(0), device=dev), go=torch.tensor(z["global_orient"].mean(0), device=dev),
             tr=torch.tensor(z["transl"].mean(0), device=dev), betas=torch.tensor(z["betas"], device=dev))
    for v in P.values():
        v.requires_grad_(True)
    with torch.no_grad():
        sm.log_scale.fill_(float(np.log(z["scale"])))
    F = torch.tensor(np.asarray(model.faces, np.int64), device=dev)
    B = torch.tensor(J.B, dtype=torch.float32, device=dev)                  # (15, 3, 3) pose -> flexion, abduction, twist
    lo, hi = (torch.tensor(np.radians(x), dtype=torch.float32, device=dev) for x in (J.lo, J.hi))
    lo[:12, 2], hi[:12, 2] = -np.radians(a.twist), np.radians(a.twist)
    lo[_HINGE, 1], hi[_HINGE, 1] = -np.radians(a.hinge + a.margin - 0.5), np.radians(a.hinge + a.margin - 0.5)   # 0.5 deg inside
    lo, hi = lo + np.radians(a.margin), hi - np.radians(a.margin)
    clo, chi = check_limits(J, a.hinge)
    bone = model.lbs_weights.detach().argmax(1)
    child = {b: end for (j1, j2, j3), tip in fsm._FINGER_CHAINS                # bone -> its far end: joint or fingertip vertex
             for b, end in ((j1, ("j", j2)), (j2, ("j", j3)), (j3, ("v", tip)))}
    on_axis = [n[-1] in "23" and not n.startswith("Palm") for n in mk]

    def along(s_, V, Jm):
        """Pin ``s_`` plus its mesh neighbours most nearly straight distal and proximal along its bone."""
        nb = torch.unique(F[(F == s_).any(1)])
        nb = nb[nb != s_]
        b = int(bone[s_])
        kind, e = child[b]
        u = (Jm[e] if kind == "j" else V[e]) - Jm[b]
        u = u / u.norm()
        rel = V[nb] - V[s_]
        ax = rel @ u
        side = (rel - ax[:, None] * u).norm(dim=-1)
        return torch.stack([s_, nb[(ax - 2 * side).argmax()], nb[(-ax - 2 * side).argmax()]])

    def midline(c, V, Jm, d):
        """|Distance| (mm) of vertices ``c`` (one bone) from the plane through that bone's axis and dorsal ``d``."""
        b = int(bone[c[0]])
        kind, e = child[b]
        axis = (Jm[e] if kind == "j" else V[e]) - Jm[b]
        lat = torch.cross(axis, d, dim=0)
        return (((V[c] - Jm[b]) @ (lat / lat.norm())) * 1000).abs()

    def fwd():
        o = sm(global_orient=P["go"][None], hand_pose=P["hp"][None], transl=P["tr"][None], betas=P["betas"][None], return_verts=True)
        V = o.vertices[0]
        fn = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=-1)
        N = torch.zeros_like(V).index_add_(0, F[:, 0], fn).index_add_(0, F[:, 1], fn).index_add_(0, F[:, 2], fn)
        return o, V, N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def angles():
        return torch.einsum("jk,jkc->jc", P["hp"].view(15, 3), B)

    def dorsals():
        """The Judge's dorsal direction for each marker on the current mesh."""
        with torch.no_grad():
            J21 = J.j21(fwd()[0])[0].cpu().numpy()
        pd = J.palm_dorsal(J21)
        rows = []
        for n in mk:
            rj = J.fsm._JOINT_TO_REPO.get(n.lower())
            rows.append(pd if rj is None else J.fsm.segment_dorsals(J21, pd, [rj], "proj" if rj <= 3 else "cross")[0])
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=dev)

    def report(att):
        with torch.no_grad():
            _, V, N = fwd()
            d = ((M - V[att]).norm(dim=-1) * 1000).cpu().numpy()
            c = np.degrees(angles().cpu().numpy())
        ex = np.maximum(clo - c, c - chi).clip(0)
        ex[ex < 0.05] = 0                                                     # rounding
        return d, ex, c

    pin = torch.tensor([pins[n] for n in mk], device=dev)
    with torch.no_grad():
        V0 = fwd()[1]
    fixed = set(a.fix.split(","))
    cand = [torch.nonzero((bone == bone[s]) & ((V0 - V0[s]).norm(dim=-1) * 1000 <= (0 if n in fixed else a.radius))).flatten()
            for s, n in zip(pin, mk)]
    if a.along:
        with torch.no_grad():
            o0 = fwd()[0]
        cand = [along(s, V0, o0.joints[0]) if on_axis[i] else c for i, (s, c) in enumerate(zip(pin, cand))]
    att = torch.tensor(z["att"], device=dev) if "att" in z and list(z["names"]) == mk else pin.clone()
    params = [P["hp"], P["betas"], sm.log_scale, P["go"], P["tr"]]
    for r in range(a.rounds):
        dors = dorsals()
        with torch.no_grad():                                                  # best dorsal vertex for each marker
            o, V, N = fwd()
            for i, c in enumerate(cand):
                ok = (N[c] @ dors[i]) > a.min_dorsal
                c = c[ok] if ok.any() else c
                if on_axis[i] and not a.along:                                 # dorsal midline only
                    off = midline(c, V, o.joints[0], dors[i])
                    c = c[off <= a.axis_tol] if (off <= a.axis_tol).any() else c[off.argsort()[:3]]
                e = (M[i] - (V[c] + a.standoff * 1e-3 * N[c])).norm(dim=-1) + torch.relu(-((M[i] - V[c]) * N[c]).sum(-1))
                att[i] = c[e.argmin()]
        opt = torch.optim.Adam([{"params": [q], "lr": 2e-2 if q is P["hp"] else 2e-3 if q is P["tr"] else 5e-3} for q in params])
        for _ in range(a.iters):
            opt.zero_grad()
            _, V, N = fwd()
            diff = M - V[att]
            tgt = (M - (V[att] + a.standoff * 1e-3 * N[att])).norm(dim=-1)
            near = torch.cdist(M, V).argmin(1).detach()                       # the Judge's test: nearest skin, any bone
            inside = torch.relu(-(diff * N[att]).sum(-1)) + torch.relu(-((M - V[near]) * N[near]).sum(-1) + 5e-4)
            inside = inside + a.w_dorsal * torch.relu(0.2 - (N[near] * dors).sum(-1))
            c = angles()
            nat = (torch.relu(c - hi) ** 2 + torch.relu(lo - c) ** 2).sum()
            nat = nat + (torch.relu(c[[j + 1 for j in _PIP], 0] - 2 / 3 * c[_PIP, 0] - np.radians(a.couple)) ** 2).sum()
            loss = ((tgt ** 2).mean() + a.w_cap * (torch.relu(diff.norm(dim=-1) - a.cap * 1e-3) ** 2).sum()
                    + a.w_inside * (inside ** 2).sum() + a.w_nat * nat + a.w_shape * (P["betas"] ** 2).sum())
            loss.backward()
            opt.step()
        d, ex, _ = report(att)
        print(f"round {r}: max {d.max():.2f} mm ({mk[int(d.argmax())]}), mean {d.mean():.2f}, pose past the limits {ex.max():.1f} deg",
              flush=True)

    d, ex, c = report(att)
    dors = dorsals()
    with torch.no_grad():
        o, V, _ = fwd()
        off = {n: round(float(midline(att[i:i + 1], V, o.joints[0], dors[i])[0]), 1) for i, n in enumerate(mk) if on_axis[i]}
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        o = fwd()[0]
    V1, J21 = o.vertices[0].cpu().numpy() * 1000, mano_output_to_joints21(o)[0].cpu().numpy() * 1000
    write_view(out / "static.html", V1, F.cpu().numpy(), M.cpu().numpy() * 1000, mk, V1[att.cpu().numpy()], J21,
               f"{p} static: max |marker - attached vertex| {d.max():.1f} mm, pose {ex.max():.1f} deg past the limits")
    g = J.judge(out / "static.html")
    (out / "diag.json").write_text(json.dumps(g, indent=1))
    np.savez(out / "params.npz", hand_pose=P["hp"].detach().cpu().numpy()[None], global_orient=P["go"].detach().cpu().numpy()[None],
             transl=P["tr"].detach().cpu().numpy()[None], betas=P["betas"].detach().cpu().numpy(), scale=float(sm.log_scale.exp()),
             names=np.array(mk), att=att.cpu().numpy(), dist=d, excess=ex)
    met = bool(d.max() < 5 and ex.max() <= 0 and g["n_not_dorsal"] == 0 and g["natural_deg"] <= 0.5)
    summ = dict(participant=p, start=str(start), goal_met=met, max_mm=round(float(d.max()), 2), worst=mk[int(d.argmax())],
                pose_past_limits_deg=round(float(ex.max()), 2), judge_natural_deg=g["natural_deg"], not_dorsal=g["n_not_dorsal"],
                inside=g["n_inside"], judge_score=g["score"], scale=round(float(sm.log_scale.exp()), 4),
                attachments={n: int(v) for n, v in zip(mk, att.cpu().numpy())},
                moved_from_pin_mm={n: round(float(np.linalg.norm(V1[int(v)] - V1[pins[n]])), 1) for n, v in zip(mk, att.cpu().numpy())},
                marker_mm={n: round(float(x), 2) for n, x in zip(mk, d)}, off_midline_mm=off, options=vars(a))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("goal_met", "max_mm", "worst", "pose_past_limits_deg", "judge_natural_deg", "not_dorsal", "inside", "scale")}))


if __name__ == "__main__":
    main()
