#!/usr/bin/env python3
"""Search the static surface fit's settings for the best static per participant.

Priorities, in order: every marker on the DORSAL side of its segment, markers CLOSE TO THE SKIN (just outside it),
and a NATURAL static pose. Each run is the surface fit's static stage only (scripts/mano/fit_surface_markers.py,
run from a frozen checkout so edits to the live tree cannot change a run), and is judged on the mesh its own
static view shows:

  dorsal     per marker: is the nearest skin point on the back of the marker's segment (outward normal . the
             segment's dorsal direction > 0; dorsal directions as the fit script defines them; thumb with "proj")
  distance   per marker: signed distance to the nearest skin point, mm, negative = inside the skin
  natural    joint angles of that mesh (pose recovered from its vertices) against the repo's limits;
             finger twist counts above 5 deg

score = 10 * (markers absent from the view + not dorsal + inside) + mean |distance - 3 mm| + 0.2 * (degrees outside limits)
(3 mm = the configured standoff). Lower is better. All raw numbers are kept, so the ranking can be redone.

    python scripts/soma/static_search.py --participants P3,P4 --n-configs 30     # resumable
    python scripts/soma/static_search.py --report --onedrive <dir>
    python scripts/soma/static_search.py --refine --participants P3,P4         # best setting +/- scale, palm-first
    python scripts/soma/static_search.py --refine --report --onedrive <dir>
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "results/soma/static_search"
CODE = REPO.parent / "vicon2mano_frozen"                    # frozen checkout the fit script runs from
sys.path.insert(0, str(CODE / "scripts" / "mano"))            # its helpers too, not the live tree's
STANDOFF = 3.0
# refine arms: the best search setting plus these fit flags
PALM_W = 10.0                                                 # --palm-first weight of the palm markers
ARMS = {"v_best": {}, "v_scale": {"fit_scale": True}, "v_palm": {"palm_first": PALM_W},
        "v_scale_palm": {"fit_scale": True, "palm_first": PALM_W}}
# static-stage settings searched (first value of each = the production setting)
GRID = {
    "min_dorsal_dot": [0.0, 0.3, 0.5],
    "w_inside": [10.0, 100.0],
    "icp_rounds": [5, 10],
    "thumb_dorsal": ["cross", "proj"],
    "min_own_weight": [0.0, 0.25],
    "biomech": [1.0, 10.0],
    "twist_weight": [10.0, 100.0],
    "w_pose": [0.0, 1e-4, 1e-3],
    "n_frames": [30, 60],
    "complete_frames": [False, True],
    "iters_surface": [600, 1500],
    "palm1_pin": [True, False],
}


def configs(n: int, seed: int, participant: str) -> list[dict]:
    """The production setting first, then ``n`` distinct random draws from GRID."""
    grid = dict(GRID)
    if participant in ("P1", "P2"):
        grid["palm1_pin"] = [False]                          # their attach map has no Palm1 pin
    base = {k: v[0] for k, v in grid.items()}
    rng, seen, out = random.Random(f"{seed}-{participant}"), set(), [base]
    seen.add(json.dumps(base, sort_keys=True))
    total = int(np.prod([len(v) for v in grid.values()]))
    while len(out) < min(n + 1, total):
        c = {k: rng.choice(v) for k, v in grid.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def fit_cmd(p: str, static: Path, c: dict, rdir: Path) -> list[str]:
    import run_fullset as rf
    rf.REPO = CODE                                            # config and attach files from the frozen checkout
    cfg = CODE / "configs/surface_fit" / (f"{p}.json" if p in ("P1", "P2") else "P10.json")
    args = rf.fit_args_from_config(cfg)
    present = rf.present_markers(static)
    attach, _ = rf.prune_attach(cfg, present - (set() if c["palm1_pin"] else {"Palm1"}), rdir / "attach.json")
    if attach is not None:
        args += ["--attach", str(attach)]
    for k in ("min_dorsal_dot", "w_inside", "icp_rounds", "thumb_dorsal", "min_own_weight", "biomech", "twist_weight",
              "w_pose", "n_frames", "iters_surface"):
        args += ["--" + k.replace("_", "-"), str(c[k])]
    for k in ("complete_frames", "fit_scale"):
        if c.get(k):
            args.append("--" + k.replace("_", "-"))
    if c.get("palm_first"):
        args += ["--palm-first", str(c["palm_first"])]
    from vicon2mano.core.dataset import DATA_ROOT
    return [sys.executable, "-W", "ignore", "scripts/mano/fit_surface_markers.py",
            "--mano-dir", str(DATA_ROOT / "mano_v1_2" / "models"), "--static", str(static),
            "--out", str(rdir / "static.html")] + args


def _arr(v):
    if isinstance(v, dict):                                   # plotly typed array
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v, dtype=float)


def read_view(html: Path) -> dict:
    """Mesh, faces, named markers and joint centres from a fit_surface_markers static view."""
    t = html.read_text(encoding="utf-8")
    i = t.index("[", t.index("Plotly.newPlot("))
    depth = 0
    for k in range(i, len(t)):
        depth += {"[": 1, "]": -1}.get(t[k], 0)
        if depth == 0:
            break
    traces = {tr.get("name"): tr for tr in json.loads(t[i:k + 1])}
    mesh = next(tr for tr in traces.values() if tr.get("type") == "mesh3d")
    mk = next(tr for n, tr in traces.items() if n and n.startswith("Vicon markers"))
    J = traces["MANO joint centres"]
    xyz = lambda tr: np.stack([_arr(tr[a]) for a in "xyz"], 1).astype(float)
    return dict(V=xyz(mesh), F=np.stack([_arr(mesh[a]) for a in "ijk"], 1).astype(np.int64),
                M=xyz(mk), names=list(mk["text"]), J=xyz(J))


def nearest_skin(V, F, P):
    """Nearest point on the mesh surface for each of P (no outward-side preference, so a marker inside the skin
    is seen as inside): (points, unit outward face normals). Same projection as the fit script's attachment search."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, C - A
    n = np.cross(e1, e2)
    nn = np.maximum((n * n).sum(1), 1e-18)
    q_out, n_out = [], []
    for p in P:
        w = p - A
        u = np.maximum((np.cross(w, e2) * n).sum(1) / nn, 0.0)
        v = np.maximum((np.cross(e1, w) * n).sum(1) / nn, 0.0)
        s = np.maximum(u + v, 1.0)
        q = A + (u / s)[:, None] * e1 + (v / s)[:, None] * e2
        f = np.argmin(((p - q) ** 2).sum(1))
        q_out.append(q[f])
        n_out.append(n[f] / np.sqrt(nn[f]))
    return np.array(q_out), np.array(n_out)


class Judge:
    """Everything that needs MANO, loaded once per process."""

    def __init__(self):
        import torch
        import fit_surface_markers as fsm
        from vicon2mano.core.dataset import DATA_ROOT
        from vicon2mano.core.fitter import FitConfig, MANOFitter, mano_output_to_joints21
        self.torch, self.fsm, self.j21 = torch, fsm, mano_output_to_joints21
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MANOFitter(FitConfig(mano_model_path=str(DATA_ROOT / "mano_v1_2" / "models"),
                                          hand_side="right", use_pca=False))._model.to(self.dev)
        self.B = fsm.joint_axis_basis(self.model, self.dev, 45)
        self.lo, self.hi = (np.degrees(a) for a in fsm.basis_limit_bounds(False, False))
        # handedness: which way cross(index MCP, pinky MCP) points relative to a known dorsal vertex
        with torch.no_grad():
            o = self.model(betas=torch.zeros(1, 10, device=self.dev), hand_pose=torch.zeros(1, 45, device=self.dev),
                           global_orient=torch.zeros(1, 3, device=self.dev), transl=torch.zeros(1, 3, device=self.dev))
        V0, J0 = o.vertices[0].cpu().numpy(), self.j21(o)[0].cpu().numpy()
        self.chir = float(np.sign((V0[212] - J0[6]) @ np.cross(J0[5] - J0[0], J0[17] - J0[0])))   # 212: P10's Index2 pin

    def palm_dorsal(self, J):
        n = np.cross(J[5] - J[0], J[17] - J[0])
        return self.chir * n / np.linalg.norm(n)

    def pose_of(self, V_mm):
        """MANO pose that reproduces the shown mesh (betas, global scale, orient, transl, pose fitted to its vertices)."""
        torch = self.torch
        V = torch.tensor(V_mm / 1000.0, dtype=torch.float32, device=self.dev)
        z = lambda k: torch.zeros(1, k, device=self.dev, requires_grad=True)
        with torch.no_grad():                                   # rigid start: template onto the mesh
            T = self.model(betas=torch.zeros(1, 10, device=self.dev), hand_pose=torch.zeros(1, 45, device=self.dev),
                           global_orient=torch.zeros(1, 3, device=self.dev), transl=torch.zeros(1, 3, device=self.dev)).vertices[0]
        a, b = T.cpu().numpy(), V.cpu().numpy()
        U, _, Wt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
        d = np.sign(np.linalg.det(Wt.T @ U.T))
        Rm = Wt.T @ np.diag([1, 1, d]) @ U.T
        from scipy.spatial.transform import Rotation
        orient = torch.tensor(Rotation.from_matrix(Rm).as_rotvec()[None], dtype=torch.float32, device=self.dev,
                              requires_grad=True)
        betas, hp, ls, tr = z(10), z(45), z(1), torch.tensor((b.mean(0) - Rm @ a.mean(0))[None], dtype=torch.float32,
                                                   device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [orient, tr, ls], "lr": 5e-3}, {"params": [betas, hp], "lr": 2e-2}])
        for _ in range(1500):
            opt.zero_grad()
            v = ls.exp() * self.model(betas=betas, hand_pose=hp, global_orient=orient, transl=tr).vertices[0]
            loss = ((v - V) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
        return hp.detach().cpu().numpy()[0], float(loss.sqrt().item() * 1000)

    def judge(self, html: Path) -> dict:
        v = read_view(html)
        seen = np.isfinite(v["M"]).all(1)                      # a marker with no position in the view is "absent"
        cp, n = np.full_like(v["M"], np.nan), np.full_like(v["M"], np.nan)
        cp[seen], n[seen] = nearest_skin(v["V"], v["F"], v["M"][seen])
        signed = np.einsum("ij,ij->i", v["M"] - cp, n)
        pd = self.palm_dorsal(v["J"])
        rows = []
        for k, name in enumerate(v["names"]):
            if not seen[k]:
                rows.append(dict(marker=name, dorsal=None, dorsal_dot=None, dorsal_dot_cross=None, distance_mm=None, absent=True))
                continue
            key = name.lower()
            alt = None
            if key in self.fsm._JOINT_TO_REPO:
                rj = self.fsm._JOINT_TO_REPO[key]
                # thumb: the fit script notes that cross(ulnar, axis) can point to the wrong side of the thumb,
                # so it is judged with the palm dorsal projected off the thumb axis ("proj"); "cross" kept for reference
                seg_d = self.fsm.segment_dorsals(v["J"], pd, [rj], "proj" if rj <= 3 else "cross")[0]
                if rj <= 3:
                    alt = round(float(n[k] @ self.fsm.segment_dorsals(v["J"], pd, [rj], "cross")[0]), 3)
            else:                                               # palm markers: the palm's own dorsal
                seg_d = pd
            dot = float(n[k] @ seg_d)
            rows.append(dict(marker=name, dorsal=bool(dot > 0), dorsal_dot=round(dot, 3), dorsal_dot_cross=alt,
                             distance_mm=round(float(signed[k]), 2)))
        hp, rms = self.pose_of(v["V"])
        comp = np.degrees(np.einsum("jk,jkc->jc", hp.reshape(15, 3), self.B))
        over = []
        for j in range(15):
            for c in range(3):
                o = self.lo[j, c] - comp[j, c] if comp[j, c] < self.lo[j, c] else comp[j, c] - self.hi[j, c]
                if j < 12 and c == 2:
                    o = abs(comp[j, c]) - 5.0
                if o > 0:
                    over.append([self.fsm._FINGER_JOINTS[j], ("flexion", "abduction", "twist")[c],
                                 round(float(comp[j, c]), 1), round(float(o), 1)])
        d = np.array([r["distance_mm"] for r in rows if r["distance_mm"] is not None])
        n_absent = sum(r.get("absent", False) for r in rows)
        n_not_dorsal = sum(r["dorsal"] is False for r in rows)
        n_inside = int((d < 0).sum())
        skin = float(np.abs(d - STANDOFF).mean())
        natural = float(sum(o[3] for o in over))
        return dict(markers=rows, n_absent=n_absent, n_not_dorsal=n_not_dorsal, n_inside=n_inside, skin_mm=round(skin, 2),
                    outside_limits=over, natural_deg=round(natural, 1), pose_comp=comp.round(1).tolist(),
                    pose_recovery_rms_mm=round(rms, 3),
                    score=round(10 * (n_absent + n_not_dorsal + n_inside) + skin + 0.2 * natural, 3))


def fit_one(J, p, static, ci, c, rdir):
    """Fit and judge one setting into ``rdir`` (resumable: an existing result is only re-scored if stale)."""
    res = rdir / "diag.json"
    if res.exists():
        old = json.loads(res.read_text())
        stale = "judge_error" in old or ("score" in old and not np.isfinite(old["score"]))
        if stale and (rdir / "static.html").exists():                       # re-score, no refit
            try:
                old = {k: v for k, v in old.items() if k != "judge_error"} | J.judge(rdir / "static.html")
            except Exception as e:
                old["judge_error"] = f"{type(e).__name__}: {e}"
            res.write_text(json.dumps(old, indent=1))
        return
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "fit.log", "w", encoding="utf-8", errors="replace") as f:
        try:
            rc = subprocess.run(fit_cmd(p, static, c, rdir), cwd=CODE, stdout=f, stderr=subprocess.STDOUT,
                                timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    out = dict(participant=p, config_id=ci, config=c, static=str(static), returncode=rc, code_commit=head)
    if rc == 0 and (rdir / "static.html").exists():
        try:
            out |= J.judge(rdir / "static.html")
        except Exception as e:                          # a view we cannot read is a result, not a crash
            out["judge_error"] = f"{type(e).__name__}: {e}"
    res.write_text(json.dumps(out, indent=1))
    print(f"{p} {rdir.name} rc={rc} score={out.get('score')} not-dorsal={out.get('n_not_dorsal')} "
          f"inside={out.get('n_inside')} skin={out.get('skin_mm')} natural={out.get('natural_deg')}", flush=True)


def run(participants, n, seed):
    import static_pose_check as spc
    J = Judge()
    for p in participants:
        static = spc.sources(p)[p][0]
        for ci, c in enumerate(configs(n, seed, p)):
            fit_one(J, p, static, ci, c, OUT / p / f"c{ci:02d}")


def best_run(p):
    """The participant's best-scoring search run (finite score), or None."""
    rows = [json.loads(f.read_text()) for f in (OUT / p).glob("c*/diag.json")]
    rows = [r for r in rows if "score" in r and np.isfinite(r["score"])]
    return min(rows, key=lambda r: r["score"]) if rows else None


def refine(participants):
    """Each participant's best search setting again, in every arm of ARMS."""
    import static_pose_check as spc
    J = Judge()
    for p in participants:
        best = best_run(p)
        if best is None:
            print(f"{p}: no usable search run, skipped", flush=True)
            continue
        static = spc.sources(p)[p][0]
        for arm, extra in ARMS.items():
            fit_one(J, p, static, best["config_id"], best["config"] | extra, OUT / p / arm)


def refine_report(onedrive: Path | None):
    import re
    import shutil
    key = lambda p: int("".join(ch for ch in p if ch.isdigit()))
    lines = ["# Best static setting, with a global scale and palm-first", "",
             "Each participant's best search setting (cNN), refit with the current fit script in every arm. Cells: "
             "score (not-dorsal / inside / skin mm / natural deg); scale = fitted global hand scale in v_scale_palm.", "",
             "| P | setting | search best (old code) | " + " | ".join(ARMS) + " | scale |",
             "|---|---|---|" + "---|" * len(ARMS) + "---|"]
    cell = lambda r: (f"{r['score']} ({r['n_not_dorsal']} / {r['n_inside']} / {r['skin_mm']} / {r['natural_deg']})"
                      if r and "score" in r else f"failed (rc={r.get('returncode')})" if r else "-")
    for p in sorted((d.name for d in OUT.iterdir() if (d / "v_best").exists()), key=key):
        best = best_run(p)
        arms = {a: json.loads((OUT / p / a / "diag.json").read_text()) for a in ARMS if (OUT / p / a / "diag.json").exists()}
        log = OUT / p / "v_scale_palm" / "fit.log"
        m = re.findall(r"fitted global scale = ([0-9.]+)", log.read_text(errors="replace")) if log.exists() else []
        lines.append(f"| {p} | c{best['config_id']:02d} | {cell(best)} | " + " | ".join(cell(arms.get(a)) for a in ARMS)
                     + f" | {m[-1] if m else '-'} |")
        for a, r in arms.items():
            if onedrive and (OUT / p / a / "static.html").exists():
                (onedrive / p).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OUT / p / a / "static.html", onedrive / p / f"{p}_static_{a[2:]}.html")
                (onedrive / p / f"{p}_{a[2:]}_diagnostics.json").write_text(json.dumps(r, indent=1))
    text = "\n".join(lines) + "\n"
    print(text)
    for d in ([OUT] + ([onedrive] if onedrive else [])):
        (d / "README_scale_palm.md").write_text(text, encoding="utf-8")


def report(onedrive: Path | None):
    import csv
    import shutil
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*/c*/diag.json"))]
    ok = [r for r in rows if "score" in r and np.isfinite(r["score"])]
    key = lambda p: int("".join(ch for ch in p if ch.isdigit()))
    lines = ["# Static-fit search (overnight) - best settings per participant", "",
             "Priorities: dorsal, then close to skin, then natural pose. score = 10*(absent + not-dorsal + inside) + "
             "mean|distance-3mm| + 0.2*degrees outside the joint limits (lower is better).", "",
             "| P | runs | production: not-dorsal / inside / skin mm / natural deg | best: not-dorsal / inside / skin mm / natural deg | best settings (changed from production) |",
             "|---|---|---|---|---|"]
    summary = []
    for p in sorted({r["participant"] for r in rows}, key=key):
        mine = [r for r in ok if r["participant"] == p]
        if not mine:
            lines.append(f"| {p} | 0 usable | | | |")
            continue
        base = next((r for r in mine if r["config_id"] == 0), None)
        best = min(mine, key=lambda r: r["score"])
        fmt = lambda r: f"{r['n_not_dorsal']} / {r['n_inside']} / {r['skin_mm']} / {r['natural_deg']}" if r else "-"
        diff = {k: v for k, v in best["config"].items() if base is None or base["config"].get(k) != v}
        lines.append(f"| {p} | {len(mine)} | {fmt(base)} | {fmt(best)} | {json.dumps(diff) if diff else 'production is best'} |")
        for tag, r in (("production", base), ("best", best)):
            if r is None:
                continue
            for m in r["markers"]:
                summary.append(dict(participant=p, which=tag, config_id=r["config_id"], marker=m["marker"], dorsal="yes" if m["dorsal"] else "NO",
                                    distance_from_skin_mm=m["distance_mm"], dorsal_dot=m["dorsal_dot"]))
            if onedrive:
                src = OUT / p / f"c{r['config_id']:02d}" / "static.html"
                dst = onedrive / p / f"{p}_static_{tag}.html"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                (onedrive / p / f"{p}_{tag}_diagnostics.json").write_text(json.dumps(r, indent=1))
    text = "\n".join(lines) + "\n"
    print(text)
    allrows = [dict(participant=r["participant"], config_id=r["config_id"], score=r.get("score"),
                    n_not_dorsal=r.get("n_not_dorsal"), n_inside=r.get("n_inside"), skin_mm=r.get("skin_mm"),
                    natural_deg=r.get("natural_deg"), returncode=r["returncode"], **{f"cfg_{k}": v for k, v in r["config"].items()})
               for r in rows]
    for d in ([OUT] + ([onedrive] if onedrive else [])):
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(text, encoding="utf-8")
        for name, data in (("all_runs.csv", allrows), ("markers_production_vs_best.csv", summary)):
            if data:
                with open(d / name, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(data[0]))
                    w.writeheader()
                    w.writerows(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="")
    ap.add_argument("--n-configs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--refine", action="store_true", help="refit each participant's best setting in every arm of ARMS")
    ap.add_argument("--onedrive", default=None, help="with --report: copy production/best views and tables here")
    a = ap.parse_args()
    if a.report:
        od = Path(a.onedrive) if a.onedrive else None
        return refine_report(od) if a.refine else report(od)
    if a.refine:
        return refine([p for p in a.participants.split(",") if p])
    run([p for p in a.participants.split(",") if p], a.n_configs, a.seed)


if __name__ == "__main__":
    main()
