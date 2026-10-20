#!/usr/bin/env python3
"""Is a fitted static pose natural? Joint angles of the static fit against the repo's limits.

Re-fits each participant's static with calibrate.fit_static (what calibration uses), then projects the
pose onto each joint's measured flexion / abduction / twist axes (fit_surface_markers.joint_axis_basis) and
compares with basis_limit_bounds(): finger twist is fixed at 0, so any twist is reported, not just excess.
Run at the production pose prior (w_pose=20) and a weak one (w_pose=1): the prior pulls toward a flat hand,
so a pose that stays natural when the prior is weak means the markers themselves ask for a natural pose.

    python scripts/soma/static_pose_check.py --participants P3,P4 --w-pose 20,1
    python scripts/soma/static_pose_check.py --table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for sub in ("", "scripts/mano"):
    sys.path.insert(0, str(REPO / sub))

OUT = REPO / "results/soma/static_pose_check"
FULLSET = REPO / "results/soma/fullset"
ARCHIVE = FULLSET / "_archive_2026-10-19/fullset"
P1P2 = dict(drop=["Forearm3", "Forearm4", "Palm3"], offset=("Forearm1", "Forearm2", "Palm1", "Palm2", "Thumb1"))
NAMES = ["index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
         "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]
TYPES = {"MCP": [0, 3, 6, 9], "PIP": [1, 4, 7, 10], "DIP": [2, 5, 8, 11]}


def sources(p: str) -> dict[str, tuple[Path, dict]]:
    """{tag: (static csv, options)}: the static the pipeline used, plus the archived real static where there is one."""
    from vicon2mano.core.dataset import DATA_ROOT
    opts = P1P2 if p in ("P1", "P2") else {}
    if p in ("P1", "P2"):
        cur = REPO / f"results/soma/labels/{p}/static/soma_labelled.csv"
    elif p == "P10":
        cur = DATA_ROOT / "P10/static_trial.csv"
    else:
        pseudo = FULLSET / p / "static/pseudo_static.csv"
        cur = pseudo if pseudo.exists() else FULLSET / p / "static/soma_labelled.csv"
    out = {p: (cur, opts)}
    old = ARCHIVE / p / "static/soma_labelled.csv"
    if old.exists():
        out[f"{p}_old_real_static"] = (old, opts)
    return out


def check(csv: Path, opts: dict, w_pose: float, base_library: Path, mano_dir: str, twist_tol: float = 5.0) -> dict:
    import torch
    import fit_surface_markers as fsm
    from vicon2mano.core.fitter import FitConfig, MANOFitter
    from vicon2mano.core.loader import load_csv
    from vicon2mano.strategies.soma.calibrate import fit_static, static_cloud
    from vicon2mano.strategies.soma.library import MarkerLibrary
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = MarkerLibrary.load(str(base_library))
    if opts.get("drop"):
        lib = lib.subset([l for l in lib.labels if l not in opts["drop"]])
    model = MANOFitter(FitConfig(mano_model_path=mano_dir, hand_side="right", use_pca=False))._model
    X, names = load_csv(str(csv))[:2]
    names = [n.split(":")[-1] for n in names]
    cloud, info = static_cloud(X[:, [names.index(n) for n in lib.labels]], 200)
    fit = fit_static(model, lib, cloud, device=dev, w_pose=w_pose, offset_only=tuple(opts.get("offset", ())))
    B = fsm.joint_axis_basis(model, dev, 45)                                    # (15, 3, 3): rows joint, cols component
    comp = np.degrees(np.einsum("jk,jkc->jc", fit["hand_pose"].reshape(15, 3), B))   # (15, 3) flexion, abduction, twist
    lo, hi = (np.degrees(a) for a in fsm.basis_limit_bounds(False, False))
    bad = []                                    # (joint, component, value, limit, overshoot), all in degrees
    for j in range(15):
        for c in range(3):
            over = lo[j, c] - comp[j, c] if comp[j, c] < lo[j, c] else comp[j, c] - hi[j, c]
            if j < 12 and c == 2:               # finger twist is fixed at 0: only a clear twist counts
                over = abs(comp[j, c]) - twist_tol
            if over > 1e-6:
                bad.append((NAMES[j], ("flexion", "abduction", "twist")[c], round(float(comp[j, c]), 1),
                            round(float(lo[j, c] if comp[j, c] < lo[j, c] else hi[j, c]), 1), round(float(over), 1)))
    return dict(w_pose=w_pose, csv=str(csv), window=[info["start"], info["start"] + info["n"] - 1],
                motion_mm=info["motion_mm"], comp=comp.round(1).tolist(), outside=bad,
                suspects=list(fit["suspects"]), max_abs_beta=float(np.abs(fit["betas"]).max()))


def table():
    rows = [json.loads(f.read_text()) | {"tag": f.stem.rsplit("_w", 1)[0]} for f in sorted(OUT.glob("*.json"))]
    print(f"{'static':24s}{'w':>3s}  {'MCP flex mean/max':>18s} {'PIP':>13s} {'DIP':>13s} {'thumb flex 1/2/3':>18s} "
          f"{'|abd|':>6s} {'|twist|':>8s}  outside limits (n: worst 3; finger twist above 5 deg counts)")
    for r in sorted(rows, key=lambda r: (int("".join(c for c in r["tag"].split("_")[0] if c.isdigit())), r["tag"], -r["w_pose"])):
        c = np.array(r["comp"])
        f = lambda t: f"{c[TYPES[t], 0].mean():5.1f}/{c[TYPES[t], 0].max():5.1f}"
        th = "/".join(f"{c[j, 0]:.0f}" for j in (12, 13, 14))
        print(f"{r['tag']:24s}{r['w_pose']:3.0f}  {f('MCP'):>18s} {f('PIP'):>13s} {f('DIP'):>13s} {th:>18s} "
              f"{np.abs(c[:12, 1]).max():6.1f} {np.abs(c[:12, 2]).max():8.1f}  "
              + (f"{len(r['outside'])}: " + "; ".join(f"{n} {k} {v:g}" for n, k, v, lim, o in sorted(r["outside"], key=lambda b: -b[4])[:3])
             if r["outside"] else "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="")
    ap.add_argument("--w-pose", default="20,1")
    ap.add_argument("--base-library", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    ap.add_argument("--mano-dir", default=None)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table()
    from vicon2mano.core.dataset import DATA_ROOT
    mano = a.mano_dir or str(DATA_ROOT / "mano_v1_2" / "models")
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [x for x in a.participants.split(",") if x]:
        for tag, (csv, opts) in sources(p).items():
            for w in (float(v) for v in a.w_pose.split(",")):
                res = check(csv, opts, w, Path(a.base_library), mano)
                (OUT / f"{tag}_w{w:g}.json").write_text(json.dumps(res))
                print(f"{tag} w_pose={w:g}: {len(res['outside'])} components outside limits", flush=True)


if __name__ == "__main__":
    main()
