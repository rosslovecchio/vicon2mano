#!/usr/bin/env python3
"""Calibrate the SOMA marker library to a participant from their static.

Fits MANO shape + pose to the static (``strategies/soma/calibrate.py``),
screens it, and writes a participant library: the base library's poses and
motion offsets, with this participant's hand shape and per-marker mean
offsets. Miss rates come from the participant's own raw trials (how often
each Vicon column is empty -- label-free).

Screening printed: per-marker fit error and suspect flag, and finger segment
lengths in the static vs the base library (a fingertip swap in the static is
NOT caught by the fit error -- a finger bends to reach it -- but shows up
here as an implausible segment).

    python scripts/soma/calibrate_static.py --participant P12 \\
        --static "<DATA_ROOT>/P12/Static_trial.csv" --out results/soma/libraries/library_P12.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT, find_session_file, load_trial_sessions, REF_CSV
from vicon2mano.core.fitter import FitConfig, MANOFitter
from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.calibrate import calibrated_library, fit_static, static_cloud
from vicon2mano.strategies.soma.library import MarkerLibrary, face_frames

SEGMENTS = [(f"{f}{a}", f"{f}{b}") for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
            for a, b in ((1, 2), (2, 3))] + [("Palm1", "Palm2"), ("Palm2", "Palm3"), ("Palm1", "Palm3")]


def base_markers(model, lib: MarkerLibrary, dev: str, n: int = 2000) -> np.ndarray:
    """(n, M, 3) mm: real base-subject marker clouds (hand-relative) rebuilt
    from library frames + their own measured residuals (NaN where absent)."""
    k = np.random.default_rng(0).choice(len(lib.hand_pose), min(n, len(lib.hand_pose)), replace=False)
    model = model.to(dev)
    with torch.no_grad():
        o = model(betas=torch.tensor(lib.betas, device=dev)[None].expand(len(k), -1),
                  hand_pose=torch.tensor(lib.hand_pose[k], device=dev),
                  global_orient=torch.zeros(len(k), 3, device=dev), transl=torch.zeros(len(k), 3, device=dev))
    V = o.vertices.cpu().numpy() * 1000
    P, R = face_frames(V, np.asarray(model.faces, np.int64), lib.face_idx, lib.bary)
    return P + np.einsum("nmij,nmj->nmi", R, lib.residual[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--base-library", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    ap.add_argument("--mano-dir", default=str(REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", default=None, help="fitted static mesh + markers (plotly)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = MarkerLibrary.load(args.base_library)
    model = MANOFitter(FitConfig(mano_model_path=args.mano_dir, hand_side="right", use_pca=False))._model

    X, names = load_csv(args.static)[:2]
    names = [n.split(":")[-1] for n in names]
    X = X[:, [names.index(n) for n in lib.labels]]
    cloud, info = static_cloud(X, args.n_frames)
    print(f"static {Path(args.static).name}: window frames {info['start']}..{info['start']+info['n']-1}, "
          f"median motion {info['motion_mm']:.2f} mm/frame; absent there: "
          f"{[n for n, p in zip(lib.labels, info['present']) if p < 0.5] or 'none'}")

    out = fit_static(model, lib, cloud, device=dev)
    print(f"betas: {np.round(out['betas'], 2).tolist()}  (base subject: {np.round(lib.betas, 2).tolist()})")
    print(f"{'marker':9s} {'fit err mm':>10s}  note")
    for m, n in enumerate(lib.labels):
        note = "forearm: not fitted, keeps base offsets" if n.startswith("Forearm") else \
               "SUSPECT: excluded, keeps base offsets" if n in out["suspects"] else \
               "absent in static" if not np.isfinite(cloud[m]).all() else ""
        e = out["err_mm"][m]
        print(f"{n:9s} {e:10.1f}  {note}" if np.isfinite(e) else f"{n:9s} {'-':>10s}  {note}")

    # segment screen: static vs the base subject's real geometry. Library
    # frames re-posed with their own residuals reproduce the real markers
    # exactly (relative to the hand), so their distances are measured ones.
    li = {n: i for i, n in enumerate(lib.labels)}
    Bm = base_markers(model, lib, dev)
    print(f"\n{'segment':16s} {'static mm':>9s} {'base median':>11s} {'base p5-p95':>13s}")
    for a, b in SEGMENTS:
        sd = np.linalg.norm(cloud[li[a]] - cloud[li[b]])
        bd = np.linalg.norm(Bm[:, li[a]] - Bm[:, li[b]], axis=-1)
        bd = bd[np.isfinite(bd)]
        q = np.percentile(bd, [5, 50, 95])
        flag = "  <- outside base p5-p95" if np.isfinite(sd) and not (q[0] <= sd <= q[2]) else ""
        print(f"{a+'-'+b:16s} {sd:9.1f} {q[1]:11.1f} {q[0]:6.1f}-{q[2]:<6.1f}{flag}")

    # miss rates from the participant's own raw trials (label-free: empty columns)
    sess = load_trial_sessions(REF_CSV).get(args.participant, {})
    pres = []
    for trial, csv_name in sess.items():
        if trial.lower() == "static":
            continue
        p = find_session_file(Path(DATA_ROOT) / args.participant, csv_name)
        if p is None:
            continue
        Y, ny = load_csv(str(p))[:2]
        ny = [n.split(":")[-1] for n in ny]
        pres.append(np.isfinite(Y[:, [ny.index(n) for n in lib.labels]]).all(-1))
        print(f"miss rates from {trial}: {Path(p).name} ({len(Y)} frames)")
    miss = 1 - np.concatenate(pres).mean(0) if pres else None

    new = calibrated_library(lib, out["betas"], out["r_s"], miss,
                             source=f"{args.participant} static {Path(args.static).name}")
    new.save(args.out)
    print(f"calibrated {int(np.isfinite(out['r_s']).all(-1).sum())}/{len(lib.labels)} markers; saved {args.out}")

    if args.html:
        import plotly.graph_objects as go
        V, F = out["vertices_mm"], np.asarray(model.faces)
        P, R = face_frames(V, F, lib.face_idx, lib.bary)
        pred = P + np.einsum("mij,mj->mi", R, np.nan_to_num(np.nanmedian(lib.residual, 0)))
        ok = np.isfinite(cloud).all(-1)
        fig = go.Figure([
            go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                      color="#e8a0a0", opacity=0.6, name="fitted MANO"),
            go.Scatter3d(x=cloud[ok, 0], y=cloud[ok, 1], z=cloud[ok, 2], mode="markers+text",
                         text=[n for n, o in zip(lib.labels, ok) if o], marker=dict(size=5, color="#1b3a5c"),
                         name="static markers"),
            go.Scatter3d(x=np.ravel([[a[0], b[0], None] for a, b, o in zip(cloud, pred, ok) if o]),
                         y=np.ravel([[a[1], b[1], None] for a, b, o in zip(cloud, pred, ok) if o]),
                         z=np.ravel([[a[2], b[2], None] for a, b, o in zip(cloud, pred, ok) if o]),
                         mode="lines", line=dict(color="#16a34a", width=4), name="marker -> model")])
        fig.update_layout(scene=dict(aspectmode="data"), width=1000, height=800,
                          title=f"{args.participant} static fit; suspects: {out['suspects'] or 'none'}")
        Path(args.html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(args.html, include_plotlyjs=True)
        print(f"saved {args.html}")


if __name__ == "__main__":
    main()
