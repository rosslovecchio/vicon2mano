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

    # a mapped static (preferred):
    python scripts/soma/calibrate_static.py --participant P10 --session Static --out ...
    # P12's real static is not in manual_frames.csv, so it is passed as a path:
    python scripts/soma/calibrate_static.py --participant P12         --static results/soma/labels/P12/static/soma_labelled.csv --out results/soma/libraries/library_P12.npz
    # no static (P1/P2): a still, flat stretch of a mapped trial, reduced marker set:
    python scripts/soma/calibrate_static.py --participant P1 --session "Trial 1 Hands only"         --window 13700:13900 --drop-labels Forearm3,Forearm4,Palm3         --offset-only Forearm1,Forearm2,Palm1,Palm2,Thumb1 --out results/soma/libraries/library_P1.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import DATA_ROOT, REF_CSV, find_session_file, find_trial_csv, load_trial_sessions
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


def write_static_html(model, lib: MarkerLibrary, cloud: np.ndarray, out: dict, path, title: str) -> None:
    """Plotly page of a static fit: the fitted mesh, the static markers, and a line from each marker to its model point."""
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
    fig.update_layout(scene=dict(aspectmode="data"), width=1000, height=800, title=title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--static", default=None,
                    help="static CSV path -- only for a static manual_frames.csv does not map "
                         "(P12's Static_trial.csv); otherwise use --session")
    ap.add_argument("--session", default=None,
                    help="a manual_frames.csv session of --participant, e.g. 'Static' or "
                         "'Trial 1 Hands only' (with --window, a still stretch standing in for a static)")
    ap.add_argument("--base-library", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    ap.add_argument("--mano-dir", default=str(REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--w-pose", type=float, default=20.0,
                    help="pull of the static pose toward a flat hand (see calibrate.fit_static)")
    ap.add_argument("--drop-labels", default="",
                    help="markers this participant never had (P1/P2: Forearm3,Forearm4,Palm3)")
    ap.add_argument("--offset-only", default="",
                    help="markers placed differently from the base subject: not fitted, offsets recorded "
                         "(P1/P2: Forearm1,Forearm2,Palm1,Palm2,Thumb1)")
    ap.add_argument("--window", default=None,
                    help="start:end frames to search instead of the whole file -- e.g. a still stretch "
                         "of a movement trial standing in for a missing static")
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", default=None, help="fitted static mesh + markers (plotly)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = MarkerLibrary.load(args.base_library)
    drop = [d for d in args.drop_labels.split(",") if d]
    if drop:
        lib = lib.subset([l for l in lib.labels if l not in drop])
    model = MANOFitter(FitConfig(mano_model_path=args.mano_dir, hand_side="right", use_pca=False))._model

    if (args.static is None) == (args.session is None):
        raise SystemExit("give exactly one of --static or --session")
    if args.session:
        static_path, *_ = find_trial_csv(args.participant, args.session)
        if static_path is None:
            raise SystemExit(f"manual_frames.csv maps no file to {args.participant}/{args.session}")
        args.static = str(static_path)
        print(f"{args.participant}/{args.session} -> {static_path.name} (manual_frames.csv)")
    X, names = load_csv(args.static)[:2]
    names = [n.split(":")[-1] for n in names]
    X = X[:, [names.index(n) for n in lib.labels]]
    off = 0
    if args.window:
        off, end = (int(v) for v in args.window.split(":"))
        X = X[off:end]
    cloud, info = static_cloud(X, args.n_frames)
    info["start"] += off
    print(f"static {Path(args.static).name}: window frames {info['start']}..{info['start']+info['n']-1}, "
          f"median motion {info['motion_mm']:.2f} mm/frame; absent there: "
          f"{[n for n, p in zip(lib.labels, info['present']) if p < 0.5] or 'none'}")

    offset_only = tuple(o for o in args.offset_only.split(",") if o)
    out = fit_static(model, lib, cloud, device=dev, w_pose=args.w_pose, offset_only=offset_only)
    print(f"betas: {np.round(out['betas'], 2).tolist()}  (base subject: {np.round(lib.betas, 2).tolist()})")
    print(f"{'marker':9s} {'fit err mm':>10s}  note")
    for m, n in enumerate(lib.labels):
        note = "not fitted (layout differs), offset from static" if n in offset_only else \
               "forearm: not fitted, keeps base offsets" if n.startswith("Forearm") else \
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
        if a not in li or b not in li:          # reduced marker set
            continue
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
        write_static_html(model, lib, cloud, out, args.html,
                          f"{args.participant} static fit; suspects: {out['suspects'] or 'none'}")
        print(f"saved {args.html}")


if __name__ == "__main__":
    main()
