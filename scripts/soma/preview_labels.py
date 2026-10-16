#!/usr/bin/env python3
"""Side-by-side 3-D animation: Vicon's labels vs SOMA's, no fit needed.

Each finger is drawn as a chain (Palm2 -> <finger>1 -> 2 -> 3) in its own
colour, so a mislabelled marker shows up as a crossed, stretched or folded
finger. The window defaults to the stretch of the trial where the two
labellings disagree most.

    python scripts/soma/preview_labels.py --raw "<P12>/Static_trial01.csv" \\
        --labels results/soma/labels/P12/Trial1_handsonly/labels_P12model.npz --out results/soma/previews/P12_Trial1_handsonly_labels.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.loader import load_csv
from vicon2mano.strategies.soma.realdata import relabelled_markers

FINGERS = {"Thumb": "#d62728", "Index": "#ff7f0e", "Middle": "#2ca02c", "Ring": "#1f77b4", "Pinky": "#9467bd"}
PALM = ["Palm1", "Palm2", "Palm3", "Palm1"]
FOREARM = ["Forearm1", "Forearm2", "Forearm3", "Forearm4", "Forearm1"]


def chains(X, L):
    """Line segments per finger + palm + forearm, as (name, colour, (k,3) with NaN breaks)."""
    out = []
    for f, c in FINGERS.items():
        idx = [L.index("Palm2")] + [L.index(f"{f}{i}") for i in (1, 2, 3)]
        out.append((f, c, X[idx]))
    out.append(("Palm", "#444444", X[[L.index(n) for n in PALM]]))
    out.append(("Forearm", "#aaaaaa", X[[L.index(n) for n in FOREARM]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--labels", required=True, help="npz from eval_trial.py --out")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--frames", type=int, default=400, help="window length")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw, nr = load_csv(args.raw)[:2]
    nr = [n.split(":")[-1] for n in nr]
    z = np.load(args.labels)
    L = [str(x) for x in z["labels"]]
    col = [nr.index(l) for l in L]
    R = raw[:, col]
    v = np.where(z["voted"] >= 0, z["voted"], -1)[:, col]
    S = relabelled_markers(R, v, len(L))
    T = len(R)
    dis = (np.isfinite(R).all(-1) & (v != np.arange(len(L))[None])).sum(1)
    if args.start is None:
        w = np.convolve(dis, np.ones(args.frames), "valid")
        args.start = int(w.argmax())
    fr = np.arange(args.start, min(T, args.start + args.frames), args.stride)
    print(f"frames {fr[0]}..{fr[-1]} (stride {args.stride}); disagreeing points in window: {int(dis[fr].sum())}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def traces(t):
        tr = []
        for X, scene in ((R[t], "scene"), (S[t], "scene2")):
            for name, c, P in chains(X, L):
                tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines+markers",
                                       line=dict(color=c, width=6), marker=dict(size=4, color=c),
                                       name=name, showlegend=(scene == "scene"), scene=scene,
                                       hoverinfo="name"))
        return tr

    # fixed axis ranges so the hand does not jump between frames
    allp = np.concatenate([R[fr].reshape(-1, 3), S[fr].reshape(-1, 3)])
    allp = allp[np.isfinite(allp).all(1)]
    lo, hi = np.percentile(allp, 1, 0) - 20, np.percentile(allp, 99, 0) + 20
    axes = dict(xaxis=dict(range=[lo[0], hi[0]], visible=False), yaxis=dict(range=[lo[1], hi[1]], visible=False),
                zaxis=dict(range=[lo[2], hi[2]], visible=False), aspectmode="cube")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=["Vicon labels", "SOMA labels"])
    for tr in traces(fr[0]):
        fig.add_trace(tr)
    fig.frames = [go.Frame(data=traces(t), name=str(t),
                           layout=go.Layout(title=f"frame {t}  -  {dis[t]} points labelled differently"))
                  for t in fr]
    fig.update_layout(scene=axes, scene2=axes, height=750, width=1500,
                      title=f"frame {fr[0]}  -  {dis[fr[0]]} points labelled differently",
                      updatemenus=[dict(type="buttons", x=0.05, y=0.02, buttons=[
                          dict(label="Play", method="animate",
                               args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                          dict(label="Pause", method="animate",
                               args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                      sliders=[dict(steps=[dict(method="animate", label=str(t),
                                                args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                           for t in fr], x=0.15, len=0.8, y=0.02)])
    # rotate both views together is not built into plotly; start them from the same camera
    cam = dict(eye=dict(x=1.4, y=1.4, z=0.9))
    fig.update_layout(scene_camera=cam, scene2_camera=cam)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
