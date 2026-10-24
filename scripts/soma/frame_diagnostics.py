#!/usr/bin/env python3
"""Per-frame diagnostics of a fit: frames whose pose is not natural or changes too fast.

A fitted frame is flagged when
  pose    a joint is past the natural-pose limits by more than --pose deg (mv.py's check: anatomical limits,
          finger twist 5 deg, PIP/DIP abduction 2 deg)
  speed   a joint angle changes by more than --speed deg, or the hand's orientation by more than --turn deg,
          from the previous fitted frame
Marker status per frame (for the player): 0 observed, 1 gap-filled, 2 missing, 3 suspect (on a flagged finger;
every hand marker for a fast turn). Writes flags.json next to the fit.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma")]
import static_search as ss
from vicon2mano.core.loader import load_csv

ap = argparse.ArgumentParser()
ap.add_argument("--filled", required=True, help="gap-filled CSV (gapfill.py; its .npz says what was filled)")
ap.add_argument("--fit", required=True, help="mv.py output folder (fit.npz)")
ap.add_argument("--pose", type=float, default=2.0)
ap.add_argument("--speed", type=float, default=30.0, help="deg per fitted frame")
ap.add_argument("--turn", type=float, default=45.0, help="deg per fitted frame")
ap.add_argument("--hinge", type=float, default=2.0)
a = ap.parse_args()

J = ss.Judge()
fit = np.load(Path(a.fit) / "fit.npz")
fr, mk = fit["frames_fitted"], [str(n) for n in fit["names"]]
T = len(fr)
X, names = load_csv(a.filled)[:2]
names = [n.split(":")[-1] for n in names]
X = X[fr][:, [names.index(n) for n in mk]]
h = np.load(Path(a.filled).with_suffix(".npz"))
how = h["how"][fr][:, [list(h["names"]).index(n) for n in mk]]
status = np.where(np.isfinite(X).all(-1), np.where(how == 0, 0, 1), 2).astype(np.uint8)

JN = J.fsm._FINGER_JOINTS
c = np.degrees(np.einsum("tjk,jkc->tjc", fit["hand_pose"].reshape(T, 15, 3), J.B))
lo, hi = J.lo.copy(), J.hi.copy()
lo[:12, 2], hi[:12, 2] = -5, 5
H = [j for j, n in enumerate(JN) if not n.startswith("thumb") and n[-1] in "23"]
lo[H, 1], hi[H, 1] = -a.hinge, a.hinge
ex = np.maximum(lo - c, c - hi).clip(0)                                     # (T, 15, 3) deg past the limits
dc = np.r_[np.zeros((1, 15, 3)), np.abs(np.diff(c, axis=0))]
R = Rotation.from_rotvec(fit["global_orient"])
turn = np.r_[0, np.degrees((R[1:] * R[:-1].inv()).magnitude())]
finger_of = {n: n.rstrip("123").capitalize() for n in JN}
COMP = ("flexion", "abduction", "twist")

reasons, count = [], dict(pose=0, speed=0, turn=0)
for t in range(T):
    r, bad = [], set()
    jk = list(zip(*np.nonzero(ex[t] > a.pose)))
    if jk:
        r.append("pose: " + ", ".join(f"{JN[j]} {COMP[k]} {c[t, j, k]:.0f} deg ({ex[t, j, k]:.0f} past)" for j, k in jk))
        bad |= {finger_of[JN[j]] for j, _ in jk}
        count["pose"] += 1
    jk = list(zip(*np.nonzero(dc[t] > a.speed)))
    if jk:
        r.append("fast: " + ", ".join(f"{JN[j]} {COMP[k]} {dc[t, j, k]:.0f} deg in one step" for j, k in jk))
        bad |= {finger_of[JN[j]] for j, _ in jk}
        count["speed"] += 1
    if turn[t] > a.turn:
        r.append(f"fast: hand turned {turn[t]:.0f} deg in one step")
        bad |= {"Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"}
        count["turn"] += 1
    for i, n in enumerate(mk):
        if n.rstrip("123") in bad and status[t, i] != 2:
            status[t, i] = 3
    reasons.append(r)

flagged = [t for t in range(T) if reasons[t]]
out = dict(thresholds=dict(pose_deg=a.pose, speed_deg_per_step=a.speed, turn_deg_per_step=a.turn), frames=T,
           flagged=len(flagged), by_check=count, reasons=reasons, status=status.tolist(),
           filled_pct=round(100 * float((status == 1).mean()), 1), missing_pct=round(100 * float((status == 2).mean()), 1))
(Path(a.fit) / "flags.json").write_text(json.dumps(out))
print(f"{len(flagged)} of {T} frames flagged; by check {count}; markers filled {out['filled_pct']}%, missing {out['missing_pct']}%")
print("flagged source frames:", [int(fr[t]) for t in flagged][:40])
