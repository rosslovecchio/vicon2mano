#!/usr/bin/env python3
"""Label a participant's movement trials with another SOMA model and compare with existing labels, slot by slot.

    python scripts/soma_all/compare_labels.py P16 --model results/soma/models/labeler_P16_final.pt \
        --old results/soma/fullset --new results/soma/retrain_final
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vicon2mano.core.dataset import find_trial_csv  # noqa: E402
from vicon2mano.core.loader import load_csv  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("participant")
ap.add_argument("--model", required=True)
ap.add_argument("--old", default="results/soma/fullset", help="<old>/<P>/<trial>/soma_labelled.csv")
ap.add_argument("--new", required=True, help="written as <new>/<P>/<trial>/soma_labelled.csv")
a = ap.parse_args()
P = a.participant
for session in ("Trial 1 Hands only", "Trial 1 HOI", "Trial2 Hands only", "Trial2 HOI"):
    sg = session.replace(" ", "_")
    d = REPO / a.new / P / sg
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "soma_labelled.csv").exists():
        with open(d / "label.log", "w") as f:
            subprocess.run([sys.executable, "-W", "ignore", "scripts/soma_all/eval_trial.py", "--model", a.model,
                            "--raw", find_trial_csv(P, session)[0], "--truth", "", "--csv-out", d / "soma_labelled.csv",
                            "--out", d / "labels.npz"], cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    A, na = load_csv(str(REPO / a.old / P / sg / "soma_labelled.csv"))[:2]
    B, nb = load_csv(str(d / "soma_labelled.csv"))[:2]
    na, nb = [n.split(":")[-1] for n in na], [n.split(":")[-1] for n in nb]
    B = B[:, [nb.index(n) for n in na]]
    fa, fb = np.isfinite(A).all(-1), np.isfinite(B).all(-1)
    same = (fa & fb & (np.linalg.norm(A - B, axis=-1) < 1e-3)) | (~fa & ~fb)
    hand = [i for i, n in enumerate(na) if not n.startswith("Forearm")]
    print(f"{P} {session}: {len(A)} frames; hand slots differing {100 * (1 - same[:, hand].mean()):.2f}% "
          f"(filled only in old {100 * (fa & ~fb)[:, hand].mean():.2f}%, only in new {100 * (~fa & fb)[:, hand].mean():.2f}%); "
          f"frames with any hand change {100 * (~same[:, hand]).any(1).mean():.1f}%; by marker: "
          + ", ".join(f"{na[i]} {100 * (1 - same[:, i].mean()):.1f}" for i in hand if (1 - same[:, i].mean()) > 0.01), flush=True)
