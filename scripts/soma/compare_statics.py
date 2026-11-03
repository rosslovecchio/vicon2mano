#!/usr/bin/env python3
"""Compare each participant's historical accepted static with the static_passes.py pick, without changing either.

Historical: soma_transfer/static_final/<P>/<P>_static_final_params.npz, matched by sha256 to the pass it came from
(results/soma/static_final_fit_along/<P>/pass*/). Script: the ``best`` pass in <passes>/<P>/passes.json.

    python scripts/soma/compare_statics.py --passes results/soma/static_passes --out results/soma/static_passes/compare.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vicon2mano.core.dataset import DATA_ROOT  # noqa: E402

ACCEPTED = DATA_ROOT / "soma_transfer/static_final"
HISTORY = REPO / "results/soma/static_final_fit_along"


def row(d: Path) -> dict:
    """Scores of one static_final_fit.py output folder."""
    s, z = json.loads((d / "summary.json").read_text()), np.load(d / "params.npz")
    return dict(pass_=d.name, max_mm=s["max_mm"], median_mm=round(float(np.median(list(s["marker_mm"].values()))), 2),
                scale=round(s["scale"], 4), checks=s["not_dorsal"] == 0 and s["inside"] == 0 and s["pose_past_limits_deg"] == 0,
                natural_deg=s["judge_natural_deg"], goal=s["goal_met"], att=z["att"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default=str(REPO / "results/soma/static_passes"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for pj in sorted(Path(a.passes).glob("*/passes.json")):
        p = pj.parent.name
        new = row(pj.parent / json.loads(pj.read_text())["best"])
        acc = ACCEPTED / p / f"{p}_static_final_params.npz"
        h = hashlib.sha256(acc.read_bytes()).hexdigest() if acc.exists() else None
        old = next((row(d) for d in sorted(HISTORY.glob(f"{p}/pass*")) if (d / "params.npz").exists()
                    and hashlib.sha256((d / "params.npz").read_bytes()).hexdigest() == h), None)
        r = dict(participant=p, **{f"script_{k}": v for k, v in new.items() if k != "att"})
        if old:
            r |= {f"historical_{k}": v for k, v in old.items() if k != "att"}
            r["attachments_changed"] = int((old["att"] != new["att"]).sum())
        else:
            r["historical_pass_"] = "not from the pass runs" if h else "no accepted static"
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
