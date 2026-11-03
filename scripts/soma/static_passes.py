#!/usr/bin/env python3
"""Repeated static_final_fit.py passes for one participant, each starting from the best pass so far.

Best = no marker off the dorsal side or inside the skin, then no pose past the limits, then the smallest max
marker-to-attachment distance. Stops when a pass meets the goal (static_final_fit's goal_met) or after --passes.
Writes <out>/passes.json: per pass its start, arguments, input and output sha256, the scores and goal_met. The pass to
accept is still chosen by reviewing the views; ``best`` in passes.json is only the rule's pick.

    python scripts/soma/static_search.py --participants P16 --palm-manual configs/surface_fit/P4_palm_pins.json \\
        --radius 8 --arm palm_manual_r8
    python scripts/soma/static_passes.py --participant P16 --start palm_manual_r8 --out results/soma/static_final_fit_along/P16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import static_search as ss  # noqa: E402


def sha(path: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_file(start: str, participant: str) -> Path:
    """The params file static_final_fit.py reads for ``--start`` (an output folder or a static_search arm)."""
    d = Path(start) if Path(start).is_dir() else ss.OUT / participant / start
    return d / "params.npz" if (d / "params.npz").exists() else d / "static_params.npz"


def rank(s: dict) -> tuple:
    """Sort key: valid placement first, then a pose within the limits, then the smallest max distance."""
    return s["not_dorsal"] + s["inside"] > 0, s["pose_past_limits_deg"] > 0, s["max_mm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--start", required=True, help="static_search arm (e.g. palm_manual_r8) or a static_final_fit folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=7)
    ap.add_argument("--fit-args", default="--radius 15 --along", help="passed to every static_final_fit.py pass")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec, start = [], a.start
    for i in range(a.passes):
        d = out / f"pass{i}"
        args = ["--participant", a.participant, "--start", str(start), "--out", str(d)] + a.fit_args.split()
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / "static_final_fit.py"), *args], check=True,
                       stdout=open(out / f"pass{i}.log", "w"), stderr=subprocess.STDOUT)
        s = json.loads((d / "summary.json").read_text())
        rec.append(dict(name=d.name, start=str(start), args=args, input_sha256=sha(params_file(str(start), a.participant)),
                        output_sha256=sha(d / "params.npz"), **{k: s[k] for k in ("goal_met", "max_mm", "worst",
                        "pose_past_limits_deg", "judge_natural_deg", "not_dorsal", "inside", "scale")}))
        print(f"{d.name} <- {Path(str(start)).name}: max {s['max_mm']} mm, goal {s['goal_met']}", flush=True)
        if s["goal_met"]:
            break
        start = out / min(rec, key=rank)["name"]
    best = min(rec, key=rank)["name"]
    (out / "passes.json").write_text(json.dumps(dict(participant=a.participant, best=best, passes=rec), indent=1))
    print(f"best by the rule: {best} (accept only after reviewing {out / best / 'static.html'})")


if __name__ == "__main__":
    main()
