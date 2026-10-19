#!/usr/bin/env python3
"""Does P10's pinned Palm1 vertex hurt? Refit P10 Trial2 hands only three ways,
identical settings (P10 config, --cuda-graph --mv-chunk 3000 --mv-overlap 150):

  pin         P10's attachments as shipped (Palm1 pinned to vertex 200)
  nopin       the same minus Palm1 (derived from the config's attach file), so the static
              stage finds Palm1's attachment itself
  manual_pin  fitted to the manual-label export instead of SOMA's labels, pinned, as a
              stand-in reference until the real manual-label fit is available
  iters400    as pin but 400/400/600 iterations (--only iters400; not run by default)

SOMA arms use the labels run_fullset.py produced for P10 Trial2 hands only. Score
with scripts/mano/score_fit_vs_reference.py.

    python scripts/soma/palm1_pin_test.py --out results/perf/palm1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import provenance  # noqa: E402
import run_fullset as rf  # noqa: E402
from vicon2mano.core.dataset import DATA_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "results/perf/palm1"))
    ap.add_argument("--labels", default=str(REPO / "results/soma/fullset/P10/Trial2_Hands_only/soma_labelled.csv"))
    ap.add_argument("--manual", default=str(DATA_ROOT / "P10/Trial2_handsonly_manuallylabelled_filled.csv"))
    ap.add_argument("--static", default=str(DATA_ROOT / "P10/static_trial.csv"))
    ap.add_argument("--mano-dir", default=str(DATA_ROOT / "mano_v1_2" / "models"))
    ap.add_argument("--fit-config", default=str(REPO / "configs/surface_fit/P10.json"))
    ap.add_argument("--only", default="", help="comma list of arms to run")
    ap.add_argument("--code-dir", default=str(REPO),
                    help="checkout to run the fit script from (a frozen worktree, so edits to the live tree "
                         "cannot change a run in flight)")
    args = ap.parse_args()

    import json
    attach = json.loads((REPO / json.loads(Path(args.fit_config).read_text())["attach_file"]).read_text())
    nopin_file = Path(args.out) / "nopin" / "attach_minus_Palm1.json"       # derived, so it cannot drift
    nopin_file.parent.mkdir(parents=True, exist_ok=True)
    nopin_file.write_text(json.dumps({k: v for k, v in attach.items() if k != "Palm1"}))
    arms = {
        "pin": (args.labels, []),
        "nopin": (args.labels, ["--attach", str(nopin_file)]),
        "manual_pin": (args.manual, []),
        # the fit the earlier docs' "176 frames above 120 deg" came from: 400/400/600 iterations
        "iters400": (args.labels, ["--iters-thumb", "400", "--iters-finger", "400", "--iters-mv", "600"]),
    }
    want = [a for a in args.only.split(",") if a] or ["pin", "nopin", "manual_pin"]
    procs = []
    for name in want:
        movement, override = arms[name]
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        provenance.write(d / "fit_provenance.json", provenance.snapshot("driver", repo=Path(args.code_dir)))
        cmd = [sys.executable, "scripts/mano/fit_surface_markers.py", "--mano-dir", args.mano_dir,
               "--static", args.static, "--movement", movement, "--out", str(d / "static_fit.html")]
        cmd += rf.fit_args_from_config(Path(args.fit_config)) + override
        cmd += ["--cuda-graph", "--mv-chunk", "3000", "--mv-overlap", "150",
                "--npz-out", str(d / "fit.npz"), "--pred-out", str(d / "pred.npz")]
        log = open(d / "fit.log", "w", encoding="utf-8")
        procs.append((name, subprocess.Popen(cmd, cwd=args.code_dir, stdout=log, stderr=subprocess.STDOUT,
                                             env=dict(os.environ, PYTHONUNBUFFERED="1"))))
        print(f"started {name}", flush=True)
    bad = [n for n, p in procs if p.wait() != 0]
    print("done" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
