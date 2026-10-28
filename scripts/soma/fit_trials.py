#!/usr/bin/env python3
"""Gap-fill each SOMA-labelled trial of a participant, then fit, diagnose and render it in 10k-frame batches.

Every 10th frame is fitted with the participant's final static. Everything is copied to OneDrive
soma_transfer/results/soma/<name>/<trial>/. Batches already on OneDrive are skipped, and a filled CSV already in
--work is reused, so a killed run resumes.

    python scripts/soma/fit_trials.py P16                                   # rigid fill -> results/soma/movement
    python scripts/soma/fit_trials.py P16 --filler results/soma/real_fill/filler_{p}_all.pt --max-gap 100000 \
        --work results/soma/real_fill/movement --name {p}_realfill --note "real-data gap fill"
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vicon2mano.core.dataset import DATA_ROOT  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("participants", nargs="+")
ap.add_argument("--labels", default="results/soma/fullset", help="<labels>/<P>/<trial>/soma_labelled.csv")
ap.add_argument("--work", default="results/soma/movement")
ap.add_argument("--name", default="{p}", help="OneDrive folder under soma_transfer/results/soma")
ap.add_argument("--filler", help="gap_fill.py --filler ({p} = participant)")
ap.add_argument("--max-gap", type=int)
ap.add_argument("--w-filled", type=float, default=1.0, help="movement_fit.py --w-filled")
ap.add_argument("--note", default="", help="added to the viewer title")
a = ap.parse_args()
DATA = DATA_ROOT / "soma_transfer"
BATCH, NB, STEP = 10000, 5, 10


def sh(args, log):
    r = subprocess.run([sys.executable, "-W", "ignore"] + [str(x) for x in args], cwd=REPO, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{args[0]} failed, see {log}")
    return r.stdout


for p in a.participants:
    static = DATA / f"static_final/{p}/{p}_static_final_params.npz"
    for tdir in sorted(d for d in (REPO / a.labels / p).iterdir() if (d / "soma_labelled.csv").exists() and d.name != "static"):
        trial = tdir.name
        w, one = REPO / a.work / p / trial, DATA / "results/soma" / a.name.format(p=p) / trial
        w.mkdir(parents=True, exist_ok=True)
        one.mkdir(parents=True, exist_ok=True)
        try:
            filled = w / "soma_labelled_filled.csv"
            if not filled.exists():
                extra = (["--filler", a.filler.format(p=p)] if a.filler else []) + (["--max-gap", a.max_gap] if a.max_gap else [])
                sh(["scripts/soma/gap_fill.py", "--csv", tdir / "soma_labelled.csv", "--out", filled] + extra, w / "gap_fill.log")
                for suf, dst in ((".csv", "_soma_labelled_filled.csv"), (".json", "_soma_labelled_filled_report.json"),
                                 (".npz", "_soma_labelled_filled_status.npz")):
                    shutil.copy(filled.with_suffix(suf), one / f"{p}_{trial}{dst}")
            T = sum(1 for _ in open(filled)) - 1
            for b in range(NB):
                s, e = b * BATCH, min(T, (b + 1) * BATCH)
                if s >= T:
                    break
                tag = f"{s}-{e}_every{STEP}"
                html = one / f"{p}_{trial}_{tag}_live.html"
                if html.exists():
                    continue
                bd = w / tag
                sh(["scripts/soma/movement_fit.py", "--participant", p, "--static", static, "--csv", filled, "--out", bd,
                    "--first", s, "--max-frames", e, "--step", STEP, "--chunk", 1000, "--iters", 800,
                    "--w-filled", a.w_filled], w / f"{tag}_fit.log")
                flags = next(l for l in sh(["scripts/soma/frame_diagnostics.py", "--filled", filled, "--fit", bd],
                                           w / f"{tag}_diag.log").splitlines() if "flagged" in l)
                title = f"{p} {trial.replace('_', ' ')}, frames {s}-{e - 1} every {STEP}, gap-filled + fit" + (f", {a.note}" if a.note else "")
                sh(["scripts/soma/live_viewer.py", "--fit", bd / "fit.npz", "--csv", bd / "markers.csv", "--flags", bd / "flags.json",
                    "--out", bd / html.name, "--title", title], w / f"{tag}_view.log")
                for f in ("fit.npz", "pred.npz", "summary.json", "flags.json"):
                    shutil.copy(bd / f, one / f"{p}_{trial}_{tag}_{f}")
                shutil.copy(bd / html.name, html)
                m = json.loads((bd / "summary.json").read_text())["max_dist_mm"]
                print(f"{p} {trial} {tag}: max dist median {m['median']} p95 {m['p95']} | {flags}", flush=True)
        except Exception as ex:                                              # one bad trial must not stop the rest
            print(f"{p} {trial}: FAILED {ex}", flush=True)
    print(f"{p} done", flush=True)
