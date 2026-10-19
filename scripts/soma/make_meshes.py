#!/usr/bin/env python3
"""Render the fitted-MANO mesh animation for every trial that run_fullset.py has
finished fitting: results/soma/mesh/<P>/<P>_<trial>.html (the same naming and viewer
as the earlier P10/P12 meshes; scripts/mano/animate_mesh.py does the rendering).

A trial is ready when its ``angles.csv`` exists (fit and flags done). Trials
that already have a mesh are skipped, so it is safe to run repeatedly or to
keep ``--watch`` running next to a long fit.

The overlay is the SOMA-relabelled markers the fit actually used. Playback rate
comes from the raw CSV's ``Trajectories`` header (200 Hz where present); the
single-header exports have none, so 200 Hz is assumed and logged.

    python scripts/soma/make_meshes.py                 # render what is ready, once
    python scripts/soma/make_meshes.py --watch         # keep going until the fit run is done
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_fullset as rf  # noqa: E402
from vicon2mano.core.dataset import DATA_ROOT, REF_CSV, load_trial_sessions  # noqa: E402

DEFAULT_HZ = 200.0


def capture_rate(csv_path: Path) -> float | None:
    """Marker rate from the 'Trajectories' section header, or None."""
    try:
        with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                if line.strip() == "Trajectories":
                    return float(next(f).split(",")[0])
    except (OSError, ValueError, StopIteration):
        pass
    return None


def trial_list(out: Path, participants: list[str]):
    """[(participant, slug, raw_csv, fit_dir)] for every logged trial of these participants."""
    sessions = load_trial_sessions(REF_CSV)
    rows = []
    for p in participants:
        for _session, sg, raw in rf.resolve_trials(p, sessions.get(p, {}), DATA_ROOT / p):
            if raw is not None:
                rows.append((p, sg, raw, out / p / sg))
    return rows


def flagged_participants(status: Path) -> set[str]:
    if not status.exists():
        return set()
    out = set()
    for line in status.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("stage") == "FLAG":
            out.add(r["participant"])
    return out


def render(p, sg, raw, tdir, mesh_dir, args) -> bool:
    final = mesh_dir / p / f"{p}_{sg}.html"
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.stem + ".part.html")
    hz = capture_rate(raw)
    note = "" if hz else f" (no rate in header: assuming {DEFAULT_HZ:.0f} Hz)"
    t0 = time.time()
    log = tdir / "mesh.log"
    cmd = [sys.executable, "scripts/mano/animate_mesh.py", "--npz", tdir / "fit.npz",
           "--csv", tdir / "soma_labelled.csv", "--out", tmp, "--mano-dir", args.mano_dir,
           "--n-out", args.n_out, "--fps", hz or DEFAULT_HZ]
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                            env=dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")).returncode
    if rc != 0 or not tmp.exists():
        print(f"[{time.strftime('%H:%M:%S')}] {p} {sg}: mesh FAILED, see {log}", flush=True)
        return False
    os.replace(tmp, final)
    print(f"[{time.strftime('%H:%M:%S')}] {p} {sg}: {final.name} "
          f"{final.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s{note}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="P3,P4,P9,P10,P11,P12,P13,P14,P15,P16,P17")
    ap.add_argument("--out-root", default=str(REPO / "results/soma/fullset"))
    ap.add_argument("--mesh-dir", default=str(REPO / "results/soma/mesh"))
    ap.add_argument("--mano-dir", default=str(DATA_ROOT / "mano_v1_2" / "models"))
    ap.add_argument("--n-out", type=int, default=1500, help="frames sampled into each animation")
    ap.add_argument("--watch", action="store_true", help="keep rendering as fits finish, until the run is done")
    ap.add_argument("--poll-min", type=float, default=5.0)
    ap.add_argument("--idle-min", type=float, default=60.0,
                    help="--watch stops once nothing is left to render and the fit run has been quiet this long")
    args = ap.parse_args()

    out, mesh_dir = Path(args.out_root), Path(args.mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    ps = [x.strip() for x in args.participants.split(",") if x.strip()]
    status = out / "status.jsonl"
    failed: set[tuple[str, str]] = set()
    while True:
        rows = trial_list(out, ps)
        todo = [r for r in rows if (r[3] / "angles.csv").exists() and (r[3] / "fit.npz").exists()
                and not (mesh_dir / r[0] / f"{r[0]}_{r[1]}.html").exists() and (r[0], r[1]) not in failed]
        for p, sg, raw, tdir in todo:
            if not render(p, sg, raw, tdir, mesh_dir, args):
                failed.add((p, sg))
        if not args.watch:
            break
        skip = flagged_participants(status)
        left = [r for r in rows if r[0] not in skip and (r[0], r[1]) not in failed
                and not (mesh_dir / r[0] / f"{r[0]}_{r[1]}.html").exists()]
        quiet = (time.time() - status.stat().st_mtime) / 60 if status.exists() else 1e9
        if not left:
            print("all trials have a mesh (or their participant is flagged)", flush=True)
            break
        if not todo and quiet > args.idle_min:
            print(f"fit run quiet for {quiet:.0f} min; {len(left)} trial(s) never finished fitting: "
                  f"{[(r[0], r[1]) for r in left][:6]}", flush=True)
            break
        if not todo:
            time.sleep(args.poll_min * 60)
    if failed:
        print(f"{len(failed)} mesh render(s) failed: {sorted(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
