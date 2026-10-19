#!/usr/bin/env python3
"""Record which version of the fit code a fit started with.

The fit script can change on disk while a long run is going, and a fit that
started earlier keeps the version it imported. ``snapshot()`` captures that
(commit, tracked files with uncommitted changes, content hashes of the code the
fit uses); run_fullset.py writes it to ``<trial>/fit_provenance.json`` just
before each fit starts.

For fits already running under a driver that does not write it, ``--watch``
polls the trial folders and records the version found when each fit's log
appears (``source: "watcher"``, within a few seconds of the start). ``--reconstruct``
fills in trials that started before the watcher from file times and the git
history (``source: "reconstructed"``): exact when no uncommitted edit was in
flight, and it says so.

    python scripts/soma/provenance.py --watch P6,P7,P8
    python scripts/soma/provenance.py --reconstruct P6,P7,P8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = ["scripts/mano/fit_surface_markers.py", "vicon2mano/core/fitter.py", "scripts/soma/run_fullset.py"]


def _git(*args, repo: Path = REPO, strip: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout
    return out.strip() if strip else out


def snapshot(source: str = "driver", started_at: float | None = None, repo: Path = REPO) -> dict:
    """The code state of ``repo`` right now. Hashes are of the files on disk (what a new process imports)."""
    files = {}
    for rel in CODE:
        p = repo / rel
        if p.exists():
            data = p.read_bytes()
            files[rel] = dict(sha256=hashlib.sha256(data).hexdigest()[:16],
                              git_blob=_git("hash-object", rel, repo=repo)[:10], mtime=time.strftime(
                                  "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)))
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=no", repo=repo, strip=False).splitlines()]
    return dict(source=source, repo=str(repo),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at or time.time())),
                git_head=_git("rev-parse", "--short", "HEAD", repo=repo),
                git_branch=_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
                tracked_files_with_uncommitted_changes=dirty, files=files)


def write(path: Path, snap: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1), encoding="utf-8")


def _trial_dirs(out: Path, participants: list[str]):
    for p in participants:
        for t in sorted((out / p).glob("*/")):
            if t.is_dir() and not t.name.startswith("_") and t.name != "static":
                yield p, t


def _ctime(path: Path) -> float:
    return path.stat().st_ctime            # creation time on Windows


def watch(out: Path, participants: list[str], poll: float = 15.0, give_up_min: float = 600.0):
    t_end = time.time() + give_up_min * 60
    while time.time() < t_end:
        pending = 0
        for p, t in _trial_dirs(out, participants):
            prov = t / "fit_provenance.json"
            if prov.exists():
                continue
            if (t / "fit.log").exists():
                write(prov, snapshot("watcher", started_at=_ctime(t / "fit.log")))
                print(f"[{time.strftime('%H:%M:%S')}] {p}/{t.name}: recorded", flush=True)
            elif not (t / "fit.npz").exists():
                pending += 1
        if not pending and all((t / "fit_provenance.json").exists() for _, t in _trial_dirs(out, participants)) \
                and len(list(_trial_dirs(out, participants))) >= 4 * len(participants):
            break
        time.sleep(poll)


def reconstruct(out: Path, participants: list[str]):
    """Provenance for fits that started before anyone recorded it, from file times + git history."""
    script = REPO / CODE[0]
    mtime = script.stat().st_mtime
    commits = _git("log", "--format=%h %ct", "--", CODE[0]).splitlines()          # newest first
    for p, t in _trial_dirs(out, participants):
        prov, log = t / "fit_provenance.json", t / "fit.log"
        if prov.exists() or not log.exists():
            continue
        start = _ctime(log)
        head_then = next((c.split()[0] for c in commits if int(c.split()[1]) <= start), None)
        later_edit = start < mtime
        snap = dict(source="reconstructed",
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))
        snap["note"] = (f"fit started {time.strftime('%H:%M:%S', time.localtime(start))}; the script's last "
                        f"write on disk was {time.strftime('%H:%M:%S', time.localtime(mtime))}. Last commit "
                        f"touching it before the start: {head_then}. "
                        + ("It started before the last on-disk edit, so it used the committed version "
                           f"{head_then} unless an earlier uncommitted edit was in flight (not recoverable)."
                           if later_edit else "It started after the last on-disk edit, so it used the version on disk now."))
        snap["script_version_used"] = head_then if later_edit else "working tree at last edit"
        # what the fit script was then, not what the repository is now
        used = head_then if later_edit else None
        snap["git_head_of_fit_script"] = used
        snap["fit_script_git_blob"] = (_git("rev-parse", f"{used}:{CODE[0]}")[:10] if used
                                       else _git("hash-object", CODE[0])[:10])
        snap["other_files"] = "not reconstructed (no record of their state at the start)"
        write(prov, snap)
        print(f"{p}/{t.name}: {snap['script_version_used']}  ({snap['note'][:90]}...)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(REPO / "results/soma/fullset"))
    ap.add_argument("--watch", default="")
    ap.add_argument("--reconstruct", default="")
    args = ap.parse_args()
    out = Path(args.out_root)
    if args.reconstruct:
        reconstruct(out, args.reconstruct.split(","))
    if args.watch:
        watch(out, args.watch.split(","))


if __name__ == "__main__":
    main()
