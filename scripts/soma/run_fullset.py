#!/usr/bin/env python3
"""Run the SOMA pipeline for many participants: static -> calibrate -> train ->
label every trial -> surface fit -> frame flags -> angles CSV.

Stages per participant (each skipped if its output already exists, so a killed
run resumes where it stopped; ``--force`` redoes them):

  static    SOMA-relabel the participant's static with the base model (Vicon's
            labels on statics are often wrong), then a cheap static-only surface
            fit as a quality gate.
  calibrate scripts/soma/calibrate_static.py -> libraries/library_<P>.npz
  train     scripts/soma/train.py            -> models/labeler_<P>.pt
  label     scripts/soma/eval_trial.py       -> <P>/<trial>/soma_labelled.csv
  fit       scripts/mano/fit_surface_markers.py (CUDA graph, chunk overlap)
  flags     scripts/soma/flag_frames.py + angles.csv (angles with the flags)

The static is ASSUMED good. A participant is flagged and left alone (never
guessed around) when it has no static, or when the static looks bad: suspect
markers, an extreme fitted shape, hand markers absent, or a poor static
surface fit. Flagged participants are listed in ``<out>/flagged.csv``; fix the
static and re-run, the other participants are unaffected.

Trial identity comes from manual_frames.csv (``Session`` -> ``CSV name``), as
everywhere else in the repo. Only that log's participants are known (P1-P17);
Px/Pxh have no trial list yet. P1 and P2 are skipped by default (different
marker set, handled elsewhere).

    # what would run, nothing executed
    python scripts/soma/run_fullset.py --dry-run

    # everything, one process
    python scripts/soma/run_fullset.py

    # two processes (or two machines) share the work
    python scripts/soma/run_fullset.py --shard 0/2
    python scripts/soma/run_fullset.py --shard 1/2

    # quick smoke test: one trial, first 2000 frames
    python scripts/soma/run_fullset.py --participants P12 --trials "Trial 1 Hands only" --max-frames 2000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vicon2mano.core.dataset import (  # noqa: E402
    DATA_ROOT, MANO_ONLY_STATIC, REF_CSV, find_manual_csv, find_session_file,
    load_trial_sessions, participant_sort_key, slug)

PY = sys.executable
BASE = "P10"                       # the participant the base library/model come from
STAGES = ("static", "calibrate", "train", "label", "fit", "flags")
FINGERS = ["index", "middle", "pinky", "ring", "thumb"]          # smplx hand_pose order
JOINT_COLS = [f"{f}_{j}_deg" for f in FINGERS for j in ("mcp", "pip", "dip")]
STATIC_STEMS = {"static", "statictrial"}                         # exact, after normalising


# --------------------------------------------------------------------- planning

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_static(p: str, sessions: dict[str, str], pdir: Path,
                   overrides: dict[str, str] | None = None) -> tuple[Path | None, str]:
    """(static csv, how it was found). Never guesses beyond an exact name match."""
    if overrides and p in overrides:
        f = Path(overrides[p])
        return (f, "override") if f.exists() else (None, f"override missing: {f}")
    name = sessions.get("Static") or sessions.get("static")
    if name:
        f = find_session_file(pdir, name)
        if f is not None:
            return f, "manual_frames.csv"
    extra = MANO_ONLY_STATIC.get(p)
    if extra and (pdir / extra).exists():
        return pdir / extra, "MANO_ONLY_STATIC"
    cands = [f for f in sorted(pdir.glob("*.csv")) if _norm(f.stem) in STATIC_STEMS]
    if len(cands) == 1:
        return cands[0], "file name"
    if len(cands) > 1:
        return None, f"ambiguous: {[c.name for c in cands]}"
    return None, "none found"


def resolve_trials(p: str, sessions: dict[str, str], pdir: Path):
    """[(session, slug, path or None)] for every logged session except the static."""
    out = []
    for session, csv_name in sessions.items():
        if session.lower() == "static":
            continue
        out.append((session, slug(session), find_session_file(pdir, csv_name)))
    return out


def fit_args_from_config(cfg_path: Path) -> list[str]:
    """CLI flags for fit_surface_markers.py from a configs/surface_fit/*.json."""
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    rename = {"stretch_file": "stretch", "mv_chunk": None}
    args = ["--side", cfg.get("side", "right")]
    if cfg.get("attach_file"):
        args += ["--attach", str(REPO / cfg["attach_file"])]
    for k, v in cfg.get("params", {}).items():
        name = rename.get(k, k)
        if name is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(v, bool):
            if v and k not in ("freeze_root", "marker_init"):    # already their defaults
                args.append(flag)
        else:
            val = str(REPO / v) if k == "stretch_file" else str(v)
            args += [flag, val]
    return args


def parse_calibration(text: str) -> dict:
    """Quality facts from calibrate_static.py's printout."""
    m = re.search(r"betas:\s*\[([^\]]*)\]", text)
    betas = [float(x) for x in m.group(1).split(",") if x.strip()] if m else []
    return dict(
        max_abs_beta=max((abs(b) for b in betas), default=float("nan")),
        suspects=re.findall(r"^(\w+)\s+[-\d.]+\s+SUSPECT", text, flags=re.M),
        absent=re.findall(r"^(\w+)\s+-\s+absent in static", text, flags=re.M),
        n_segments_outside=text.count("<- outside base p5-p95"),
    )


def parse_static_fit_mm(text: str) -> float | None:
    """Last 'MEAN |d|' in a fit_surface_markers log (= the stage-2 surface fit)."""
    v = re.findall(r"MEAN \|d\|\s+([\d.]+) mm", text)
    return float(v[-1]) if v else None


def static_verdict(cal: dict, fit_mm: float | None, args) -> list[str]:
    """Reasons the static looks bad (empty = assume good)."""
    why = []
    if cal["suspects"]:
        why.append(f"suspect markers after the static fit: {cal['suspects']}")
    if cal["max_abs_beta"] == cal["max_abs_beta"] and cal["max_abs_beta"] > args.max_beta:
        why.append(f"extreme fitted shape: max |beta| {cal['max_abs_beta']:.1f} > {args.max_beta}")
    hand_absent = [m for m in cal["absent"] if not m.startswith("Forearm")]
    if len(hand_absent) > args.max_absent:
        why.append(f"hand markers absent in the static: {hand_absent}")
    if fit_mm is not None and fit_mm > args.max_static_fit_mm:
        why.append(f"static surface fit {fit_mm:.1f} mm > {args.max_static_fit_mm}")
    return why


# -------------------------------------------------------------------- execution

class Run:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out_root)
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = self.out / "status.jsonl"
        self.flagged: list[dict] = []
        self.failed = 0

    def note(self, p, trial, stage, status, secs=0.0, detail=""):
        rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), participant=p, trial=trial,
                   stage=stage, status=status, seconds=round(secs, 1), detail=detail)
        with open(self.status, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{rec['t'][11:]}] {p:4s} {trial or '-':22s} {stage:9s} {status}"
              f"{' (%.0fs)' % secs if secs else ''}{'  ' + detail if detail else ''}", flush=True)

    def flag(self, p, reason, detail):
        self.flagged.append(dict(participant=p, reason=reason, detail=detail))
        self.note(p, "", "FLAG", reason, detail=detail)

    def sh(self, cmd, log: Path) -> tuple[int, str, float]:
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                cwd=REPO, env=env).returncode
        return rc, log.read_text(encoding="utf-8", errors="replace"), time.time() - t0


def part(path: Path) -> Path:
    """Temporary name for an output (same extension): renamed only on success."""
    return path.with_name(path.stem + ".part" + path.suffix)


def commit(pairs):
    for tmp, final in pairs:
        os.replace(tmp, final)


def step(R: Run, p, trial, stage, outputs, cmd, log, parse=None):
    """Run ``cmd`` unless every final output exists. Outputs are (tmp, final)
    pairs already substituted into ``cmd``. Returns (ok, log text or '')."""
    if not R.a.force and all(f.exists() for _, f in outputs):
        R.note(p, trial, stage, "done before")
        return True, ""
    rc, text, secs = R.sh(cmd, log)
    if rc != 0:
        R.failed += 1
        R.note(p, trial, stage, "FAILED", secs, f"see {log}")
        return False, text
    commit(outputs)
    R.note(p, trial, stage, "ok", secs)
    return True, text


def write_angles(fit_npz: Path, flags_csv: Path, out_csv: Path):
    import numpy as np
    import pandas as pd
    fl = pd.read_csv(flags_csv)
    z = np.load(fit_npz)
    fr = fl["frame"].to_numpy()
    hp = z["hand_pose"][fr].reshape(len(fr), 15, 3)
    df = fl[["frame", "flag", "label_signal", "fit_signal", "angle_signal"]].copy()
    mag = np.degrees(np.linalg.norm(hp, axis=-1))
    for i, c in enumerate(JOINT_COLS):
        df[c] = mag[:, i].astype("float32")
    for i in range(45):
        df[f"hp_{i:02d}"] = hp.reshape(len(fr), 45)[:, i].astype("float32")
    for i in range(3):
        df[f"orient_{i}"] = z["global_orient"][fr][:, i].astype("float32")
        df[f"transl_{i}_m"] = z["transl"][fr][:, i].astype("float32")
    tmp = part(out_csv)
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)


def do_participant(R: Run, p: str, sessions: dict[str, str]):
    a = R.a
    pdir = DATA_ROOT / p
    pout = R.out / p
    static, how = resolve_static(p, sessions, pdir, a.static_map)
    trials = [t for t in resolve_trials(p, sessions, pdir)
              if not a.trials or t[0] in a.trials or t[1] in a.trials]
    missing = [t[0] for t in trials if t[2] is None]
    if missing:
        R.flag(p, "trial file missing", f"{missing} (manual_frames.csv names a file that is not on disk)")
    trials = [t for t in trials if t[2] is not None]
    if static is None:
        R.flag(p, "no static", how)
        return
    R.note(p, "", "static", "using", detail=f"{static.name} [{how}]")

    mano = a.mano_dir
    lib_dir, mdl_dir = REPO / "results/soma/libraries", REPO / "results/soma/models"
    base_lib = Path(a.base_library)
    base_model = Path(a.base_model)
    is_base = p == BASE
    lib = base_lib if is_base else lib_dir / f"library_{p}.npz"
    model = base_model if is_base else mdl_dir / f"labeler_{p}.pt"
    stages = set(a.stages)

    # ---- static: relabel with the base model (not for the base participant)
    static_use = static
    if "static" in stages and not is_base:
        sl, sa = pout / "static/soma_labelled.csv", pout / "static/labels.npz"
        ok, _ = step(R, p, "static", "relabel",
                     [(part(sl), sl), (part(sa), sa)],
                     [PY, "scripts/soma/eval_trial.py", "--model", base_model, "--raw", static,
                      "--truth", "", "--csv-out", part(sl), "--out", part(sa)],
                     pout / "static/relabel.log")
        if not ok:
            return
        static_use = sl
    elif not is_base:
        static_use = pout / "static/soma_labelled.csv"

    # ---- calibrate: shape + per-marker offsets from the static
    cal = {}
    if "calibrate" in stages and not is_base:
        ok, text = step(R, p, "", "calibrate", [(part(lib), lib)],
                        [PY, "scripts/soma/calibrate_static.py", "--participant", p, "--static", static_use,
                         "--base-library", base_lib, "--mano-dir", mano, "--out", part(lib),
                         "--html", pout / "static/calibration.html"],
                        pout / "static/calibrate.log")
        if not ok:
            return
        if not text:                                      # reused: re-read the old log if there is one
            lg = pout / "static/calibrate.log"
            text = lg.read_text(encoding="utf-8", errors="replace") if lg.exists() else ""
        cal = parse_calibration(text) if text else {}

    # ---- static quality gate: cheap static-only surface fit
    fit_mm = None
    if "static" in stages:
        sf = pout / "static/surface_fit.html"
        cmd = [PY, "scripts/mano/fit_surface_markers.py", "--mano-dir", mano, "--static", static_use,
               "--out", part(sf)] + fit_args_from_config(a.fit_config)
        ok, text = step(R, p, "", "staticfit", [(part(sf), sf)], cmd, pout / "static/surface_fit.log")
        if not ok:
            return
        if not text:
            lg = pout / "static/surface_fit.log"
            text = lg.read_text(encoding="utf-8", errors="replace") if lg.exists() else ""
        fit_mm = parse_static_fit_mm(text)
    if cal or fit_mm is not None:
        why = static_verdict(cal or dict(suspects=[], max_abs_beta=float("nan"), absent=[], n_segments_outside=0),
                             fit_mm, a)
        info = (f"max|beta| {cal.get('max_abs_beta', float('nan')):.1f}, "
                f"{cal.get('n_segments_outside', 0)} segments outside base range, "
                f"static surface fit {fit_mm if fit_mm is not None else float('nan'):.1f} mm")
        if why:
            R.flag(p, "static looks bad", "; ".join(why) + f"  [{info}]")
            return
        R.note(p, "", "static", "assumed good", detail=info)

    # ---- train the participant's labeller
    if "train" in stages and not is_base:
        if not lib.exists():
            R.note(p, "", "train", "SKIPPED", detail=f"{lib.name} missing")
            return
        mon = next((t for t in trials), None)
        mon_raw = mon[2] if mon else None
        mon_truth = (find_manual_csv(mon_raw) or mon_raw) if mon_raw else None
        ok, _ = step(R, p, "", "train", [(part(model), model)],
                     [PY, "scripts/soma/train.py", "--library", lib, "--mano-dir", mano,
                      "--steps", a.steps, "--out", part(model), "--monitor-raw", mon_raw,
                      "--monitor-truth", mon_truth,
                      "--monitor-name", "MANUAL" if mon_truth != mon_raw else "VICON"],
                     pout / "train.log")
        # train.py writes <out>.log.json next to the checkpoint
        tl = part(model).with_suffix(".log.json")
        if ok and tl.exists():
            os.replace(tl, model.with_suffix(".log.json"))
        if not ok:
            return

    # ---- every trial: label -> fit -> flags -> angles
    for session, sg, raw in trials:
        tdir = pout / sg
        lab_csv, lab_npz = tdir / "soma_labelled.csv", tdir / "labels.npz"
        fit_npz, pred_npz = tdir / "fit.npz", tdir / "pred.npz"
        flags_csv, angles_csv = tdir / "flags.csv", tdir / "angles.csv"
        if "label" in stages:
            ok, _ = step(R, p, sg, "label", [(part(lab_csv), lab_csv), (part(lab_npz), lab_npz)],
                         [PY, "scripts/soma/eval_trial.py", "--model", model, "--raw", raw, "--truth", "",
                          "--csv-out", part(lab_csv), "--out", part(lab_npz)], tdir / "label.log")
            if not ok:
                continue
        if "fit" in stages:
            extra = ["--mv-frames", a.max_frames] if a.max_frames else []
            ok, _ = step(R, p, sg, "fit", [(part(fit_npz), fit_npz), (part(pred_npz), pred_npz)],
                         [PY, "scripts/mano/fit_surface_markers.py", "--mano-dir", mano,
                          "--static", static_use, "--movement", lab_csv,
                          "--out", tdir / "static_fit.html"] + fit_args_from_config(a.fit_config)
                         + ["--cuda-graph", "--mv-chunk", a.chunk, "--mv-overlap", a.overlap,
                            "--npz-out", part(fit_npz), "--pred-out", part(pred_npz)] + extra,
                         tdir / "fit.log")
            if not ok:
                continue
        if "flags" in stages:
            ok, _ = step(R, p, sg, "flags", [(part(flags_csv), flags_csv)],
                         [PY, "scripts/soma/flag_frames.py", "--labels", lab_npz, "--csv", lab_csv,
                          "--fit", fit_npz, "--pred", pred_npz, "--out", part(flags_csv)],
                         tdir / "flags.log")
            if ok and (a.force or not angles_csv.exists()):
                write_angles(fit_npz, flags_csv, angles_csv)


# ------------------------------------------------------------------------- main

def plan(args, sessions_all):
    ps = sorted(sessions_all, key=participant_sort_key)
    if args.participants != "all":
        want = [x.strip() for x in args.participants.split(",") if x.strip()]
        unknown = [w for w in want if w not in sessions_all]
        if unknown:
            raise SystemExit(f"no trial list for {unknown}: only participants in manual_frames.csv "
                             f"are known ({', '.join(ps)}); Px/Pxh need their own list first")
        ps = [p for p in ps if p in want]
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    ps = [p for p in ps if p not in skip]
    i, n = (int(x) for x in args.shard.split("/"))
    return [p for k, p in enumerate(ps) if k % n == i]


def dry_run(args, ps, sessions_all):
    print(f"{len(ps)} participants (skipping {args.skip or 'none'}); out {args.out_root}\n")
    total = 0
    for p in ps:
        pdir = DATA_ROOT / p
        sess = sessions_all[p]
        st, how = resolve_static(p, sess, pdir, args.static_map)
        tr = [t for t in resolve_trials(p, sess, pdir) if not args.trials or t[0] in args.trials or t[1] in args.trials]
        found = [t for t in tr if t[2] is not None]
        total += len(found)
        lib = REPO / "results/soma/libraries" / f"library_{p}.npz"
        mdl = REPO / "results/soma/models" / f"labeler_{p}.pt"
        base = "base" if p == BASE else ""
        have = ("lib " + ("Y" if lib.exists() or base else "n") + ", model " + ("Y" if mdl.exists() or base else "n"))
        fin = sum((Path(args.out_root) / p / t[1] / "angles.csv").exists() for t in found)
        print(f"{p:4s} static: {(st.name if st else 'NONE'):26s} [{how}]  trials {len(found)}/{len(tr)}  {have}  "
              f"finished {fin}/{len(found)}")
        for s, sg, f in tr:
            if f is None:
                print(f"       MISSING FILE for {s!r} ({sess[s]})")
    print(f"\n{total} trials. Nothing was run.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", default="all", help="comma list, or all (those in manual_frames.csv)")
    ap.add_argument("--skip", default="P1,P2", help="participants to leave out (P1/P2: other marker set)")
    ap.add_argument("--trials", default="", help="comma list of Session names or slugs to restrict to")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th participant")
    ap.add_argument("--out-root", default=str(REPO / "results/soma/fullset"))
    ap.add_argument("--mano-dir", default=str(DATA_ROOT / "mano_v1_2" / "models"
                                              if (DATA_ROOT / "mano_v1_2" / "models").exists()
                                              else REPO / "../clean_kinematics/mano_v1_2/models"))
    ap.add_argument("--base-library", default=str(REPO / "results/soma/libraries/library_P10_Trial2.npz"))
    ap.add_argument("--base-model", default=str(REPO / "results/soma/models/labeler_P10.pt"))
    ap.add_argument("--fit-config", default=str(REPO / "configs/surface_fit/P10.json"),
                    help="surface-fit settings; P10's are the validated ones")
    ap.add_argument("--steps", type=int, default=10000, help="labeller training steps")
    ap.add_argument("--chunk", type=int, default=3000, help="fit chunk size (frames)")
    ap.add_argument("--overlap", type=int, default=150, help="fit chunk overlap (frames)")
    ap.add_argument("--max-frames", type=int, default=0, help="fit only the first N frames (smoke test)")
    ap.add_argument("--static-map", default="", help="P5=path,P6=path ... explicit statics for participants without one")
    ap.add_argument("--max-beta", type=float, default=3.0, help="flag a static whose fitted |beta| exceeds this")
    ap.add_argument("--max-absent", type=int, default=2, help="flag a static missing more hand markers than this")
    ap.add_argument("--max-static-fit-mm", type=float, default=8.0,
                    help="flag a static whose stage-2 surface fit is worse than this "
                         "(known-good statics: P10 5.25 mm, P12 4.65 mm)")
    ap.add_argument("--force", action="store_true", help="redo stages even if their outputs exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.stages = [s for s in args.stages.split(",") if s]
    args.trials = {t.strip() for t in args.trials.split(",") if t.strip()}
    args.static_map = dict(kv.split("=", 1) for kv in args.static_map.split(",") if "=" in kv)
    args.fit_config = Path(args.fit_config)

    sessions_all = load_trial_sessions(REF_CSV)
    ps = plan(args, sessions_all)
    if args.dry_run:
        return dry_run(args, ps, sessions_all)

    R = Run(args)
    t0 = time.time()
    for p in ps:
        try:
            do_participant(R, p, sessions_all[p])
        except Exception as e:                      # one participant must not stop the rest
            R.failed += 1
            R.note(p, "", "ERROR", type(e).__name__, detail=str(e)[:300])
    import csv
    with open(R.out / f"flagged{'' if args.shard == '0/1' else '_' + args.shard.replace('/', 'of')}.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "reason", "detail"])
        w.writeheader(); w.writerows(R.flagged)
    print(f"\n{len(ps)} participants in {(time.time()-t0)/60:.1f} min: "
          f"{len(R.flagged)} flagged (for you to fix), {R.failed} failed step(s). Status: {R.status}")
    for fl in R.flagged:
        print(f"  FLAG {fl['participant']}: {fl['reason']} - {fl['detail']}")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
