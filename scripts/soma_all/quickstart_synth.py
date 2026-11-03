#!/usr/bin/env python3
"""Filling-and-fitting smoke test on SYNTHETIC data (a made-up participant generated from MANO alone).

NOT A RECORDING and not a labelling test: it checks that the installation, file formats and the gap-filling and
movement-fitting commands work end to end and recover a known answer within frozen tolerances; it says nothing about
accuracy on real Vicon recordings. The test is frozen in a versioned spec (configs/quickstart_synth_v1.json: generator
settings, gaps, tolerances, the pin file's sha256, an input fingerprint and the reference results). No recorded data
is read: the hand is MANO's mean shape (scale 1) with markers on the released pins, 3 mm out along the normal; the
movement is smooth random finger poses inside the joint limits with a slow wrist motion, noise, and fixed gaps. MANO
itself is not included: it is read from core.dataset.MANO_DIR (obtain it under its licence).

    python scripts/soma_all/quickstart_synth.py --out results/soma/quickstart
Writes static.csv, static_params.npz, movement.csv (with gaps), truth.npz, then gap_fill.py -> filled.csv (rigid), a
filler trained on movement.csv -> filled_learned.csv, movement_fit.py on the learned fill -> fit/, and check.json.
Exits 1 if the inputs differ from the spec's fingerprint or a learned-fill or fit result is outside the tolerances.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/soma_all"), str(REPO / "scripts/mano")]
import fit_surface_markers as fsm  # noqa: E402
import static_search as ss  # noqa: E402
from vicon2mano.strategies.soma.realdata import write_wide_csv  # noqa: E402



def markers(J, att: torch.Tensor, go, hp, tr, standoff=3.0) -> np.ndarray:
    """Marker positions (T, M, 3) mm: attached vertices of the mean-shape MANO, ``standoff`` mm along the normal."""
    sm = fsm.ScaledModel(J.model, J.dev)
    t = lambda x: torch.tensor(np.asarray(x, np.float32), device=J.dev)
    with torch.no_grad():
        V = sm(global_orient=t(go), hand_pose=t(hp), transl=t(tr), betas=torch.zeros(len(go), 10, device=J.dev),
               return_verts=True).vertices
        F = torch.tensor(np.asarray(J.model.faces, np.int64), device=J.dev)
        fn = torch.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]], dim=-1)
        N = torch.zeros_like(V).index_add_(1, F[:, 0], fn).index_add_(1, F[:, 1], fn).index_add_(1, F[:, 2], fn)
        N = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return ((V[:, att] + standoff / 1000 * N[:, att]) * 1000).cpu().numpy()


def poses(J, T: int, rng, every: int = 250, frac: float = 0.4) -> np.ndarray:
    """Smooth random hand poses (T, 45): keyframes inside ``frac`` of the joint limits, no twist or PIP/DIP abduction."""
    lo, hi = J.lo.copy(), J.hi.copy()                                    # (15, 3) deg in the joint-axis basis
    lo[:12, 2] = hi[:12, 2] = 0
    hinge = [j for j, n in enumerate(fsm._FINGER_JOINTS) if not n.startswith("thumb") and n[-1] in "23"]
    lo[hinge, 1] = hi[hinge, 1] = 0
    keys = rng.uniform(np.minimum(lo, 0), np.maximum(hi, 0), (T // every + 2, 15, 3)) * frac
    keys[0] = 0
    s = np.arange(T) / every
    i, w = s.astype(int), (1 - np.cos(np.pi * (s % 1))) / 2
    c = np.radians(keys[i] * (1 - w)[:, None, None] + keys[i + 1] * w[:, None, None])
    return np.einsum("tjc,jck->tjk", c, np.linalg.inv(J.B)).reshape(T, 45)


def sha(path: Path) -> str:
    """sha256 of a file, line endings normalised (so a checkout's CRLF does not change it)."""
    import hashlib
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint(static: np.ndarray, truth: np.ndarray) -> dict:
    """A few statistics of the generated markers (mm), enough to notice a changed generator or MANO model."""
    return dict(static_mean=round(float(static.mean()), 2), truth_mean=round(float(truth.mean()), 2),
                truth_std=round(float(truth.std()), 2))


def run(args: list, log: Path):
    """Run a repo script with this interpreter, output to ``log``; raise if it fails."""
    subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], check=True, cwd=REPO,
                   stdout=open(log, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/soma/quickstart"))
    ap.add_argument("--spec", default=str(REPO / "configs/quickstart_synth_v1.json"))
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out, rng, T, tol = Path(a.out), np.random.default_rng(spec["seed"]), spec["frames"], spec["tolerances"]
    GAPS = {m: tuple(v) for m, v in spec["gaps"].items()}
    out.mkdir(parents=True, exist_ok=True)
    pin_file = REPO / spec["pins"]
    if sha(pin_file) != spec["pins_sha256"]:
        sys.exit(f"{pin_file} differs from the frozen spec's (sha256 mismatch)")

    J = ss.Judge()
    pins = json.loads(pin_file.read_text())
    mk = list(pins)
    att = torch.tensor([pins[m] for m in mk], device=J.dev)
    z3 = np.zeros((1, 3), np.float32)
    S = markers(J, att, z3, np.zeros((1, 45)), z3)
    write_wide_csv(out / "static.csv", S + rng.normal(0, spec["noise_mm"]["static"], (300, *S.shape[1:])), mk)
    np.savez(out / "static_params.npz", hand_pose=np.zeros((1, 45), np.float32), global_orient=z3, transl=z3,
             betas=np.zeros(10, np.float32), scale=1.0, names=np.array(mk), att=att.cpu().numpy())

    tt = np.arange(T) / 200
    go = np.stack([0.3 * np.sin(2 * np.pi * f * tt + p) for f, p in ((0.13, 0), (0.07, 1), (0.11, 2))], 1)
    tr = np.stack([0.05 * np.sin(2 * np.pi * f * tt + p) for f, p in ((0.05, 0), (0.08, 2), (0.06, 4))], 1)
    hp = poses(J, T, rng)
    X = markers(J, att, go, hp, tr)
    Xg = X + rng.normal(0, spec["noise_mm"]["movement"], X.shape)
    for m, (s, e) in GAPS.items():
        Xg[s:e, mk.index(m)] = np.nan
    write_wide_csv(out / "movement.csv", Xg, mk)
    np.savez(out / "truth.npz", markers=X, hand_pose=hp, global_orient=go, transl=tr, names=np.array(mk))
    print(f"synthetic participant written to {out} ({T} frames, gaps in {', '.join(GAPS)})", flush=True)

    from vicon2mano.core.loader import load_csv

    def errors(csv: Path) -> dict:
        Fm, fn = load_csv(str(csv))[:2]
        Fm = Fm[:, [[n.split(":")[-1] for n in fn].index(m) for m in mk]]
        e = {m: np.linalg.norm(Fm[s:e, mk.index(m)] - X[s:e, mk.index(m)], axis=-1) for m, (s, e) in GAPS.items()}
        return {m: dict(median=round(float(np.nanmedian(v)), 2), max=round(float(np.nanmax(v)), 2),
                        all_filled=bool(np.isfinite(v).all())) for m, v in e.items()}

    run(["scripts/soma_all/gap_fill.py", "--csv", out / "movement.csv", "--out", out / "filled.csv"], out / "gap_fill.log")
    run(["scripts/soma_all/train_filler.py", "--csv", out / "movement.csv", "--steps", spec["filler_steps"], "--seed", spec["seed"],
         "--out", out / "filler.pt"], out / "train_filler.log")
    run(["scripts/soma_all/gap_fill.py", "--csv", out / "movement.csv", "--out", out / "filled_learned.csv",
         "--filler", out / "filler.pt", "--max-gap", 100000], out / "gap_fill_learned.log")
    rigid, learned = errors(out / "filled.csv"), errors(out / "filled_learned.csv")
    run(["scripts/soma_all/movement_fit.py", "--participant", "SYNTH", "--static", out / "static_params.npz",
         "--static-csv", out / "static.csv", "--csv", out / "filled_learned.csv", "--out", out / "fit", "--step", 10,
         "--chunk", 1000, "--iters", 800, "--w-filled", 0.1], out / "movement_fit.log")
    f = np.load(out / "fit" / "fit.npz")
    fr = f["frames_fitted"]
    rot = lambda h: Rotation.from_rotvec(np.asarray(h).reshape(-1, 3))
    ang = np.degrees((rot(f["hand_pose"]).inv() * rot(hp[fr])).magnitude()).reshape(len(fr), 15)
    fp = fingerprint(S, X)
    res = dict(test="filling-and-fitting smoke test, synthetic data", spec=spec["version"], frames=T,
               fitted_frames=int(len(fr)), input_fingerprint=fp, input_fingerprint_matches=fp == spec["fingerprint"],
               gap_fill_mm=dict(rigid=rigid, learned=learned),
               local_joint_rotation_error_deg=dict(median=round(float(np.median(ang)), 2),
                                                   p95=round(float(np.percentile(ang, 95)), 2)),
               fit_max_marker_distance_mm=json.loads((out / "fit" / "summary.json").read_text()).get("max_dist_mm"),
               tolerances=tol)
    res["passed"] = bool(res["input_fingerprint_matches"]
                         and all(v["all_filled"] and v["median"] <= tol["learned_fill_median_mm"] for v in learned.values())
                         and res["local_joint_rotation_error_deg"]["median"] <= tol["local_joint_rotation_median_deg"])
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
