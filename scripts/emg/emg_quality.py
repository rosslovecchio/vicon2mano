#!/usr/bin/env python3
"""Per-channel quality report for the Novecento+ HD-sEMG recordings.

For every ``<EMG_ROOT>/P*/<Trial>/<Condition>/Novecento/*.otb4`` (or the
folders/files given) writes to ``results/emg/quality/``:

* ``<P>_<Trial>_<Cond>.csv``  one row per channel: metrics + flags
* ``<P>_<Trial>_<Cond>.png``  channel x time RMS heat map (drops show as dark/bright streaks)
* ``summary.csv``             one row per trial x grid, plus ramp faults and AUX1 sync activity

Usage
-----
python scripts/emg/emg_quality.py [paths ...] [--workers 3]
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vicon2mano.core.dataset import REPO_ROOT
from vicon2mano.core.emg import EMG_ROOT, channel_quality, flag_channels, ramp_faults, read_otb4

OUT = REPO_ROOT / "results" / "emg" / "quality"
METRICS = ["rail_frac", "flat_s", "offset_mv", "mono_noise_uv", "noise_uv", "active_uv",
           "snr_db", "line_ratio", "lowfreq_ratio", "nbr_corr"]


def aux_bursts(aux: np.ndarray, fs: float, gap_s: float = 0.5) -> list[tuple[float, float]]:
    """(start, end) seconds of each interval where AUX1 (the Vicon sync input) is off baseline.

    Intervals closer than ``gap_s`` merge, so a square-wave burst that crosses
    baseline each cycle reads as one interval, while a 1 s level pulse stays one.
    """
    d = np.diff(np.r_[0, np.abs(aux - np.median(aux)) > 8000, 0].astype(np.int8))
    starts, ends = np.flatnonzero(d == 1) / fs, np.flatnonzero(d == -1) / fs
    if not len(starts):
        return []
    keep = np.r_[True, starts[1:] - ends[:-1] > gap_s]
    return list(zip(starts[keep].round(3), ends[np.r_[keep[1:], True]].round(3)))


def run(otb4: Path) -> list[dict]:
    """Analyse one recording; write its channel CSV and heat map, return its summary rows."""
    tag = "_".join(otb4.relative_to(EMG_ROOT).parts[:3]) if otb4.is_relative_to(EMG_ROOT) else otb4.stem
    rec = read_otb4(otb4)
    bursts = aux_bursts(rec.aux["AUX 1"], rec.fs)
    rows, summary, rms = [], [], []
    for gi, (name, raw) in enumerate(rec.grids.items(), 1):
        q = channel_quality(raw, rec.fs, rec.mv_per_count)
        flags = flag_channels(q)
        lost, rep = ramp_faults(rec.ramps[name])
        rms.append(q["rms"])
        rows += [{"grid": gi, "channel": c + 1, **{m: round(float(q[m][c]), 5) for m in METRICS}, "flags": flags[c]}
                 for c in range(len(flags))]
        bad = [f for f in flags if f]
        summary.append({
            "trial": tag, "file": otb4.name, "duration_s": raw.shape[1] / rec.fs, "grid": gi, "channels": name,
            "n_bad": len(bad), "n_dead": sum("dead" in f for f in bad),
            **{k: sum(k in f for f in bad) for k in ("clipping", "dropout", "noisy", "line", "motion", "isolated")},
            "bad_channels": " ".join(str(c + 1) for c, f in enumerate(flags) if f),
            "median_noise_uv": round(float(np.nanmedian(q["noise_uv"])), 2),
            "median_mono_noise_uv": round(float(np.nanmedian(q["mono_noise_uv"])), 2),
            "median_snr_db": round(float(np.nanmedian(q["snr_db"])), 2),
            "median_line_ratio": round(float(np.nanmedian(q["line_ratio"])), 4),
            "ramp_skips": len(lost), "repeated_samples": len(rep),
            "aux1_bursts": " ".join(f"{a}-{b}" for a, b in bursts),
        })
    with open(OUT / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(len(rms), 1, figsize=(16, 3 * len(rms)), sharex=True)
    for ax, r, s in zip(np.atleast_1d(axes), rms, summary):
        im = ax.imshow(np.log10(np.maximum(r, 0.1)), aspect="auto", cmap="magma", vmin=0, vmax=3,
                       interpolation="nearest", extent=[0, s["duration_s"], r.shape[0] + .5, .5])
        ax.set_ylabel(f"grid {s['grid']}\nchannel")
        ax.set_title(f"{s['channels']}: {s['n_bad']} flagged [{s['bad_channels']}]", fontsize=8, loc="left")
        for a, b in bursts:
            ax.axvspan(a, b, color="cyan", alpha=.35, lw=0)
    axes[-1].set_xlabel("time (s)   [cyan = AUX1 sync bursts, white rows = dead channel]")
    fig.colorbar(im, ax=axes, label="log10 RMS 20-450 Hz CAR (uV)", fraction=.02)
    fig.suptitle(f"{tag} - {otb4.name}")
    fig.savefig(OUT / f"{tag}.png", dpi=70)
    plt.close(fig)
    print(f"{tag}: bad per grid {[s['n_bad'] for s in summary]}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", type=Path, help=".otb4 files or Novecento folders (default: all under EMG_ROOT)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    files = sorted(f for p in a.paths for f in ([p] if p.suffix == ".otb4" else p.glob("*.otb4"))) if a.paths \
        else sorted(EMG_ROOT.glob("P*/*/*/Novecento/*.otb4"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for f, fut in [(f, ex.submit(run, f)) for f in files]:
            try:
                rows += fut.result()
            except Exception as e:  # one unreadable recording should not sink the cohort run
                print(f"FAILED {f}: {e!r}", flush=True)
    if rows:
        with open(OUT / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
