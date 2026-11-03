#!/usr/bin/env python3
"""Add a trial's HD-sEMG to its RealSense player: per grid, a 16x6 electrode map and a channel x time heatmap.

Per channel: RMS of the CAR 20-450 Hz signal in 50 ms windows (core.emg.rms_envelope),
stored as log10 uV quantised to uint8 in emg.json beside the player. EMG is placed
on the Vicon clock by matching the sync pattern AUX 1 and Vicon ``Sync`` both record
(core.video_sync.mask_offset). Where the Vicon export has no Sync channel, AUX 1's
1 s pulse is matched to the RealSense trigger log instead. --emg-offset overrides both.
Channels flagged by scripts/emg/emg_quality.py are marked, if its CSV exists.

    python scripts/emg/emg_pane.py results/video/P16/Trial2_HOI
    -> results/video/P16/Trial2_HOI/{emg.json, player.html}
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "shared")]

from vicon2mano.core.emg import GRID_COLS, GRID_ROWS, read_otb4, rms_envelope  # noqa: E402
from vicon2mano.core.video_sync import find_trigger_pulse, mask_offset, read_sync_channel, sync_mask  # noqa: E402
from realsense_player import rebuild, rs_dir_of  # noqa: E402

LOG_MAX = 3.0  # log10 uV mapped to 255 (1000 uV); 0 = 1 uV, same scale as the quality heat map


def emg_sync(aux: np.ndarray, fs: float, sync: dict) -> dict | None:
    """EMG-minus-Vicon offset: AUX 1 vs Vicon Sync pattern, else AUX 1's 1 s pulse vs the RealSense trigger log."""
    a = sync_mask(aux, fs, 8000)  # AUX 1 is raw counts, +-32768 full scale
    ch = read_sync_channel(Path(sync["vicon_csv"]))
    v = sync_mask(*ch, 0.1) if ch else None
    if v is not None and a.any() and v.any():
        lag, iou, amb = mask_offset(a, v)
        return {"offset_s": round(lag, 4), "method": "AUX 1 vs Vicon Sync pattern", "iou": round(iou, 3), "ambiguity": round(amb, 3)}
    pulse = find_trigger_pulse(a.astype(float), 1000.0)
    if pulse is not None and sync.get("trigger_ms"):
        trig = (sync["trigger_ms"] - sync["vicon_start_ms"]) / 1000 + sync["offset_s"]
        return {"offset_s": round(pulse - trig, 4), "method": f"AUX 1 1 s pulse vs RealSense trigger log (video sync: {sync['method']})"}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path, help="results/video/<P>/<Trial>_<cond>, already synced")
    ap.add_argument("--emg-offset", type=float, help="EMG-minus-Vicon time in s (default: from the shared sync events)")
    ap.add_argument("--win-s", type=float, default=0.05)
    a = ap.parse_args()
    vdir = a.video_dir.resolve()
    sync = json.loads((vdir / "sync.json").read_text())
    otb4 = next((rs_dir_of(vdir).parent / "Novecento").glob("*.otb4"))

    rec = read_otb4(otb4)
    if a.emg_offset is not None:
        es = {"offset_s": a.emg_offset, "method": "manual"}
    else:
        es = emg_sync(rec.aux["AUX 1"].astype(float), rec.fs, sync)
    if es is None:
        sys.exit(f"no sync event shared by AUX 1 and Vicon / the RealSense trigger for {otb4} - pass --emg-offset")

    flags_csv = REPO / "results" / "emg" / "quality" / f"{vdir.parent.name}_{vdir.name}.csv"
    flagged = {}
    if flags_csv.exists():
        with open(flags_csv) as f:
            for r in csv.DictReader(f):
                if r["flags"]:
                    flagged.setdefault(r["grid"], []).append(int(r["channel"]))

    grids, rows = [], []
    for gi, (name, raw) in enumerate(rec.grids.items(), 1):
        rms = rms_envelope(raw, rec.fs, rec.mv_per_count, a.win_s)
        q = np.nan_to_num(np.clip(np.log10(np.maximum(rms, 1)) / LOG_MAX * 255, 0, 255)).astype(np.uint8)
        rows.append(q)
        grids.append({"name": name, "flagged": flagged.get(str(gi), [])})
        print(f"grid {gi} ({name}): {rms.shape[1]} windows, {len(grids[-1]['flagged'])} flagged", flush=True)
    data = np.concatenate(rows)  # (sum of channels, W), row-major
    emg = {"file": otb4.name, **es, "win_s": a.win_s, "log_max": LOG_MAX,
           "channels": int(data.shape[0] // len(grids)), "rows": GRID_ROWS, "cols": GRID_COLS, "windows": int(data.shape[1]), "grids": grids,
           "data": base64.b64encode(data.tobytes()).decode("ascii")}
    (vdir / "emg.json").write_text(json.dumps(emg))
    print(f"EMG {otb4.name}: emg_t = vicon_t + {es['offset_s']} s ({es['method']}, IoU {es.get('iou')}, ambiguity {es.get('ambiguity')})")

    rebuild(vdir)


if __name__ == "__main__":
    main()
