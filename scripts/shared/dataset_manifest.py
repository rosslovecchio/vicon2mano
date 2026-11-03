"""Per-trial status of every data type of the final dataset -> results/dataset/manifest.csv (docs/dataset.md).

Status per stream: usable / check (automatic but needs a look) / draft (exists, not final) /
todo (not done yet) / excluded (cannot be done), plus the reason.

    python scripts/shared/dataset_manifest.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

from vicon2mano.core.dataset import soma_fit_dir

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "results" / "video"
BAD_GRID = 48  # >= half the 96 channels flagged: the grid is unusable (docs/emg.md)


def video_status(sync: dict | None) -> tuple[str, str]:
    if sync is None:
        return "excluded", "no Vicon recording"
    if sync["method"] in ("trigger pulse", "manual") or sync.get("checked_by"):
        return "usable", sync["method"] + (f", checked by {sync['checked_by']}" if sync.get("checked_by") else "")
    lags = [v["lag_s"] for v in sync["xcorr"].values()]
    spread = 1000 * (max(lags) - min(lags))
    return ("check", f"cameras disagree by {spread:.0f} ms") if spread > 50 else ("usable", f"{sync['method']}, ~±30 ms")


def emg_sync_status(emg: dict | None, has_otb4: bool, has_vicon: bool) -> tuple[str, str]:
    if not has_vicon:
        return "excluded", "no Vicon recording to sync to"
    if not has_otb4:
        return "excluded", "no .otb4"
    if emg is None:
        return "todo", "no shared sync event: set the offset by eye"
    if emg["method"] == "manual" or emg.get("checked_by"):
        return "usable", emg["method"]
    if "pattern" in emg["method"] and emg["iou"] >= 0.95 and emg["ambiguity"] <= 0.6:
        return "usable", f"pattern IoU {emg['iou']}"
    return "check", f"{emg['method']} (IoU {emg.get('iou')}, ambiguity {emg.get('ambiguity')})"


def emg_quality_status(rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "todo", "no quality report"
    bad = [r["grid"] for r in rows if int(r["n_bad"]) >= BAD_GRID]
    return ("draft", f"grid(s) {' '.join(bad)} unusable; cleaning not done") if bad else ("draft", "cleaning not done")


def main():
    quality = {}
    qcsv = ROOT / "results" / "emg" / "quality" / "summary.csv"
    if qcsv.exists():
        with open(qcsv) as f:
            for r in csv.DictReader(f):
                quality.setdefault(r["trial"], []).append(r)
    rows = []
    for d in sorted(p for p in VIDEO.glob("*/*") if p.is_dir()):
        p, trial = d.parent.name, d.name
        sync = json.loads((d / "sync.json").read_text()) if (d / "sync.json").exists() else None
        emg = json.loads((d / "emg.json").read_text()) if (d / "emg.json").exists() else None
        trial_dir = Path(sync["vicon_csv"]).parents[2] / p / trial.split("_", 1)[0] / trial.split("_", 1)[1] if sync else None
        has_otb4 = bool(trial_dir and any((trial_dir / "Novecento").glob("*.otb4")))
        fit = soma_fit_dir(p, trial)
        status = {
            "video_sync": video_status(sync),
            "emg_sync": emg_sync_status(emg, has_otb4, sync is not None),
            "emg_clean": emg_quality_status(quality.get(f"{p}_{trial}", [])) if has_otb4 else ("excluded", "no .otb4"),
            "markers": ("draft", "SOMA relabel + gap fill, not reviewed") if fit and (fit / "soma_labelled.csv").exists()
                       else ("todo", "no SOMA labels"),
            "mesh": ("draft", "fit on current labels") if fit and (fit / "fit.npz").exists() else ("todo", "no MANO fit"),
            "depth": ("todo", "not exported"),
            "calibration": ("todo", "camera -> Vicon extrinsics not solved"),
        }
        rows.append({"trial": f"{p}/{trial}", **{k: v[0] for k, v in status.items()},
                     **{f"{k}_why": v[1] for k, v in status.items()}})
    out = ROOT / "results" / "dataset" / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    streams = [k for k in rows[0] if not k.endswith("_why") and k != "trial"]
    for k in streams:
        print(f"{k:12s}", dict(Counter(r[k] for r in rows)))
    print(f"{len(rows)} trials -> {out}")


if __name__ == "__main__":
    main()
