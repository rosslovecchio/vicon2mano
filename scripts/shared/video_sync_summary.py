"""Collect every results/video/*/*/sync.json into results/video/sync_summary.csv.

xcorr_err_ms (pulse trials only) is each camera's raw xcorr lag minus the pulse offset.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for d in sorted((ROOT / "results" / "video").glob("*/*/")):
        row = {"trial": f"{d.parent.name}/{d.name}"}
        f = d / "sync.json"
        if not f.exists():
            rows.append({**row, "method": "NO SYNC"})
            continue
        s = json.loads(f.read_text())
        lags = [v["lag_s"] for v in s["xcorr"].values()]
        row.update(vicon=Path(s["vicon_csv"]).name, method=s["method"], offset_s=s["offset_s"])
        for k, v in s["xcorr"].items():
            row.update({f"{k}_lag": v["lag_s"], f"{k}_r": v["r"]})
        row["cam_spread_ms"] = round(1000 * (max(lags) - min(lags)))
        if s["method"] == "trigger pulse":
            row["xcorr_err_ms"] = "/".join(str(round(1000 * (lag - s["offset_s"]))) for lag in lags)
        rows.append(row)
    out = ROOT / "results" / "video" / "sync_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trials -> {out}")


if __name__ == "__main__":
    main()
