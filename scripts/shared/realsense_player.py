"""Export RealSense .bag colour streams to MP4, sync them to Vicon, build an HTML player.

Each bag's colour frames are written as a variable-frame-rate MP4 whose
presentation times are the camera's own global timestamps (epoch ms), plus a
per-frame CSV. The recorder drops frames (sensor runs at 60 fps, the bags
keep ~7-20 fps, unevenly), so a constant fps must never be assumed:
sync on the timestamps.

The Vicon offset comes from the trigger pulse when Vicon caught it, else from
motion cross-correlation (see vicon2mano.core.video_sync). An offset in
configs/sync_overrides.csv (stream "video") overrides both; --offset is for a quick try.

    python scripts/shared/realsense_player.py D:/june25_data/P15/Trial1/Hands_only/RealSense
    -> results/video/P15/Trial1_Hands_only/
       player.html, cam*.mp4, sync.json,
       cam*_frames.csv  (frame_number, timestamp_ms, pts_ms, motion, vicon_t_s, vicon_frame)

MP4s already on disk are reused (--force re-encodes). Requires: pyrealsense2, av.
"""
import argparse
import base64
import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyrealsense2 as rs

from vicon2mano.core.video_sync import (XCORR_BIAS_S, find_trigger_pulse, find_vicon_csv, finger_speed, hand_speed,
                                        load_trajectories, motion_lag, read_sync_channel, sync_overrides,
                                        trigger_log_ms, vicon_start_ms)
from vicon2mano.core.viz import marker_bone_pairs, marker_color

ROOT = Path(__file__).resolve().parents[2]


def read_color_frames(bag: Path):
    """Yield (frame_number, global_timestamp_ms, rgb array) once per recorded colour frame."""
    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, str(bag), repeat_playback=False)
    cfg.enable_stream(rs.stream.color)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    last = None
    try:
        while True:
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                return
            c = fs.get_color_frame()
            if c and c.get_frame_number() != last:  # playback repeats framesets
                last = c.get_frame_number()
                yield last, c.get_timestamp(), np.asanyarray(c.get_data()).copy()
    finally:
        pipe.stop()


def export_bag(bag: Path, mp4: Path, crf: int = 23) -> list[tuple]:
    """Write one bag's colour stream to a VFR MP4; return (frame_number, ts_ms, pts_ms, motion) rows.

    motion = mean absolute grey-level change since the previous recorded frame, per second.
    """
    rows, prev = [], None
    with av.open(str(mp4), "w") as out:
        st = None
        for n, ts, img in read_color_frames(bag):
            if st is None:
                st = out.add_stream("libx264", rate=60, options={"crf": str(crf), "g": "30"})
                st.width, st.height, st.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                st.time_base = st.codec_context.time_base = Fraction(1, 1000)
            pts = round(ts - rows[0][1]) if rows else 0
            if rows and pts <= rows[-1][2]:  # out-of-order / duplicate timestamp
                continue
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = pts
            out.mux(st.encode(frame))
            h, w = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8  # 8x8 block means: averages out sensor noise
            grey = img[:h, :w].reshape(h // 8, 8, w // 8, 8, 3).mean(axis=(1, 3, 4))
            motion = 0.0 if prev is None else float(np.abs(grey - prev).mean() * 1000 / (pts - rows[-1][2]))
            rows.append((n, ts, pts, motion))
            prev = grey
        out.mux(st.encode())
    return rows


def cam_summary(name: str, rows) -> dict:
    """Frame-rate stats plus the arrays the player needs."""
    a = np.array(rows, float)
    dt = np.diff(a[:, 1])
    dur = (a[-1, 1] - a[0, 1]) / 1e3
    return {"name": name, "file": f"{name}.mp4", "t0": float(a[0, 1]), "ts": a[:, 2].astype(int).tolist(),
            "fn": a[:, 0].astype(int).tolist(), "motion": np.round(a[:, 3], 1).tolist(), "n": len(a),
            "duration_s": round(float(dur), 2), "fps_recorded": round(float((len(a) - 1) / dur), 2),
            "fps_sensor": round(float((a[-1, 0] - a[0, 0]) / dur), 2),
            "dt_ms_median": round(float(np.median(dt)), 1), "dt_ms_max": round(float(dt.max()), 1)}


def sync_to_vicon(vicon_csv: Path, rs_dir: Path, cams: list[dict], offset: float | None,
                  X: np.ndarray, labels: list[str], rate: float) -> tuple[dict, np.ndarray]:
    """Measure the RealSense->Vicon offset; return (sync record, hand-speed trace at 50 Hz)."""
    vs = vicon_start_ms(vicon_csv)
    sync_ch = read_sync_channel(vicon_csv)
    onset = find_trigger_pulse(*sync_ch) if sync_ch else None
    trig = trigger_log_ms(rs_dir)
    t, speed = hand_speed(X, labels, rate)
    xcorr = {}
    for c in cams:
        ts = np.array(c["ts"], float)
        mid = (c["t0"] - vs + (ts[1:] + ts[:-1]) / 2) / 1000  # between the two frames a motion value compares
        xcorr[c["name"]] = motion_lag(mid, np.array(c["motion"][1:]), t, speed)
    if offset is not None:
        method = "manual"
    elif onset is not None and trig is not None:
        offset, method = onset - (trig - vs) / 1000, "trigger pulse"
    else:
        best = max(xcorr, key=lambda k: xcorr[k][1])
        offset, method = xcorr[best][0] + XCORR_BIAS_S, f"motion xcorr ({best}) + bias"
    sync = {"vicon_csv": str(vicon_csv), "vicon_start_ms": vs, "vicon_rate": rate, "offset_s": round(offset, 4),
            "method": method, "pulse_onset_s": onset, "trigger_ms": trig,
            "xcorr": {k: {"lag_s": round(lag, 3), "r": round(r, 3)} for k, (lag, r) in xcorr.items()}}
    return sync, to_50hz(speed, rate)


def to_50hz(x: np.ndarray, rate: float) -> np.ndarray:
    """Block-mean a Vicon-rate trace down to 50 Hz for the player."""
    step = int(round(rate / 50))
    return np.round(x[: len(x) // step * step].reshape(-1, step).mean(axis=1), 1)


def markers_pane(X: np.ndarray, labels: list[str], rate: float, hz: float = 100.0) -> dict:
    """Raw Vicon markers for the player's live 3-D pane: int16 mm at ~hz, -32768 = missing."""
    step = max(1, int(round(rate / hz)))
    Y = X[::step]
    q = np.where(np.isfinite(Y), np.clip(np.round(np.nan_to_num(Y)), -32767, 32767), -32768).astype(np.int16)
    return {"rate": rate / step, "n": len(Y), "labels": [n.split(":")[-1] for n in labels],
            "colors": [marker_color(n) for n in labels], "bones": marker_bone_pairs(labels),
            "data": base64.b64encode(q.tobytes()).decode("ascii")}


def rs_dir_of(video_dir: Path) -> Path:
    """The RealSense folder an output folder results/video/<P>/<Trial>_<cond> was built from."""
    sync = json.loads((video_dir / "sync.json").read_text())
    p, (trial, cond) = video_dir.parent.name, video_dir.name.split("_", 1)
    return Path(sync["vicon_csv"]).parents[2] / p / trial / cond / "RealSense"


def rebuild(video_dir: Path):
    """Rebuild an existing player (reuses its MP4s and Vicon CSV; manual offsets come from configs/sync_overrides.csv)."""
    sync = json.loads((video_dir / "sync.json").read_text())
    build(rs_dir_of(video_dir), video_dir, Path(sync["vicon_csv"]))


def build(rs_dir: Path, out: Path | None = None, vicon: Path | None = None, offset: float | None = None,
          crf: int = 23, force: bool = False):
    """Export (or reuse) the cameras, sync to Vicon, and write player.html with any mesh/EMG panes."""
    rs_dir = rs_dir.resolve()
    p, trial, cond = rs_dir.parts[-4:-1]
    out = out or ROOT / "results" / "video" / p / f"{trial}_{cond}"
    out.mkdir(parents=True, exist_ok=True)
    override = sync_overrides().get((f"{p}/{trial}_{cond}", "video"), {})
    if offset is None and override.get("offset_s"):
        offset = float(override["offset_s"])

    cams = []
    for i, bag in enumerate(sorted(rs_dir.glob("*.bag")), 1):
        name, frames_csv = f"cam{i}", out / f"cam{i}_frames.csv"
        if not force and frames_csv.exists() and (out / f"{name}.mp4").exists():
            with open(frames_csv) as f:
                rows = [(int(r["frame_number"]), float(r["timestamp_ms"]), int(r["pts_ms"]), float(r["motion"]))
                        for r in csv.DictReader(f)]
        else:
            print(f"{bag.name} -> {name}.mp4 ...", flush=True)
            rows = export_bag(bag, out / f"{name}.mp4", crf)
        cams.append(cam_summary(name, rows))
        print({k: v for k, v in cams[-1].items() if k not in ("ts", "fn", "motion")})

    vicon_csv = vicon or find_vicon_csv(rs_dir)
    sync, speed, markers, fingers = None, None, None, None
    if vicon_csv and vicon_csv.with_suffix(".xcp").exists():
        X, labels, rate = load_trajectories(vicon_csv)
        sync, speed = sync_to_vicon(vicon_csv, rs_dir, cams, offset, X, labels, rate)
        sync.update({k: override[k] for k in ("checked_by", "date", "note") if override.get(k)})
        markers = markers_pane(X, labels, rate)
        fingers = to_50hz(finger_speed(X, labels, rate), rate)
        (out / "sync.json").write_text(json.dumps(sync, indent=1))
        print("sync:", {k: v for k, v in sync.items() if k != "vicon_csv"})
    else:
        print(f"[warn] no Vicon CSV/.xcp for {rs_dir} - pass --vicon; player shows camera time only")

    for c in cams:
        with open(out / f"{c['name']}_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_number", "timestamp_ms", "pts_ms", "motion"] + (["vicon_t_s", "vicon_frame"] if sync else []))
            for n, pts, m in zip(c["fn"], c["ts"], c["motion"]):
                ms = c["t0"] + pts
                row = [n, f"{ms:.3f}", pts, m]
                if sync:  # vicon_frame = the Vicon CSV's 1-based Frame column
                    vt = (ms - sync["vicon_start_ms"]) / 1000 + sync["offset_s"]
                    row += [f"{vt:.4f}", int(np.floor(vt * sync["vicon_rate"])) + 1]
                w.writerow(row)

    panes = list(cams)
    for f in sorted(out.glob("mesh*_frames.csv")) if sync else []:  # Vicon-clock videos, e.g. scripts/soma/mesh_video.py
        with open(f) as fh:
            rows = [(int(r["vicon_frame"]), sync["vicon_start_ms"] + 1000 * ((int(r["vicon_frame"]) - 1) / sync["vicon_rate"]
                     - sync["offset_s"]), int(r["pts_ms"]), 0.0) for r in csv.DictReader(fh)]
        panes.append(cam_summary(f.name[: -len("_frames.csv")], rows))

    emg = json.loads((out / "emg.json").read_text()) if sync and (out / "emg.json").exists() else None  # scripts/emg/emg_pane.py
    meta = {"title": f"{p} {trial} {cond}", "trigger_ms": trigger_log_ms(rs_dir), "cams": panes, "sync": sync, "emg": emg, "markers": markers,
            "speed50": speed.tolist() if speed is not None else None,
            "finger50": fingers.tolist() if fingers is not None else None}
    html = (Path(__file__).with_name("realsense_player.html.in").read_text(encoding="utf-8")
            .replace("/*META*/null", json.dumps(meta)))
    (out / "player.html").write_text(html, encoding="utf-8")
    print("->", out / "player.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rs_dir", type=Path, help="folder containing rsframes*.bag")
    ap.add_argument("--out", type=Path, help="default: results/video/<P>/<Trial>_<cond>")
    ap.add_argument("--vicon", type=Path, help="Vicon trial CSV (default: looked up via manual_frames.csv)")
    ap.add_argument("--offset", type=float, help="RealSense->Vicon offset in s for a quick try; "
                                                 "record kept ones in configs/sync_overrides.csv")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--force", action="store_true", help="re-encode MP4s that already exist")
    a = ap.parse_args()
    build(a.rs_dir, a.out, a.vicon, a.offset, a.crf, a.force)


if __name__ == "__main__":
    main()
