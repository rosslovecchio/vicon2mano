"""Map RealSense frame timestamps onto a Vicon trial's clock.

RealSense frames carry the recording PC's clock (epoch ms). Vicon writes the
capture's ``START_TIME`` into the trial's ``.xcp`` in its own PC's local time,
which is UTC+1 throughout June 2025. Between those two clocks there is an
offset of 0.38-0.94 s that drifts (by up to 0.15 s within a session, and by
0.45 s within one day: P14 -> P15), so it is measured per trial::

    vicon_t [s] = (rs_ms - vicon_start_ms) / 1000 + offset

Two ways to measure ``offset``:

1. **Trigger pulse** - the RealSense recorder sends a 1 s TTL pulse when it
   starts (``triggerlog_*.csv``, first ``trigger_sent=true`` row), which Vicon
   records on its ``Sync`` analog channel. It was captured in 18 of 67
   synced trials. The channel otherwise carries 50 Hz mains bursts, and some
   exports have no Devices section at all.
2. **Motion cross-correlation** - hand-marker speed vs video frame-difference
   energy, per camera; trust the camera with the higher r. Against the pulse
   (18 trials, 36 camera lags): 35 within -49..+13 ms (median -16 ms, hence
   ``XCORR_BIAS_S``), and one locked 3 s off - always the lower-r camera.
   A >50 ms disagreement between the cameras flags a trial to check by eye.

EMG (Novecento+, its own clock from t=0) gets the same sync signal on AUX 1:
``emg_t = vicon_t + aux_pulse_offset(...)`` when both sides show the 1 s pulse
(other pulse shapes: docs/emg.md, sync section).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from vicon2mano.core.dataset import _normalise_name, find_session_file, load_trial_sessions
from vicon2mano.core.loader import load_csv

VICON_UTC_SHIFT_MS = 3600_000  # .xcp local time minus UTC
XCORR_BIAS_S = 0.016  # motion_lag reads early: median of 35 camera lags vs the trigger pulse (docs/video_sync.md)


def _ms(stamp: str) -> int:
    return int(np.datetime64(stamp.replace(" DST", "").replace(" ", "T"), "ms").astype(np.int64))


def vicon_start_ms(vicon_csv: Path) -> int:
    """Epoch ms (UTC) at which the Vicon capture started, from the trial's .xcp."""
    txt = Path(vicon_csv).with_suffix(".xcp").read_text(errors="ignore")
    starts = re.findall(r'END_TIME="[^"]+"[^/]*?START_TIME="([^"]+)"\s*/>', txt)  # last = the capture
    return _ms(starts[-1]) - VICON_UTC_SHIFT_MS


def trigger_log_ms(rs_dir: Path) -> int | None:
    """Epoch ms of the first trigger_sent=true row of rs_dir's triggerlog, or None."""
    for log in Path(rs_dir).glob("triggerlog_*.csv"):
        with open(log) as f:
            return next((_ms(r[0]) for r in csv.reader(f) if len(r) > 2 and r[2] == "true"), None)
    return None


def find_vicon_csv(rs_dir: Path) -> Path | None:
    """Vicon CSV for <root>/<P>/<Trial>/<cond>/RealSense, via <root>/manual_frames.csv."""
    rs_dir = Path(rs_dir).resolve()
    root, p, trial, cond = rs_dir.parents[3], *rs_dir.parts[-4:-1]
    sessions = load_trial_sessions(root / "manual_frames.csv").get(p, {})
    name = next((v for k, v in sessions.items() if _normalise_name(k) == _normalise_name(trial + cond)), None)
    return find_session_file(root / p / p, name) if name else None


def read_sync_channel(vicon_csv: Path) -> tuple[np.ndarray, float] | None:
    """The Devices-section ``Sync`` analog channel and its rate (Hz); None for a trajectories-only export."""
    with open(vicon_csv, encoding="latin-1") as f:
        if not f.readline().strip().endswith("Devices"):  # latin-1 reads a BOM as "ï»¿"
            return None
        rate = float(f.readline().split(",")[0])
        f.readline()
        header = [h.strip() for h in f.readline().split(",")]
        if "Sync" not in header:
            return None
        k = header.index("Sync")
        f.readline()
        sig = []
        for line in f:
            cells = line.split(",")
            if len(cells) <= k or not cells[0].strip():
                break
            sig.append(float(cells[k]) if cells[k].strip() else np.nan)
    return np.array(sig), rate


def find_trigger_pulse(sig: np.ndarray, rate: float, width_s: float = 1.0, tol: float = 0.05) -> float | None:
    """Onset (s) of the first single high plateau lasting width_s +- tol, or None."""
    hi = np.r_[False, sig > 0.5, False]
    edges = np.flatnonzero(np.diff(hi.astype(int)))
    for a, b in zip(edges[::2], edges[1::2]):
        if abs((b - a) / rate - width_s) <= tol * width_s:
            return float(a / rate)
    return None


def load_trajectories(vicon_csv: Path) -> tuple[np.ndarray, list[str], float]:
    """(markers (T, N, 3) mm, labels, Trajectories rate in Hz) of a Vicon trial CSV."""
    with open(vicon_csv, encoding="latin-1") as f:
        rate = next(float(next(f).split(",")[0]) for line in f if line.strip().endswith("Trajectories"))
    X, labels = load_csv(str(vicon_csv))
    return X, labels, rate


def hand_speed(X: np.ndarray, labels: list[str], rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(t [s], median hand-marker speed [mm/s]) between consecutive Vicon frames."""
    hand = [i for i, n in enumerate(labels) if not n.split(":")[-1].lower().startswith("forearm")]
    sp = np.nanmedian(np.linalg.norm(np.diff(X[:, hand], axis=0), axis=2), axis=1) * rate
    return (np.arange(len(sp)) + 0.5) / rate, np.nan_to_num(sp)


def motion_lag(video_t: np.ndarray, video_motion: np.ndarray, vicon_t: np.ndarray, vicon_speed: np.ndarray,
               search_s: float = 3.0, step_s: float = 0.005) -> tuple[float, float]:
    """(lag, r): the shift added to video_t that best correlates log-motion with log-speed."""
    lags = np.arange(-search_s, search_s + step_s / 2, step_s)
    r = np.full(len(lags), -1.0)
    for i, lag in enumerate(lags):
        v = np.interp(video_t + lag, vicon_t, vicon_speed, left=np.nan, right=np.nan)
        ok = ~np.isnan(v)
        if ok.sum() > 100:
            r[i] = np.corrcoef(np.log1p(v[ok]), np.log1p(video_motion[ok]))[0, 1]
    return float(lags[r.argmax()]), float(r.max())


def aux_pulse_offset(aux: np.ndarray, fs: float, vicon_csv: Path) -> float | None:
    """EMG-minus-Vicon time (s) of the 1 s sync pulse on Novecento AUX 1 and Vicon ``Sync``, or None."""
    a = find_trigger_pulse((np.abs(aux - np.median(aux)) > 8000).astype(float), fs)  # AUX is raw counts, +-32768 FS
    sync = read_sync_channel(vicon_csv)
    v = find_trigger_pulse(*sync) if sync else None
    return None if a is None or v is None else a - v
