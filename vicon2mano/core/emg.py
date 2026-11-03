"""OT Bioelettronica Novecento+ ``.otb4`` reader and per-channel HD-sEMG quality metrics.

An ``.otb4`` is a tar of ``Tracks_000.xml`` plus one interleaved int32 ``.sig``
per probe. Decoding follows OT's own reader
(github.com/OTBioelettronica/OTB-Python, ``OpenOTB4.py``): Fortran-order
reshape to (channels, samples), then ``raw * ADC_Range / 2**ADC_Nbits * 1000 /
Gain`` -> mV. That script cannot be imported (file dialog at module level) and
expects XML keys Novecento+ exports lack, so the layout is read from
``ChannelsInBlock`` / ``ChannelOffsetInSubPacket`` instead. Verified
bit-identical (5e-17 mV) to OTBiolab's own ``.npy`` export.

Each probe file holds 96 EMG + 4 quaternion + buffer + ramp channels. The ramp
is a +1/sample 16-bit counter, so any other step is a lost or repeated sample.
"""
from __future__ import annotations

import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

EMG_ROOT = Path(r"D:\june25_data")
RAIL = 32767  # EMG samples never leave int16 range; +-RAIL is clipping
GRID_ROWS, GRID_COLS = 16, 6  # HD08MM1606, channel k (0-based) at row k % 16, column k // 16: inferred from
# neighbour correlation (P16/Trial2/HOI: down 0.61-0.70, across 0.40-0.65, column wrap 16->17 <= 0.13)


@dataclass
class Otb4Recording:
    """Decoded ``.otb4``: raw int32 EMG per grid, ramps, and AUX channels (raw counts)."""
    fs: float
    grids: dict[str, np.ndarray] = field(default_factory=dict)   # title -> (96, T) int32
    ramps: dict[str, np.ndarray] = field(default_factory=dict)   # title -> (T,) int32
    aux: dict[str, np.ndarray] = field(default_factory=dict)     # "AUX 1" -> (T,) int32, +-32768 full scale
    mv_per_count: float = 0.0
    sensor: str = ""


def read_otb4(path: Path) -> Otb4Recording:
    """Stream an ``.otb4`` from its tar without extracting it to disk."""
    with tarfile.open(path) as tar:
        tracks = ET.parse(tar.extractfile("Tracks_000.xml")).getroot().findall("TrackInfo")
        sigs = {}

        def block(t):
            name = t.findtext("SignalStreamPath")
            if name not in sigs:
                n = int(t.findtext("ChannelsInBlock"))
                sigs[name] = np.frombuffer(tar.extractfile(name).read(), np.int32).reshape(n, -1, order="F")
            return sigs[name]

        rec = None
        for t in tracks:
            fs, off = float(t.findtext("SamplingFrequency")), int(t.findtext("ChannelOffsetInSubPacket"))
            title, chans = t.findtext("Title"), t.findtext("StringsDescriptions/Channels")
            if rec is None:
                rec = Otb4Recording(fs=fs)
            if fs != rec.fs:          # 8 kHz device-control stream, not needed
                continue
            if title.startswith("Novecento+ (") and t.findtext("MaxNumberOfChannels") == "96":
                rec.grids[chans] = block(t)[off:off + 96]
                rec.mv_per_count = float(t.findtext("ADC_Range")) / 2 ** int(t.findtext("ADC_Nbits")) * 1000 / float(t.findtext("Gain"))
                rec.sensor = t.findtext("StringsDescriptions/OriginalSensor")
                grid_key = chans
            elif title == "Novecento+ Ramp":
                rec.ramps[grid_key] = block(t)[off]
            elif chans.startswith("AUX"):
                rec.aux[chans] = block(t)[off]
    return rec


def ramp_faults(ramp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample indices where the ramp counter skipped (lost samples) or repeated."""
    step = np.diff(ramp.astype(np.int64)) % 65536
    return np.flatnonzero(step > 1), np.flatnonzero(step == 0)


def windowed(x: np.ndarray, n: int) -> np.ndarray:
    """Reshape (C, T) to (C, T // n, n), dropping the ragged tail."""
    w = x.shape[-1] // n
    return x[..., :w * n].reshape(*x.shape[:-1], w, n)


def _bandpass_sos(fs: float):
    return signal.butter(4, [20, 450], "bandpass", fs=fs, output="sos")


def car_bandpass(raw: np.ndarray, fs: float, mv_per_count: float):
    """(per-second p2p counts, dead mask, monopolar uV, CAR uV, CAR 20-450 Hz uV) of one (C, T) grid; dead rows NaN."""
    p2p = np.ptp(windowed(raw, int(fs)), axis=-1)                         # (C, seconds)
    dead = (p2p <= 2).all(1)
    x = raw.astype(np.float32) * np.float32(mv_per_count * 1000)          # uV
    x[dead] = np.nan
    car = x - np.nanmedian(x, axis=0)
    bp = signal.sosfiltfilt(_bandpass_sos(fs), np.nan_to_num(car), axis=1).astype(np.float32)
    bp[dead] = np.nan
    return p2p, dead, x, car, bp


def rms_envelope(raw: np.ndarray, fs: float, mv_per_count: float, win_s: float = 0.05) -> np.ndarray:
    """(C, W) RMS in uV of the CAR 20-450 Hz signal over consecutive win_s windows."""
    bp = car_bandpass(raw, fs, mv_per_count)[4]
    return np.sqrt((windowed(bp, int(win_s * fs)) ** 2).mean(-1))


def channel_quality(raw: np.ndarray, fs: float, mv_per_count: float, win_s: float = 0.5) -> dict[str, np.ndarray]:
    """Per-channel quality metrics for one (C, T) monopolar grid, plus the (C, W) windowed RMS in uV.

    Rail/flat are judged on raw counts. Everything else is on the common-average
    reference (grid median removed): monopolar signals are dominated by shared
    common mode (every channel correlates ~0.98), which hides bad contacts.
    ``mono_noise_uv`` keeps the monopolar floor. ``nbr_corr`` is the mean of a
    channel's 3 highest correlations with the rest of the grid: neighbours on an
    8 mm grid share most of their signal, so one matching nothing has poor contact.
    Dead (all-flat) channels get NaN.
    """
    p2p, dead, x, car, bp = car_bandpass(raw, fs, mv_per_count)
    mono = signal.sosfiltfilt(_bandpass_sos(fs), np.nan_to_num(x), axis=1).astype(np.float32)
    rms = np.sqrt((windowed(bp, int(win_s * fs)) ** 2).mean(-1))          # (C, W)
    mono_rms = np.sqrt((windowed(mono, int(win_s * fs)) ** 2).mean(-1))

    f, pxx = signal.welch(car, fs=fs, nperseg=int(fs), axis=1)
    band = lambda lo, hi: pxx[:, (f >= lo) & (f < hi)].sum(1)
    emg = band(20, 450)
    line = sum(band(h - 1, h + 2) for h in (50, 100, 150, 200, 250))

    z = (bp - bp.mean(1, keepdims=True)) / bp.std(1, keepdims=True)
    z = np.nan_to_num(z)
    corr = (z @ z.T) / z.shape[1]
    np.fill_diagonal(corr, -1)
    noise, active = np.percentile(rms, 5, axis=1), np.percentile(rms, 99, axis=1)
    return dict(
        dead=dead,
        rail_frac=(np.abs(raw) >= RAIL).mean(1),
        flat_s=(p2p <= 2).sum(1).astype(float),
        offset_mv=np.nanmedian(x, axis=1) / 1000,
        mono_noise_uv=np.where(dead, np.nan, np.percentile(mono_rms, 5, axis=1)),
        noise_uv=noise, active_uv=active, snr_db=20 * np.log10(active / noise),
        line_ratio=line / emg, lowfreq_ratio=band(1, 20) / emg,
        nbr_corr=np.where(dead, np.nan, np.sort(corr, axis=1)[:, -3:].mean(1)),
        rms=rms,
    )


def flag_channels(q: dict[str, np.ndarray]) -> list[str]:
    """Per-channel ';'-joined reasons a channel is unusable ('' = good).

    Noise and line thresholds are relative to the grid median *or* absolute,
    so a grid that fails as a whole (P17/Trial2/Hands_only: 250 uV, 98% mains)
    cannot hide behind its own median.
    """
    med = lambda k: np.nanmedian(q[k])
    rules = {
        "dead": q["dead"],
        "clipping": q["rail_frac"] > 1e-3,
        "dropout": (q["flat_s"] > 1) & ~q["dead"],
        "noisy": q["noise_uv"] > min(3 * med("noise_uv"), 50),
        "line": q["line_ratio"] > min(2 * med("line_ratio"), 0.5),
        "motion": q["lowfreq_ratio"] > 1,
        "isolated": q["nbr_corr"] < 0.5,
    }
    return [";".join(k for k, m in rules.items() if m[i]) for i in range(len(q["dead"]))]
