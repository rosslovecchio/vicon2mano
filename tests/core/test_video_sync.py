"""Tests for vicon2mano.core.video_sync -- RealSense-to-Vicon clock offset."""

from __future__ import annotations

import numpy as np

from vicon2mano.core import video_sync as vs


def _mains_burst(n: int, rate: float) -> np.ndarray:
    """Clipped 50 Hz pickup, as seen on the Sync channel when the pulse is absent."""
    return np.clip(5 * np.sin(2 * np.pi * 50 * np.arange(n) / rate), -1.25, 1.25)


def test_trigger_pulse_found_after_mains_noise():
    rate = 1000.0
    sig = np.full(20_000, 0.007)
    sig[2_000:4_000] = _mains_burst(2_000, rate)
    sig[8_706:9_706] = 1.25
    assert vs.find_trigger_pulse(sig, rate) == 8.706


def test_no_pulse_in_mains_noise_or_short_spike():
    rate = 1000.0
    sig = np.full(20_000, 0.007)
    sig[2_000:9_000] = _mains_burst(7_000, rate)
    sig[12_547:12_770] = 1.25  # 223 ms, the P15 Trial1 spike
    assert vs.find_trigger_pulse(sig, rate) is None


def test_motion_lag_recovers_shift():
    rng = np.random.default_rng(0)
    vicon_t = np.arange(0, 120, 1 / 200)
    speed = np.convolve(rng.gamma(0.3, 200, vicon_t.size), np.ones(40) / 40, "same")
    true = 0.57
    video_t = np.sort(rng.uniform(5, 110, 2000))  # irregular, like a frame-dropping recorder
    motion = np.interp(video_t + true, vicon_t, speed) * rng.uniform(0.8, 1.2, video_t.size)
    lag, r = vs.motion_lag(video_t, motion, vicon_t, speed)
    assert abs(lag - true) <= 0.01 and r > 0.8


def test_aux_pulse_offset_matches_emg_to_vicon(tmp_path):
    fs_e = 2000.0
    vicon = np.full(12_000, 0.007)
    vicon[1_906:2_906] = 1.25                              # Vicon Sync: 1 s pulse at 1.906 s
    csv_path = tmp_path / "trial.csv"
    csv_path.write_text("﻿Devices\n1000\n,,Imported Generic Analog #1 - Electric Potential,\nFrame,Sub Frame,Sync\n,,V\n"
                        + "".join(f"{i // 5 + 1},{i % 5},{v}\n" for i, v in enumerate(vicon)) + "\nTrajectories\n200\n",
                        encoding="utf-8")
    aux = np.zeros(30_000)
    aux[11_676:13_676] = 30_000                            # AUX 1 (raw counts): same pulse at 5.838 s
    assert abs(vs.aux_pulse_offset(aux, fs_e, csv_path) - 3.932) < 1e-9
    assert vs.aux_pulse_offset(np.zeros(30_000), fs_e, csv_path) is None
