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


def test_mask_offset_matches_pattern_with_extra_event():
    vicon = np.full(60_000, 0.007)                          # Vicon Sync, 1 kHz, volts
    vicon[12_550:12_770] = 1.25                             # lone pulse
    vicon[13_690:15_390] = _mains_burst(1_700, 1000.0)      # burst
    aux = np.zeros(140_000)                                 # AUX 1, 2 kHz, raw counts; EMG starts 9.77 s earlier
    aux[2 * 22_320:2 * 22_540] = 30_000
    aux[2 * 23_460:2 * 25_160] = 30_000 * np.sign(_mains_burst(3_400, 2000.0))
    aux[2 * 60_000:2 * 60_400] = 30_000                     # an event Vicon never saw
    lag, iou, amb = vs.mask_offset(vs.sync_mask(aux, 2000.0, 8000), vs.sync_mask(vicon, 1000.0, 0.1))
    assert abs(lag - 9.77) <= 0.002 and iou > 0.7 and amb < 0.5  # the unmatched event lowers IoU


def test_sync_mask_keeps_only_the_falling_edge_of_a_segment_high_at_start():
    sig = np.zeros(10_000)
    sig[:4_000] = 1.25                                      # already high when recording starts
    m = vs.sync_mask(sig, 1000.0, 0.1)
    assert m[:3_800].sum() == 0 and m[3_800:4_000].all() and not m[4_000:].any()


def test_sync_overrides_reads_offsets_and_confirmations(tmp_path):
    f = tmp_path / "sync_overrides.csv"
    f.write_text("trial,stream,offset_s,checked_by,date,note\n"
                 "P5/Trial1_Hands_only,emg,6.25,RL,2026-11-03,set by eye\n"
                 "P16/Trial2_HOI,emg,,RL,2026-11-03,auto value confirmed\n")
    ov = vs.sync_overrides(f)
    assert float(ov[("P5/Trial1_Hands_only", "emg")]["offset_s"]) == 6.25
    assert ov[("P16/Trial2_HOI", "emg")]["offset_s"] == "" and ov[("P16/Trial2_HOI", "emg")]["checked_by"] == "RL"
    assert vs.sync_overrides(tmp_path / "missing.csv") == {}
