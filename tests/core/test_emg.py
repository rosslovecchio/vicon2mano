"""Tests for vicon2mano.core.emg -- .otb4 decoding and channel quality flags."""

from __future__ import annotations

import io
import tarfile

import numpy as np

from vicon2mano.core.emg import channel_quality, flag_channels, ramp_faults, read_otb4

FS, T = 2000, 2000 * 6


def _track(title, sig, n_in_block, off, chans, nmax=1):
    return (f"<TrackInfo><Title>{title}</Title><SignalStreamPath>{sig}</SignalStreamPath>"
            f"<MaxNumberOfChannels>{nmax}</MaxNumberOfChannels><ChannelsInBlock>{n_in_block}</ChannelsInBlock>"
            f"<ChannelOffsetInSubPacket>{off}</ChannelOffsetInSubPacket><SamplingFrequency>{FS}</SamplingFrequency>"
            f"<ADC_Range>4.8</ADC_Range><ADC_Nbits>24</ADC_Nbits><Gain>1</Gain>"
            f"<StringsDescriptions><Channels>{chans}</Channels><OriginalSensor>HD08MM1606</OriginalSensor>"
            f"</StringsDescriptions></TrackInfo>")


def _grid(rng):
    common = rng.normal(0, 300, T)                         # common mode, removed by CAR
    local = np.repeat(rng.normal(0, 200, (24, T)), 4, 0)   # shared by groups of 4 -> neighbours correlate
    emg = (common + local + rng.normal(0, 60, (96, T))).astype(np.int32)
    emg[10] = 0                                            # dead
    emg[20, ::50] = -32768                                 # clipping (2%)
    return emg


def _write_otb4(path, rng):
    probe = np.zeros((102, T), np.int32)
    probe[:96] = emg = _grid(rng)
    ramp = (np.arange(T) - 12000).astype(np.int32)
    ramp[500:] -= 1                                        # one repeated sample at 500
    ramp[900:] += 2                                        # one skipped sample at 900
    probe[101] = ramp
    aux = np.zeros((16, T), np.int32)
    aux[0, 4000:4100] = 30000
    xml = ("<ArrayOfTrackInfo>" + _track("Novecento+ (1 - 96)", "a.sig", 102, 0, "1 - 96", 96)
           + _track("Novecento+ Ramp", "a.sig", 102, 101, "1 - 96")
           + _track("Novecento+", "b.sig", 16, 0, "AUX 1") + "</ArrayOfTrackInfo>")
    with tarfile.open(path, "w") as tar:
        for name, data in [("Tracks_000.xml", xml.encode()), ("a.sig", probe.tobytes(order="F")),
                           ("b.sig", aux.tobytes(order="F"))]:
            info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    return emg


def test_read_otb4_roundtrip_and_scale(tmp_path):
    emg = _write_otb4(tmp_path / "x.otb4", np.random.default_rng(0))
    rec = read_otb4(tmp_path / "x.otb4")
    assert rec.fs == FS and list(rec.grids) == ["1 - 96"]
    np.testing.assert_array_equal(rec.grids["1 - 96"], emg)
    assert rec.mv_per_count == 4.8 / 2 ** 24 * 1000      # OT's own formula
    assert rec.aux["AUX 1"][4050] == 30000
    lost, rep = ramp_faults(rec.ramps["1 - 96"])
    assert list(lost) == [899] and list(rep) == [499]


def test_ramp_wrap_is_not_a_fault():
    ramp = ((np.arange(70000) + 32000) % 65536 - 32768).astype(np.int32)
    lost, rep = ramp_faults(ramp)
    assert not len(lost) and not len(rep)


def test_flags_dead_and_clipping_only():
    raw = _grid(np.random.default_rng(1))
    q = channel_quality(raw, FS, 4.8 / 2 ** 24 * 1000)
    flags = flag_channels(q)
    assert flags[10] == "dead" and np.isnan(q["noise_uv"][10])
    assert "clipping" in flags[20]
    assert sum(bool(f) for f in flags) == 2
