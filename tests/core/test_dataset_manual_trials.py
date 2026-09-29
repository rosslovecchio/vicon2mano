"""Tests for the manually-labelled-trial discovery added to
vicon2mano.core.dataset for scripts/shared/aggregate_manual_agreement.py.

Uses a throwaway directory tree (tmp_path) shaped like a participant's data
folder, rather than the real OneDrive data root, so these run without that
mount present.
"""

from __future__ import annotations

from vicon2mano.core import dataset as ds


def _touch(path):
    path.write_text("", encoding="utf-8")


def test_find_manual_csv_prefers_filled_suffix(tmp_path):
    vicon = tmp_path / "Trial1_handsonly.csv"
    _touch(vicon)
    _touch(tmp_path / "Trial1_handsonly_manuallylabelled.csv")
    _touch(tmp_path / "Trial1_handsonly_manuallylabelled_filled.csv")
    found = ds.find_manual_csv(vicon)
    assert found.name == "Trial1_handsonly_manuallylabelled_filled.csv"


def test_find_manual_csv_falls_back_to_plain_suffix(tmp_path):
    vicon = tmp_path / "Trial1_handsonly.csv"
    _touch(vicon)
    _touch(tmp_path / "Trial1_handsonly_manuallylabelled.csv")
    found = ds.find_manual_csv(vicon)
    assert found.name == "Trial1_handsonly_manuallylabelled.csv"


def test_find_manual_csv_returns_none_when_absent(tmp_path):
    vicon = tmp_path / "Trial1_handsonly.csv"
    _touch(vicon)
    assert ds.find_manual_csv(vicon) is None


def test_find_manually_labelled_trials_across_participants(tmp_path):
    p10 = tmp_path / "P10"
    p10.mkdir()
    _touch(p10 / "Trial2_handsonly.csv")
    _touch(p10 / "Trial2_handsonly_manuallylabelled.csv")
    _touch(p10 / "Trial2_handsonly_manuallylabelled_filled.csv")

    p7 = tmp_path / "P7"
    p7.mkdir()
    _touch(p7 / "Trial1_handsonly.csv")
    _touch(p7 / "Trial1_handsonly_manuallylabelled.csv")

    found = ds.find_manually_labelled_trials(tmp_path)
    by_participant = {p: (t, v.name, m.name) for p, t, v, m in found}

    assert len(found) == 2
    assert by_participant["P10"] == ("Trial2_handsonly", "Trial2_handsonly.csv",
                                     "Trial2_handsonly_manuallylabelled_filled.csv")
    assert by_participant["P7"] == ("Trial1_handsonly", "Trial1_handsonly.csv",
                                    "Trial1_handsonly_manuallylabelled.csv")


def test_find_manually_labelled_trials_skips_orphaned_manual_export(tmp_path, capsys):
    p1 = tmp_path / "P1"
    p1.mkdir()
    # Manual export with no corresponding raw vicon CSV on disk.
    _touch(p1 / "Trial1_handsonly_manuallylabelled.csv")

    found = ds.find_manually_labelled_trials(tmp_path)
    assert found == []
    assert "no matching vicon CSV" in capsys.readouterr().out


def test_find_manually_labelled_trials_empty_root_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert ds.find_manually_labelled_trials(missing) == []
