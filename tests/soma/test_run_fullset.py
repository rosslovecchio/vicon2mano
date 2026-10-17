"""Tests for the decision logic in scripts/soma/run_fullset.py: which file is a
participant's static, how the surface-fit command is built from a config, how a
bad static is recognised, and participant selection."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPTS_SOMA = Path(__file__).resolve().parents[2] / "scripts" / "soma"
if str(SCRIPTS_SOMA) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SOMA))

import run_fullset as rf  # noqa: E402


def _touch(d: Path, *names):
    for n in names:
        (d / n).write_text("x")


def test_static_from_log_wins(tmp_path):
    _touch(tmp_path, "static_trial.csv", "Static.csv")
    f, how = rf.resolve_static("PX", {"Static": "static_trial"}, tmp_path)
    assert f.name == "static_trial.csv" and how == "manual_frames.csv"


def test_static_by_exact_name_only(tmp_path):
    _touch(tmp_path, "Static trial.csv", "Trial1.csv")
    f, how = rf.resolve_static("PX", {}, tmp_path)
    assert f.name == "Static trial.csv" and how == "file name"


def test_numbered_static_is_not_a_static(tmp_path):
    # P12's Static_trial01..04 are movement trials; Static01 on P17 is mapped to a trial
    _touch(tmp_path, "Static_trial01.csv", "Static01.csv")
    f, how = rf.resolve_static("PX", {}, tmp_path)
    assert f is None and how == "none found"


def test_ambiguous_static_is_not_guessed(tmp_path):
    _touch(tmp_path, "Static.csv", "static_trial.csv")
    f, how = rf.resolve_static("PX", {}, tmp_path)
    assert f is None and how.startswith("ambiguous")


def test_override_beats_everything(tmp_path):
    _touch(tmp_path, "Static.csv")
    other = tmp_path / "mine.csv"; other.write_text("x")
    f, how = rf.resolve_static("PX", {}, tmp_path, {"PX": str(other)})
    assert f == other and how == "override"


def test_resolve_trials_skips_static_and_reports_missing(tmp_path):
    _touch(tmp_path, "a.csv")
    out = rf.resolve_trials("PX", {"Static": "s", "Trial 1 Hands only": "a", "Trial2 HOI": "gone"}, tmp_path)
    assert [(s, g, p is not None) for s, g, p in out] == [
        ("Trial 1 Hands only", "Trial_1_Hands_only", True), ("Trial2 HOI", "Trial2_HOI", False)]


def test_fit_args_from_config(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "side": "right", "attach_file": "configs/surface_fit/P10_attach.json",
        "params": {"standoff": 3.0, "staged": True, "freeze_root": True, "marker_init": True,
                   "torsion_only": False, "no_pca": True, "iters_thumb": 800, "mv_chunk": 6000,
                   "stretch_file": "configs/surface_fit/P10_stretch.json"}}))
    a = rf.fit_args_from_config(cfg)
    assert a[:2] == ["--side", "right"]
    assert "--staged" in a and "--no-pca" in a
    assert "--torsion-only" not in a                    # False -> omitted
    assert "--freeze-root" not in a and "--marker-init" not in a   # already defaults
    assert "--mv-chunk" not in a                        # the driver sets chunking itself
    assert a[a.index("--iters-thumb") + 1] == "800"
    assert a[a.index("--stretch") + 1].endswith("P10_stretch.json")
    assert a[a.index("--attach") + 1].endswith("P10_attach.json")


def test_real_p10_config_is_accepted_by_the_fit_cli():
    # every flag the driver emits must exist in fit_surface_markers
    import re
    src = (Path(rf.REPO) / "scripts/mano/fit_surface_markers.py").read_text(encoding="utf-8", errors="replace")
    known = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))
    for flag in rf.fit_args_from_config(Path(rf.REPO) / "configs/surface_fit/P10.json"):
        if flag.startswith("--"):
            assert flag in known, flag


CAL_GOOD = """betas: [0.1, -1.2, 0.3]  (base subject: [0.0])
marker    fit err mm  note
Thumb3           2.1
Forearm1             -  forearm: not fitted, keeps base offsets
Index3               -  absent in static
Index1-Index2        40.0      39.0     30.0-48.0
"""
CAL_BAD = """betas: [0.1, -4.5, 0.3]
Index3          20.1  SUSPECT: excluded, keeps base offsets
Ring1               -  absent in static
Ring2               -  absent in static
Ring3               -  absent in static
Index1-Index2        90.0      39.0     30.0-48.0  <- outside base p5-p95
"""
ARGS = Namespace(max_beta=3.0, max_absent=2, max_static_fit_mm=8.0)


def test_parse_calibration():
    c = rf.parse_calibration(CAL_GOOD)
    assert c["max_abs_beta"] == pytest.approx(1.2) and c["suspects"] == [] and c["absent"] == ["Index3"]
    b = rf.parse_calibration(CAL_BAD)
    assert b["suspects"] == ["Index3"] and b["max_abs_beta"] == pytest.approx(4.5)
    assert b["n_segments_outside"] == 1


def test_good_static_is_assumed_good():
    assert rf.static_verdict(rf.parse_calibration(CAL_GOOD), 5.2, ARGS) == []


def test_bad_static_is_flagged_for_every_reason():
    why = rf.static_verdict(rf.parse_calibration(CAL_BAD), 12.0, ARGS)
    text = " | ".join(why)
    assert "suspect markers" in text and "extreme fitted shape" in text
    assert "hand markers absent" in text and "static surface fit" in text


def test_forearm_absence_alone_does_not_flag():
    cal = dict(suspects=[], max_abs_beta=0.5, absent=["Forearm3", "Forearm4"], n_segments_outside=0)
    assert rf.static_verdict(cal, 5.0, ARGS) == []


def test_parse_static_fit_takes_the_last_mean():
    log = "  MEAN |d|      7.24 mm   (17/18 markers inside the mesh)\n...\n  MEAN |d|      5.25 mm   (2/18)\n"
    assert rf.parse_static_fit_mm(log) == pytest.approx(5.25)
    assert rf.parse_static_fit_mm("nothing") is None


def test_plan_skips_and_shards():
    sessions = {p: {} for p in ("P1", "P2", "P3", "P10", "P4")}
    a = Namespace(participants="all", skip="P1,P2", shard="0/2")
    assert rf.plan(a, sessions) == ["P3", "P10"]         # numeric order P3,P4,P10 -> shard 0 takes P3,P10
    a.shard = "1/2"
    assert rf.plan(a, sessions) == ["P4"]
    a.participants, a.shard = "P3,P4", "0/1"
    assert rf.plan(a, sessions) == ["P3", "P4"]


def test_unknown_participant_is_an_error_not_a_guess():
    with pytest.raises(SystemExit, match="no trial list"):
        rf.plan(Namespace(participants="Px5", skip="", shard="0/1"), {"P3": {}})


def test_part_name_keeps_extension():
    assert rf.part(Path("a/b/fit.npz")).name == "fit.part.npz"


def test_prune_attach_drops_only_markers_the_static_lacks(tmp_path):
    cfg = rf.REPO / "configs/surface_fit/P10.json"
    full = json.loads((rf.REPO / "configs/surface_fit/P10_attach.json").read_text())
    present = set(full) - {"Ring2"}
    out, dropped = rf.prune_attach(cfg, present, tmp_path / "attach.json")
    kept = json.loads(out.read_text())
    assert dropped == ["Ring2"] and "Ring2" not in kept and set(kept) == present


def test_prune_attach_without_attach_file(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"params": {}}))
    assert rf.prune_attach(cfg, {"Palm1"}, tmp_path / "a.json") == (None, [])
