"""_iterate_label_agnostic's mapping carry-through, convergence/cycle
detection, and _assign_and_verify's label-agnostic final-application branch.

MANO fitting itself (``mr.fit_frames``/``mr.predict_marker_positions``) is
monkeypatched out in the iteration tests -- these exercise the *bookkeeping*
around it (does the returned mapping actually correspond to the returned
prediction, does convergence/cycle detection fire correctly), not the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import relabel_with_mano as rwm   # noqa: E402


class _FakeRes:
    joints = np.zeros((1, 21, 3))
    betas = np.zeros(10)


def _fake_fit_frames(*_args, **_kwargs):
    return _FakeRes()


# Three present markers, mm. Marker 0 and 1 are swapped relative to `pred_a`.
_MARKERS = np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]])
_LABELS = ["A", "B", "C"]
_STATUS_ALL_CORRECT = np.array([[2, 2, 2]])
_TARGETS = np.array([0])

# metres; matches the 0<->1 swap in _MARKERS
_PRED_A = np.array([[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
# metres; matches a 1<->2 swap instead
_PRED_B = np.array([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]])
# metres; matches a 3-cycle (0<-2, 1<-0, 2<-1) -- distinct from both A and B
_PRED_C = np.array([[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])


def test_iterate_label_agnostic_converges_and_returns_consistent_mapping(monkeypatch):
    monkeypatch.setattr(rwm.mr, "fit_frames", _fake_fit_frames)
    monkeypatch.setattr(rwm.mr, "predict_marker_positions", lambda *a, **k: _PRED_A)

    res_t, pred, n_iters, mapping, converged = rwm._iterate_label_agnostic(
        _MARKERS, _LABELS, _STATUS_ALL_CORRECT, _TARGETS, _PRED_A, _FakeRes(),
        m2j={}, offsets={}, betas=np.zeros(10), mano_dir="unused", side="right",
        max_dist_mm=30.0, max_iters=5, min_trusted=1)

    assert converged is True
    assert n_iters == 2          # iter 1 finds the swap, iter 2 confirms it's stable
    assert mapping == {0: {0: 1, 1: 0}}
    # `pred` is the prediction the returned mapping was actually computed
    # from -- a caller must be able to trust this pairing.
    np.testing.assert_array_equal(pred, _PRED_A)


def test_iterate_label_agnostic_detects_an_oscillating_cycle(monkeypatch):
    # predict_marker_positions alternates B, A, ... after each refit, which
    # makes the assignment flip back and forth between two mappings forever
    # if nothing detects it. Iteration 3 should recognise iteration 1's
    # mapping recurring and stop rather than spinning to max_iters.
    calls = iter([_PRED_B, _PRED_A])
    monkeypatch.setattr(rwm.mr, "fit_frames", _fake_fit_frames)
    monkeypatch.setattr(rwm.mr, "predict_marker_positions",
                        lambda *a, **k: next(calls))

    res_t, pred, n_iters, mapping, converged = rwm._iterate_label_agnostic(
        _MARKERS, _LABELS, _STATUS_ALL_CORRECT, _TARGETS, _PRED_A, _FakeRes(),
        m2j={}, offsets={}, betas=np.zeros(10), mano_dir="unused", side="right",
        max_dist_mm=30.0, max_iters=10, min_trusted=1)

    assert converged == "cycle"
    assert n_iters == 3
    assert mapping == {0: {0: 1, 1: 0}}   # iteration 3's (repeated) mapping


def test_iterate_label_agnostic_reports_no_convergence_at_max_iters(monkeypatch):
    # predict cycles through three genuinely distinct, non-repeating
    # mappings (a 0<->1 swap, a 1<->2 swap, then a 3-cycle) so the loop
    # never converges and never revisits an earlier mapping within
    # max_iters=3 -- it must honestly report non-convergence rather than
    # claiming stability just because the loop ended.
    calls = iter([_PRED_B, _PRED_C, _PRED_A])
    monkeypatch.setattr(rwm.mr, "fit_frames", _fake_fit_frames)
    monkeypatch.setattr(rwm.mr, "predict_marker_positions",
                        lambda *a, **k: next(calls))

    _, _, n_iters, mapping, converged = rwm._iterate_label_agnostic(
        _MARKERS, _LABELS, _STATUS_ALL_CORRECT, _TARGETS, _PRED_A, _FakeRes(),
        m2j={}, offsets={}, betas=np.zeros(10), mano_dir="unused", side="right",
        max_dist_mm=30.0, max_iters=3, min_trusted=1)

    assert converged is False
    assert n_iters == 3
    assert mapping == {0: {0: 2, 1: 0, 2: 1}}   # the 3-cycle from iteration 3


def test_iterate_label_agnostic_respects_min_trusted_gate(monkeypatch):
    # Only 1 CORRECT marker in the frame; min_trusted=2 means the frame is
    # never eligible, so no reassignment should ever be proposed for it.
    status_undertrusted = np.array([[2, 1, 1]])
    monkeypatch.setattr(rwm.mr, "fit_frames", _fake_fit_frames)
    monkeypatch.setattr(rwm.mr, "predict_marker_positions", lambda *a, **k: _PRED_A)

    _, _, _, mapping, converged = rwm._iterate_label_agnostic(
        _MARKERS, _LABELS, status_undertrusted, _TARGETS, _PRED_A, _FakeRes(),
        m2j={}, offsets={}, betas=np.zeros(10), mano_dir="unused", side="right",
        max_dist_mm=30.0, max_iters=3, min_trusted=2)

    assert mapping == {}
    assert converged is True   # {} == {} immediately


class _DummyCtx(dict):
    pass


def _minimal_verify_ctx(*, fit_on, final_mapping):
    """Just enough of `ctx` to exercise `_assign_and_verify`'s branch logic
    for the reassignment step in isolation, without a real cascade/MANO run.
    """
    markers = np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]])
    labels = ["A", "B", "C"]
    status = np.array([[2, 1, 2]])   # B (index 1) is cascade-INCORRECT; A, C CORRECT
    return dict(participant="Px", trial="Tx", markers=markers, labels=labels,
                bones=[], status=status, pct=np.array([66.0]), targets=np.array([0]),
                pred=np.zeros((1, 3, 3)), frame_idx=np.array([0]), ref=np.array([0]),
                static_markers=None, m2j={}, res_t=_FakeRes(), n_confident=0,
                n_iters_run=1, fit_on=fit_on, final_mapping=final_mapping,
                converged=True, reliability=dict(ratio=0.4, verdict="good",
                                                  p95_err_mm=1.0, spacing_mm=2.0))


def test_label_agnostic_final_stage_can_move_an_originally_correct_marker(monkeypatch):
    # Marker A (index 0, cascade-CORRECT) and marker B (index 1,
    # cascade-INCORRECT) are swapped in the final converged mapping. The
    # flagged-only mechanism used by fit_on=correct/all could never propose
    # this (A was never flagged) -- this is the capability label-agnostic
    # mode exists to add.
    ctx = _minimal_verify_ctx(fit_on="label-agnostic",
                              final_mapping={0: {0: 1, 1: 0}})
    monkeypatch.setattr(rwm.lmq, "label_quality_cascade",
                        lambda *a, **k: (ctx["status"].copy(), []))
    monkeypatch.setattr(rwm.lmq, "build_reference", lambda *a, **k: ({}, None))

    out = rwm._assign_and_verify(ctx, max_dist_mm=30.0, min_trusted=1, verbose=False)

    s = out["summary"]
    assert s["orig_correct_moved"] == 1     # marker A moved despite being CORRECT
    assert s["orig_incorrect_moved"] == 1   # marker B moved too
    np.testing.assert_array_equal(out["relabelled"][0, 0], [100.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["relabelled"][0, 1], [0.0, 0.0, 0.0])


def test_correct_mode_final_stage_ignores_carried_mapping(monkeypatch):
    # fit_on="correct" must keep doing its own flagged-only recomputation
    # from `pred`, completely ignoring `final_mapping` -- even a
    # final_mapping that (wrongly, if it leaked through) would swap the two
    # cascade-CORRECT markers A and C.
    ctx = _minimal_verify_ctx(fit_on="correct",
                              final_mapping={0: {0: 2, 2: 0}})
    # Two flagged markers this time (B, C) so a real flagged<->flagged move
    # is actually possible -- a single flagged marker has no eligible
    # partner and could never move (see relabel_frame's docstring).
    ctx["status"] = np.array([[2, 1, 1]])
    # Predicts that B and C's *observed* positions should swap.
    ctx["pred"] = np.array([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    monkeypatch.setattr(rwm.lmq, "label_quality_cascade",
                        lambda *a, **k: (ctx["status"].copy(), []))
    monkeypatch.setattr(rwm.lmq, "build_reference", lambda *a, **k: ({}, None))

    out = rwm._assign_and_verify(ctx, max_dist_mm=30.0, min_trusted=1, verbose=False)

    s = out["summary"]
    # The real flagged<->flagged move (B<->C) happened...
    assert s["orig_incorrect_moved"] == 2
    # ...but A (never flagged) was not touched, unlike what the carried
    # final_mapping (A<->C) would have done if it had leaked through.
    assert s["orig_correct_moved"] == 0
    np.testing.assert_array_equal(out["relabelled"][0, 0], ctx["markers"][0, 0])
    np.testing.assert_array_equal(out["relabelled"][0, 1], [200.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["relabelled"][0, 2], [100.0, 0.0, 0.0])


def test_regression_rejection_keeps_original_marker_array_unchanged(monkeypatch):
    # The proposed repair makes the cascade verdict worse -- the whole
    # relabelled array must be discarded, not just flagged as bad.
    ctx = _minimal_verify_ctx(fit_on="label-agnostic",
                              final_mapping={0: {0: 1, 1: 0}})
    before_status = ctx["status"].copy()
    worse_status = np.array([[1, 1, 1]])   # strictly fewer CORRECT than before

    monkeypatch.setattr(rwm.lmq, "label_quality_cascade",
                        lambda *a, **k: (worse_status.copy(), []))
    monkeypatch.setattr(rwm.lmq, "build_reference", lambda *a, **k: ({}, None))

    out = rwm._assign_and_verify(ctx, max_dist_mm=30.0, min_trusted=1, verbose=False)

    s = out["summary"]
    assert s["rejected"] is True
    assert s["moved"] == 0
    np.testing.assert_array_equal(out["relabelled"], ctx["markers"])
    np.testing.assert_array_equal(out["status"], before_status)
