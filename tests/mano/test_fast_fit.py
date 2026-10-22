"""Tests for the pieces that make the surface fit fast without changing it:
the sync-free smplx kinematic chain, the masked mean, and ``run_adam``
(eager vs CUDA-graph).

fit_surface_markers.py is imported by path since scripts/ isn't a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_MANO = Path(__file__).resolve().parents[2] / "scripts" / "mano"
if str(SCRIPTS_MANO) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_MANO))

import fit_surface_markers as fsm  # noqa: E402
from vicon2mano.core import fitter  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _random_tree(n=16):
    parents = torch.tensor([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14][:n])
    return parents


def test_nosync_rigid_transform_matches_upstream():
    if not fitter._SMPLX_AVAILABLE:
        pytest.skip("smplx not installed")
    torch.manual_seed(0)
    B, N = 5, 16
    parents = _random_tree(N)
    rot = torch.linalg.matrix_exp(torch.randn(B, N, 3, 3) * 0.3 - torch.randn(B, N, 3, 3).transpose(-1, -2) * 0.3)
    joints = torch.randn(B, N, 3)
    up_j, up_t = fitter._UPSTREAM_BATCH_RIGID_TRANSFORM(rot, joints, parents)
    new_j, new_t = fitter._batch_rigid_transform_nosync(rot, joints, parents)
    torch.testing.assert_close(new_j, up_j)
    torch.testing.assert_close(new_t, up_t)


def test_masked_sq_mean_matches_boolean_indexing():
    torch.manual_seed(1)
    d = torch.randn(7, 5, 3)
    mask = torch.rand(7, 5) > 0.3
    w = mask.float()
    want = (d ** 2).sum(-1)[mask].mean()
    got = fsm._masked_sq_mean(d, w, w.sum())
    torch.testing.assert_close(got, want)


def _quadratic(device):
    target = torch.tensor([1.0, -2.0, 3.0], device=device)
    x = torch.nn.Parameter(torch.zeros(3, device=device))
    return x, target


def test_run_adam_eager_converges_and_reports():
    x, target = _quadratic("cpu")
    opt = torch.optim.Adam([x], lr=0.1)
    out = fsm.run_adam(opt, lambda: ((x - target) ** 2).sum(), 300,
                       report=lambda: float(((x - target) ** 2).sum()))
    assert out < 1e-3


def test_run_adam_raises_on_non_finite():
    x, target = _quadratic("cpu")
    opt = torch.optim.Adam([x], lr=0.1)
    with pytest.raises(RuntimeError, match="non-finite"):
        fsm.run_adam(opt, lambda: (x - target).sum() * float("nan"), 5, label="t")


def test_run_adam_post_grad_masks_gradient():
    x, target = _quadratic("cpu")
    mask = torch.tensor([1.0, 0.0, 1.0])
    opt = torch.optim.Adam([x], lr=0.1)
    fsm.run_adam(opt, lambda: ((x - target) ** 2).sum(), 50,
                 post_grad=lambda: x.grad.mul_(mask))
    assert x[1].item() == 0.0 and x[0].item() > 0.5


@needs_cuda
def test_run_adam_graph_matches_eager():
    results = {}
    for graph in (False, True):
        x, target = _quadratic("cuda")
        opt = torch.optim.Adam([x], lr=0.1, capturable=graph)
        fsm.run_adam(opt, lambda: ((x - target) ** 2).sum(), 120, graph=graph)
        results[graph] = x.detach().cpu().numpy()
    np.testing.assert_allclose(results[True], results[False], atol=1e-4)


def test_run_adam_graph_flag_is_harmless_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("covered by the CUDA test")
    x, target = _quadratic("cpu")
    opt = torch.optim.Adam([x], lr=0.1)
    fsm.run_adam(opt, lambda: ((x - target) ** 2).sum(), 100, graph=True)
    assert float(((x - target) ** 2).sum()) < 1e-2


def test_angles_from_markers_needs_only_palm2():
    """P1/P2 never have Palm1/Palm3; the seed must still read the fingers."""
    names = ["Palm2", "Index1", "Index2", "Index3"]
    # wrist at origin, finger bent 90 deg at the PIP
    mk = np.array([[[0, 0, 0], [0, 50, 0], [0, 80, 0], [0, 80, 30]]], float)
    Bmat = np.zeros((15, 3, 3), np.float32)
    Bmat[:, 0, 0] = 1.0
    hp = fsm.angles_from_markers(mk, names, Bmat)
    assert abs(hp[0, 3] - np.pi / 2) < 1e-4          # Index PIP flexion
    assert fsm.angles_from_markers(mk, names[1:], Bmat).any() == False  # no Palm2 -> flat


def test_thumb_slide_moves_only_thumb_markers_along_their_segment():
    J = torch.zeros(1, 21, 3)
    J[0, 2] = torch.tensor([0.0, 0.0, 0.0]); J[0, 3] = torch.tensor([0.0, 0.1, 0.0])
    off = fsm.thumb_slide(J, ["Index2", "Thumb3"], torch.tensor([5.0, 5.0]))
    assert torch.allclose(off[0, 0], torch.zeros(3))                       # not a thumb marker
    assert torch.allclose(off[0, 1], torch.tensor([0.0, 0.005, 0.0]), atol=1e-7)   # 5 mm along 2->3


def _stub_model(weights):
    """A MANO stand-in: flat palm in the z=0 plane, one vertex 10 mm above the wrist plane, one fingertip vertex."""
    class Out:
        pass

    class Stub:
        lbs_weights = torch.tensor(weights, dtype=torch.float32)

        def __call__(self, **kw):
            o = Out()
            J = torch.zeros(1, 16, 3)
            J[0, 1] = torch.tensor([1.0, 0.0, 0.0])                # index MCP
            J[0, 7] = torch.tensor([0.0, 1.0, 0.0])                # pinky MCP -> palm normal +z
            o.joints = J
            o.vertices = torch.tensor([[[0.2, 0.2, 0.01], [0.5, 0.5, 0.01]]])
            return o
    return Stub()


def test_palm_thickness_scales_only_palm_weighted_vertices_along_the_normal():
    model = fsm.ScaledModel(_stub_model([[1.0] + [0.0] * 15, [0.0] * 16]), "cpu", thickness=True, limit=1.0)
    base = model(a=1).vertices.clone()
    assert torch.allclose(base, torch.tensor([[[0.2, 0.2, 0.01], [0.5, 0.5, 0.01]]]))           # thickness 1 = unchanged
    model.log_thick.data.fill_(float(np.log(2.0)))
    out = model(a=1).vertices
    assert torch.allclose(out[0, 0], torch.tensor([0.2, 0.2, 0.02]), atol=1e-6)               # palm vertex: height doubled
    assert torch.allclose(out[0, 1], base[0, 1])                                              # finger vertex (palm weight 0): untouched


def test_palm_thickness_is_bounded():
    model = fsm.ScaledModel(_stub_model([[1.0] + [0.0] * 15, [0.0] * 16]), "cpu", thickness=True, limit=0.25)
    model.log_thick.data.fill_(float(np.log(0.1)))                    # asks for a paper-thin palm
    assert torch.allclose(model(a=1).vertices[0, 0, 2], torch.tensor(0.01 / 1.25 ), atol=1e-6)
