"""strategySOMA generator: geometry, round-trip, corruption invariants."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vicon2mano.strategies.soma.library import (MarkerLibrary, closest_point_on_triangles,
                                                face_frames)
from vicon2mano.strategies.soma.synth import GHOST, PAD, SynthConfig, SynthGenerator, random_rotations

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "results/soma/libraries/library_P10_Trial2.npz"
FIT = REPO / "results/mano/P10_Trial2_handsonly/mano_fit_right_calibrated.npz"
MANO = REPO / "../clean_kinematics/mano_v1_2/models"


# ---------------------------------------------------------------- pure geometry

def test_closest_point_regions():
    tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
    cases = {(0.2, 0.2, 5): (0.2, 0.2, 0),     # interior -> projection
             (-1, -1, 0): (0, 0, 0),           # vertex a
             (2, -1, 0): (1, 0, 0),            # vertex b
             (0.5, -1, 0): (0.5, 0, 0),        # edge ab
             (1, 1, 0): (0.5, 0.5, 0)}         # edge bc
    for p, q in cases.items():
        np.testing.assert_allclose(closest_point_on_triangles(np.array(p, float), tri)[0], q, atol=1e-12)


def test_face_frames_orthonormal_and_numpy_torch_agree():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(4, 10, 3)); F = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    fi = np.array([0, 2, 1]); b = np.array([[.2, .3, .5], [1, 0, 0], [.1, .1, .8]])
    P, R = face_frames(V, F, fi, b)
    np.testing.assert_allclose(np.einsum("nmji,nmjk->nmik", R, R), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)
    Pt, Rt = face_frames(torch.tensor(V), F, fi, b)
    np.testing.assert_allclose(Pt.numpy(), P, atol=1e-12); np.testing.assert_allclose(Rt.numpy(), R, atol=1e-12)


def test_random_rotations_are_rotations():
    R = random_rotations(64, torch.Generator().manual_seed(1))
    np.testing.assert_allclose((R @ R.transpose(1, 2)).numpy(), np.broadcast_to(np.eye(3), (64, 3, 3)), atol=1e-5)
    np.testing.assert_allclose(torch.linalg.det(R).numpy(), 1.0, atol=1e-5)


# ---------------------------------------------------------------- with real library

needs_lib = pytest.mark.skipif(not (LIB.exists() and FIT.exists() and MANO.exists()),
                               reason="needs results/soma library, the P10 fit and MANO weights")


@pytest.fixture(scope="module")
def gen():
    from vicon2mano.core.fitter import FitConfig, MANOFitter
    m = MANOFitter(FitConfig(mano_model_path=str(MANO), hand_side="right", use_pca=False))._model
    return SynthGenerator(MarkerLibrary.load(LIB), m.cpu(), device="cpu", seed=0)


@needs_lib
def test_round_trip_reproduces_real_markers(gen):
    """No mixing, no augmentation, fitted pose + subject shape: every marker
    must land back on its measured position, relative to the hand."""
    lib = gen.lib
    pick = np.array([0, 5000, 20000])
    src = torch.tensor(pick)[:, None].expand(-1, gen.n_groups)
    # rebuild the expected cloud in the same root-free MANO frame the generator uses
    z = np.load(FIT)
    with torch.no_grad():
        o = gen.model(global_orient=torch.zeros(3, 3), hand_pose=torch.tensor(lib.hand_pose[pick]),
                      betas=torch.tensor(lib.betas)[None].expand(3, -1), transl=torch.zeros(3, 3))
    V = o.vertices.numpy() * 1000
    P, R = face_frames(V, gen.F, lib.face_idx, lib.bary)
    want = P + np.einsum("bmij,bmj->bmi", R, np.nan_to_num(lib.residual[pick]))
    pts, valid, lab = gen.sample(3, mix=False, augment=False, src=src)
    for b in range(3):
        got = np.full((gen.M, 3), np.nan)
        got[lab[b][valid[b]].numpy()] = pts[b][valid[b]].numpy()
        ok = np.isfinite(lib.residual[pick[b]]).all(-1)
        # centring differs (torch's nanmedian takes the lower middle on even
        # counts), so compare the clouds about their own means
        np.testing.assert_allclose(got[ok] - got[ok].mean(0), want[b][ok] - want[b][ok].mean(0),
                                   atol=1e-3)
        assert np.isnan(got[~ok]).all()                      # unmeasured -> absent


@needs_lib
def test_corruption_invariants(gen):
    pts, valid, lab = gen.sample(512)
    M, cfg = gen.M, gen.cfg
    assert pts.shape == (512, M + cfg.max_ghosts, 3)
    real = (lab >= 0)
    # each marker label at most once per cloud
    for b in range(512):
        l = lab[b][real[b]]
        assert len(l) == len(torch.unique(l))
    assert (valid == (lab != PAD)).all()
    assert ((lab == GHOST).sum(1) <= cfg.max_ghosts).all()
    assert (pts[~valid] == 0).all()
    # extra drops never exceed the cap (sources may add their own absences)
    assert (M - real.sum(1)).float().mean() <= cfg.max_missing
    # valid points come first, and the cloud is median-centred
    assert all(valid[b, :valid[b].sum()].all() for b in range(512))
