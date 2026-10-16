"""strategySOMA static calibration: fit recovers a synthetic static, flags a
mislabelled marker, and re-centres residuals as specified."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vicon2mano.strategies.soma.calibrate import calibrated_library, fit_static, static_cloud
from vicon2mano.strategies.soma.library import MarkerLibrary, face_frames

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "results/soma/libraries/library_P10_Trial2.npz"
MANO = REPO / "../clean_kinematics/mano_v1_2/models"
needs_lib = pytest.mark.skipif(not (LIB.exists() and MANO.exists()),
                               reason="needs results/soma library and MANO weights")


def test_static_cloud_picks_still_complete_window():
    rng = np.random.default_rng(0)
    T, M = 600, 4
    X = rng.normal(size=(T, M, 3)) * 20                  # moving
    X[300:500] = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10.]]) + rng.normal(size=(200, M, 3)) * 0.05
    X[100:150, 2] = np.nan
    cloud, info = static_cloud(X, n=150)
    assert 300 <= info["start"] <= 350
    np.testing.assert_allclose(cloud, [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], atol=0.05)


def test_calibrated_library_recentres_only_calibrated_markers():
    rng = np.random.default_rng(1)
    N, M = 50, 3
    lib = MarkerLibrary(labels=["A", "B", "C"], face_idx=np.zeros(M, int), bary=np.ones((M, 3)) / 3,
                        hand_pose=np.zeros((N, 45), np.float32), betas=np.zeros(10, np.float32),
                        residual=rng.normal(size=(N, M, 3)).astype(np.float32), frames=np.arange(N),
                        miss_rate=np.zeros(M))
    r_s = np.array([[5, 0, 0], [np.nan, np.nan, np.nan], [0, 0, -2.]])
    out = calibrated_library(lib, np.ones(10), r_s)
    med0, med1 = np.nanmedian(lib.residual, 0), np.nanmedian(out.residual, 0)
    np.testing.assert_allclose(med1[0], [5, 0, 0], atol=1e-5)
    np.testing.assert_allclose(med1[2], [0, 0, -2], atol=1e-5)
    np.testing.assert_allclose(out.residual[:, 1], lib.residual[:, 1])      # uncalibrated: untouched
    # dynamics kept: deviations from the median unchanged
    np.testing.assert_allclose(out.residual[:, 0] - med1[0], lib.residual[:, 0] - med0[0], atol=1e-5)
    assert (out.betas == 1).all()


@pytest.fixture(scope="module")
def setup():
    from vicon2mano.core.fitter import FitConfig, MANOFitter
    model = MANOFitter(FitConfig(mano_model_path=str(MANO), hand_side="right", use_pca=False))._model.cpu()
    lib = MarkerLibrary.load(LIB)
    k = int(np.argmin(np.abs(lib.hand_pose).sum(1)))                    # flattest library pose
    betas = torch.tensor([1.0, -0.5] + [0.0] * 8)
    with torch.no_grad():
        o = model(betas=betas[None], hand_pose=torch.tensor(lib.hand_pose[k])[None],
                  global_orient=torch.tensor([[0.3, -1.2, 0.4]]), transl=torch.tensor([[0.1, -0.2, 0.9]]))
    V = o.vertices[0].numpy() * 1000
    P, R = face_frames(V, np.asarray(model.faces, np.int64), lib.face_idx, lib.bary)
    cloud = P + np.einsum("mij,mj->mi", R, np.nanmedian(lib.residual, 0))
    return model, lib, cloud


@needs_lib
def test_fit_recovers_a_synthetic_static(setup):
    model, lib, cloud = setup
    out = fit_static(model, lib, cloud, iters_root=200, iters=800)
    hand = [i for i, n in enumerate(lib.labels) if not n.startswith("Forearm")]
    assert np.nanmax(out["err_mm"][hand]) < 4.0, out["err_mm"][hand]
    assert out["suspects"] == []
    assert np.isnan(out["r_s"][[i for i, n in enumerate(lib.labels) if n.startswith("Forearm")]]).all()


@needs_lib
def test_mislabelled_static_marker_is_flagged(setup):
    """Palm3 and Index2 swapped: no finger bend can explain a palm marker on
    a finger, so one of them must be flagged and left uncalibrated. (A
    fingertip-to-fingertip swap is NOT detectable this way -- a finger can
    bend to reach it -- which is why statics are also screened by segment
    lengths.)"""
    model, lib, cloud = setup
    c = cloud.copy()
    a, b = lib.labels.index("Palm3"), lib.labels.index("Index2")
    c[[a, b]] = c[[b, a]]
    out = fit_static(model, lib, c, iters_root=200, iters=800)
    assert {"Palm3", "Index2"} & set(out["suspects"]), out["suspects"]
    flagged = [lib.labels.index(s) for s in out["suspects"]]
    assert np.isnan(out["r_s"][flagged]).all()
