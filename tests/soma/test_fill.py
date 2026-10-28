import numpy as np
import torch

from vicon2mano.strategies.soma.fill import RIGID, Filler, align, frame_sampler, predict, train_filler

LABELS = list(RIGID) + ["Index2", "Index3", "Thumb2"]


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return torch.tensor(q * np.sign(np.linalg.det(q)), dtype=torch.float32)


def test_align_recovers_a_rigid_motion():
    T = torch.randn(7, 3) * 30
    X = (T @ _rot(0).mT + torch.tensor([5.0, -20.0, 100.0]))[None]
    R, cx, ct = align(X, torch.ones(1, 7), T)
    assert torch.allclose((X - cx) @ R.mT + ct, T[None], atol=1e-3)


def test_prediction_moves_with_the_hand():
    torch.manual_seed(0)
    f = Filler(LABELS, np.random.default_rng(1).normal(size=(7, 3)) * 30).eval()
    X = torch.randn(4, len(LABELS), 3) * 40
    vis = torch.ones(4, len(LABELS), dtype=torch.bool)
    vis[:, -2:] = False
    R, t = _rot(2), torch.tensor([100.0, 0.0, -50.0])
    with torch.no_grad():
        a, b = f(X, vis) @ R.mT + t, f(X @ R.mT + t, vis)
    assert torch.allclose(a, b, atol=1e-2)



def test_trains_on_real_frames_of_a_moving_hand():
    rng = np.random.default_rng(3)
    hand = rng.normal(size=(len(LABELS), 3)) * 30
    truth = np.stack([hand @ _rot(i).numpy().T + rng.normal(size=3) * 100 for i in range(64)]).astype(np.float32)
    X = truth.copy()
    X[rng.random(X.shape[:2]) < 0.1] = np.nan
    torch.manual_seed(0)
    f = train_filler(frame_sampler(X), LABELS, steps=300, batch=64, log=lambda s: None)
    Z = truth[:8].copy()
    Z[:, -1] = np.nan
    assert np.median(np.linalg.norm(predict(f, Z)[:, -1] - truth[:8, -1], axis=-1)) < 10
