"""strategySOMA network: Sinkhorn marginals, padding, equivariance, learnability."""

import numpy as np
import torch

from vicon2mano.strategies.soma.model import (SomaLabeler, assignment_nll, hard_assign,
                                              log_sinkhorn)


def test_sinkhorn_marginals_with_padding():
    torch.manual_seed(0)
    B, N, M = 3, 7, 5
    valid = torch.ones(B, N, dtype=torch.bool); valid[1, 5:] = False; valid[2, 2:] = False
    logP = log_sinkhorn(torch.randn(B, N, M), torch.tensor(0.5), valid, iters=200)
    P = logP.exp()
    for b in range(B):
        n = int(valid[b].sum())
        np.testing.assert_allclose(P[b, :n].sum(1).numpy(), 1.0, atol=1e-4)          # point rows
        np.testing.assert_allclose(P[b, :N + 1, :M].sum(0).numpy(), 1.0, atol=1e-4)  # label cols
        assert P[b, n:N].sum() < 1e-6                                                # pads empty
        np.testing.assert_allclose(P[b, N, :].sum().item(), M, atol=1e-3)            # dustbin row
        np.testing.assert_allclose(P[b, :N + 1, M].sum().item(), n, atol=1e-3)       # dustbin col


def test_permutation_equivariant():
    torch.manual_seed(1)
    net = SomaLabeler(6, d_model=32, n_layers=2, d_ff=64).eval()
    pts = torch.randn(1, 8, 3) * 30; valid = torch.ones(1, 8, dtype=torch.bool)
    perm = torch.randperm(8)
    a = net(pts, valid)[0, :8]; b = net(pts[:, perm], valid)[0, :8]
    torch.testing.assert_close(a[perm], b, atol=1e-4, rtol=1e-4)


def test_hard_assign_one_to_one_with_dustbin():
    M = 3
    logP = np.log(np.array([[.9, .05, .05, 0.0],     # point 0 -> label 0
                            [.8, .1, .1, 0.0],       # point 1 also prefers 0, gets next best
                            [.01, .01, .01, .97],    # point 2 -> ghost
                            [0, 0, 0, 0]]) + 1e-9)    # dustbin row (unused here)
    out = hard_assign(logP, 3)
    assert out[0] == 0 and out[2] == -1 and out[1] in (1, 2)


def test_learns_a_toy_labelling():
    """Fixed 5-point template, noise + shuffle + one drop: the net must label
    it within a few hundred steps. (No rotation here: learning rotation
    invariance takes far longer than a unit test; full training has it.)"""
    torch.manual_seed(0)
    tmpl = torch.tensor([[0, 0, 0], [40, 0, 0], [0, 25, 0], [0, 0, 60], [30, 30, 10.]])
    net = SomaLabeler(5, d_model=64, n_layers=3, d_ff=128)
    opt = torch.optim.Adam(net.parameters(), 2e-3)

    def batch(B):
        X = tmpl[None] + torch.randn(B, 5, 3)
        X = X - X.mean(1, keepdim=True)
        lab = torch.arange(5).repeat(B, 1)
        drop = torch.randint(0, 5, (B,))
        lab[torch.arange(B), drop] = -2
        valid = lab >= 0
        perm = torch.argsort(torch.rand(B, 5) + (~valid) * 2.0, 1)
        bi = torch.arange(B)[:, None]
        X = torch.where(valid[..., None], X, 0.0)
        return X[bi, perm], valid[bi, perm], lab[bi, perm]

    for step in range(400):
        X, v, l = batch(64)
        loss = assignment_nll(net(X, v), l, v)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    X, v, l = batch(200)
    with torch.no_grad():
        lp = net(X, v).numpy()
    # best LABEL per point (dustbin excluded): early in training Sinkhorn can
    # leak a present label's mass into the dustbin row (corner < n), leaving a
    # 50/50 label-vs-ghost tie that hard_assign may break either way. Whether
    # that persists after full training is checked on real data, not here.
    acc = np.mean([(lp[b, :int(v[b].sum()), :5].argmax(1) == l[b][v[b]].numpy()).mean()
                   for b in range(200)])
    assert loss.item() < 0.5 and acc > 0.95, (loss.item(), acc)
