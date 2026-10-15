"""SOMA-style labelling network: self-attention over the cloud + Sinkhorn.

Per frame, the unordered cloud (B, N, 3) mm, median-centred, goes through a
point-wise embedding and ``n_layers`` transformer encoder layers, so every
point is described in the context of the whole hand. A linear head scores
each point against the M labels. Those scores, bordered by one learnable
"dustbin" score -- a point may be a ghost, a label may be absent -- go
through log-space Sinkhorn normalisation (SuperGlue, Sarlin et al. 2020; as
used by SOMA), giving a soft assignment whose rows and columns each sum to
one, with the dustbin row/column absorbing the rest.

Training loss is the negative log-likelihood of the true assignment:
each real point to its label, each ghost to the dustbin column, each absent
label to the dustbin row.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

SCALE_MM = 100.0      # inputs are divided by this: a hand spans ~2 units


class SomaLabeler(nn.Module):
    def __init__(self, n_labels: int, d_model: int = 128, n_layers: int = 8, n_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.0, sinkhorn_iters: int = 30):
        super().__init__()
        self.M, self.iters = n_labels, sinkhorn_iters
        self.embed = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_labels))
        self.dustbin = nn.Parameter(torch.tensor(1.0))

    def forward(self, pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """pts (B, N, 3) mm, valid (B, N) bool -> log assignment (B, N+1, M+1)."""
        h = self.embed(pts / SCALE_MM)
        h = self.encoder(h, src_key_padding_mask=~valid)
        scores = self.head(h)                                           # (B, N, M)
        return log_sinkhorn(scores, self.dustbin, valid, self.iters)


def log_sinkhorn(scores: torch.Tensor, alpha: torch.Tensor, valid: torch.Tensor,
                 iters: int) -> torch.Tensor:
    """Log-space Sinkhorn with a dustbin row and column, padding-aware.

    Row marginals: 1 per valid point, M for the dustbin row (it can absorb
    every label); columns: 1 per label, n_valid for the dustbin column.
    Returned log-probabilities are rescaled so a valid row sums to 1.
    """
    B, N, M = scores.shape
    n = valid.sum(1).float()                                            # (B,)
    Z = torch.cat([scores, alpha.expand(B, N, 1)], 2)
    Z = torch.cat([Z, alpha.expand(B, 1, M + 1)], 1)                    # (B, N+1, M+1)
    neg = torch.finfo(Z.dtype).min / 4
    rowmask = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=Z.device)], 1)
    Z = Z.masked_fill(~rowmask[:, :, None], neg)
    norm = -(n + M).log()                                               # (B,)
    log_mu = torch.cat([norm[:, None].expand(B, N), math.log(M) + norm[:, None]], 1)
    log_mu = log_mu.masked_fill(~rowmask, neg)
    log_nu = torch.cat([norm[:, None].expand(B, M), n.log()[:, None] + norm[:, None]], 1)
    u = torch.zeros_like(log_mu); v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v[:, None, :], 2)
        v = log_nu - torch.logsumexp(Z + u[:, :, None], 1)
    return Z + u[:, :, None] + v[:, None, :] - norm[:, None, None]


def assignment_nll(logP: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean NLL of the true assignment. label (B, N): marker index, -1 ghost, -2 pad."""
    B, N1, M1 = logP.shape
    N, M = N1 - 1, M1 - 1
    bi, pi = torch.nonzero(valid, as_tuple=True)
    tgt = label[bi, pi].clone()
    tgt[tgt < 0] = M                                                    # ghost -> dustbin column
    loss_pts = -logP[bi, pi, tgt]
    present = torch.zeros(B, M, dtype=torch.bool, device=logP.device)
    lb = label.clamp(min=0)
    present.scatter_(1, lb, (label >= 0))
    loss_abs = -logP[:, N, :M][~present]                                # absent label -> dustbin row
    return (loss_pts.sum() + loss_abs.sum()) / (len(loss_pts) + len(loss_abs))


def hard_assign(logP: np.ndarray, n_valid: int) -> np.ndarray:
    """One-to-one labels from a (N+1, M+1) log assignment for one cloud.

    Each point may take one label or stay unlabelled (its own dustbin), each
    label is used at most once. Returns (n_valid,) label index or -1.
    """
    M = logP.shape[1] - 1
    L = logP[:n_valid, :M]
    dust = logP[:n_valid, M]
    big = 1e9
    cost = np.full((n_valid, M + n_valid), big)
    cost[:, :M] = -L
    cost[np.arange(n_valid), M + np.arange(n_valid)] = -dust
    r, c = linear_sum_assignment(cost)
    out = np.full(n_valid, -1)
    keep = c < M
    out[r[keep]] = c[keep]
    return out
