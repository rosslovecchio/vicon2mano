"""Learned marker completion: an MLP trained on synthetic clouds predicts every marker from the visible ones.

Each cloud is first aligned (weighted Kabsch) to a template on the near-rigid palm and knuckle markers it shows,
so the network only sees hand-frame coordinates. Synthetic training clouds come from ``SynthGenerator.sample(complete=True)``:
real library poses on the participant's calibrated shape, with markers hidden at random; the loss is on the hidden
markers only, so the network learns how markers move together (PIP/DIP coupling, neighbouring fingers) rather than
a rigid offset. ``frame_sampler`` trains on real labelled recordings the same way instead.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

RIGID = ("Palm1", "Palm2", "Palm3", "Index1", "Middle1", "Ring1", "Pinky1")


def align(X: torch.Tensor, w: torch.Tensor, T: torch.Tensor):
    """Weighted Kabsch of X (B, K, 3) onto template T (K, 3) with weights w (B, K): (R, cx, ct), canonical = (X - cx) @ R.mT + ct."""
    ws = w.sum(1, keepdim=True).clamp_min(1e-9)[..., None]
    cx = (w[..., None] * X).sum(1, keepdim=True) / ws
    ct = (w[..., None] * T[None]).sum(1, keepdim=True) / ws
    H = ((X - cx) * w[..., None]).mT @ (T[None] - ct)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.mT @ U.mT))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return Vt.mT @ D @ U.mT, cx, ct


class Filler(nn.Module):
    def __init__(self, labels: list[str], template: np.ndarray, width: int = 512, depth: int = 4):
        super().__init__()
        self.labels = list(labels)
        self.ridx = [self.labels.index(n) for n in RIGID if n in self.labels]
        self.register_buffer("template", torch.as_tensor(template, dtype=torch.float32))
        M = len(labels)
        layers, d = [], 4 * M
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 3 * M))

    def forward(self, X: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """X (B, M, 3) mm (anything where not ``vis``), vis (B, M) bool -> every marker (B, M, 3) mm, same frame as X."""
        w = vis[:, self.ridx].float()
        R, cx, ct = align(X[:, self.ridx], w, self.template)
        Y = torch.where(vis[..., None], (X - cx) @ R.mT + ct, 0.0)
        out = self.net(torch.cat([Y.flatten(1) / 100, vis.float()], 1)).view_as(X) * 100
        return (out - ct) @ R + cx


def frame_sampler(X: np.ndarray, device: str = "cpu"):
    """A ``train_filler`` source drawing random frames of real recordings X (N, M, 3) mm, NaN = unobserved."""
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    ok = torch.isfinite(X).all(-1)
    X = torch.nan_to_num(X)
    return lambda B: (lambda i: (X[i], ok[i]))(torch.randint(len(X), (B,), device=device))


def train_filler(sample, labels: list[str], steps: int = 20000, batch: int = 1024, lr: float = 1e-3,
                 log=print) -> Filler:
    """Train a Filler on clouds from ``sample(B) -> (X (B, M, 3) mm, present (B, M) bool)`` in ``labels`` order,
    hiding each marker with a per-cloud probability up to 0.5."""
    X, ok = sample(8192)
    dev = X.device
    ri = [labels.index(n) for n in RIGID if n in labels]
    w = ok[:, ri].float()
    tmpl = X[w.sum(1).argmax(), ri]
    for _ in range(3):                                   # generalised Procrustes: align to the template, average
        R, cx, ct = align(X[:, ri], w, tmpl)
        Y = (X[:, ri] - cx) @ R.mT + ct
        tmpl = (w[..., None] * Y).sum(0) / w.sum(0)[:, None]
        tmpl = tmpl - tmpl.mean(0)
    f = Filler(labels, tmpl.cpu().numpy()).to(dev)
    opt = torch.optim.AdamW(f.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for s in range(1, steps + 1):
        X, ok = sample(batch)
        p = 0.5 * torch.rand(batch, 1, device=dev)
        vis = ok & (torch.rand(ok.shape, device=dev) >= p)
        enough = vis[:, ri].sum(1) >= 3
        err = (f(X, vis) - X).norm(dim=-1)
        loss = err[ok & ~vis & enough[:, None]].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 1000 == 0 or s == steps:
            log(f"step {s:6d}  hidden-marker error mean {loss.item():.2f} mm")
    return f.eval()


@torch.no_grad()
def predict(f: Filler, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Every marker (N, M, 3) mm from the observed ones in X (N, M, 3) mm, NaN = unobserved; NaN rows where fewer
    than 3 rigid markers are seen."""
    dev = f.template.device
    out = np.full(X.shape, np.nan, np.float32)
    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=dev)
        vis = torch.isfinite(x).all(-1)
        p = f(torch.nan_to_num(x), vis).cpu().numpy()
        p[vis[:, f.ridx].sum(1).cpu().numpy() < 3] = np.nan
        out[s:s + batch] = p
    return out


def save(f: Filler, path) -> None:
    torch.save(dict(labels=f.labels, template=f.template.cpu(), state=f.state_dict()), path)


def load(path, device: str = "cpu") -> Filler:
    z = torch.load(path, map_location=device)
    f = Filler(z["labels"], z["template"].cpu().numpy())
    f.load_state_dict(z["state"])
    return f.to(device).eval()
