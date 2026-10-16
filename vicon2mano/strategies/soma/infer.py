"""Run a trained SOMA labeller over a whole raw trial.

Per frame: the unlabelled cloud -> one-to-one labels with confidences. Per
piece (raw column cut at gaps / >5 mm jumps): summed-confidence vote, clashes
resolved (``realdata.vote_pieces``). Labels are indices into the model's
label list, -1 = unlabelled; arrays are (T, C) over the raw file's columns.
"""

from __future__ import annotations

import numpy as np
import torch

from vicon2mano.strategies.soma.model import SomaLabeler, hard_assign
from vicon2mano.strategies.soma.realdata import clouds, pieces, vote_pieces


def load_model(path, device: str):
    ck = torch.load(path, map_location=device)
    net = SomaLabeler(len(ck["labels"]), d_model=ck["d_model"], n_layers=ck["layers"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, list(ck["labels"]), ck["n_points"]


def label_trial(net, raw: np.ndarray, n_points: int, device: str, batch: int = 2048):
    """Returns per_frame (T, C), conf (T, C), voted (T, C), piece ids (T, C)."""
    T, C = raw.shape[:2]
    pres = np.isfinite(raw).all(-1)
    n_pres = pres.sum(1)
    frames = np.flatnonzero((n_pres > 0) & (n_pres <= n_points))
    pts, valid, _, col = clouds(raw, np.where(pres, 0, -2), frames, n_points)
    lab = np.full((T, C), -1)
    conf = np.zeros((T, C))
    with torch.no_grad():
        for s in range(0, len(frames), batch):
            lp = net(torch.tensor(pts[s:s + batch], device=device),
                     torch.tensor(valid[s:s + batch], device=device)).float().cpu().numpy()
            for b in range(lp.shape[0]):
                k = s + b
                nv = int(valid[k].sum())
                h = hard_assign(lp[b], nv)
                c = col[k, :nv]
                lab[frames[k], c] = h
                conf[frames[k], c] = np.where(h >= 0, np.exp(lp[b, np.arange(nv), np.maximum(h, 0)]), 0)
    pid = pieces(raw)
    return lab, conf, vote_pieces(lab, conf, pid, lp.shape[2] - 1 if len(frames) else 0), pid
