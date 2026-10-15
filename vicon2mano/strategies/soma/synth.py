"""Synthetic marker clouds for training the labeller, generated on the fly.

One sample = one frame's raw, unlabelled cloud, built like this:

1. **Pose by finger mixing.** In a ``p_mix`` share of clouds each finger's 9
   pose components come from an independently drawn library frame, so the
   library yields combinations no single recording contains while every
   finger on its own stays a pose a real hand made; palm and forearm follow a
   separate "root" draw. The rest keep one frame for every finger: mixing
   breaks the real coupling between neighbouring fingers, and on its own it
   made converging fingertips (the confusable case) too rare -- nearest-
   neighbour p10 23-27mm synthetic vs 15-19mm real. Small Gaussian pose noise
   on top either way.
2. **Shape.** ``betas`` from the source subject plus noise, or from the MANO
   prior, so one model can serve participants with different hands.
3. **Markers.** Attachment point on the posed mesh + that marker's measured
   residual from the frame its finger was drawn from, re-expressed in the
   new mesh's face frame (see ``library.py``), plus jitter.
4. **Corruption.** Markers whose residual is NaN are absent (a real
   occlusion pattern). Extra markers are dropped, weighted by each marker's
   real miss rate, up to ``max_missing`` absent in total. Ghost points are
   added near the hand.
5. **Normalisation.** Random global rotation, median-centred, shuffled.

Output, all on ``device``:
    points (B, N, 3) float32 mm, padded with zeros
    valid  (B, N)    bool
    label  (B, N)    int64: marker index, GHOST (-1) or PAD (-2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vicon2mano.strategies.soma.library import (FINGER_BLOCKS, MarkerLibrary, face_frames,
                                                marker_group)

GHOST, PAD = -1, -2


@dataclass
class SynthConfig:
    pose_noise_rad: float = 0.04      # per pose component (0.08 made converging tips too rare)
    p_mix: float = 0.2                # share of clouds with fingers from different frames;
                                      # the rest keep one frame's real finger coupling
                                      # (mixing alone under-produces converging tips)
    p_subject_shape: float = 0.5      # else betas ~ N(0, prior_sd)
    subject_shape_sd: float = 0.3
    prior_shape_sd: float = 1.0
    jitter_mm: float = 1.0            # on top of the measured residual
    max_missing: int = 4              # absent markers per frame, total
    max_ghosts: int = 2
    ghost_spread_mm: float = 25.0     # ghosts ~ N(random real marker, spread)
    full_rotation: bool = True        # uniform SO(3); False = about vertical only


class SynthGenerator:
    def __init__(self, lib: MarkerLibrary, model, cfg: SynthConfig | None = None,
                 device: str = "cpu", seed: int = 0):
        self.lib, self.cfg, self.dev = lib, cfg or SynthConfig(), device
        self.model = model.to(device)
        self.F = np.asarray(model.faces, dtype=np.int64)
        self.M = len(lib.labels)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.hp = torch.tensor(lib.hand_pose, device=device)
        self.res = torch.tensor(lib.residual, device=device)              # (L, M, 3)
        self.betas0 = torch.tensor(lib.betas, device=device)
        self.face_idx, self.bary = lib.face_idx, lib.bary
        groups = ["Root"] + list(FINGER_BLOCKS)
        self.group_of = torch.tensor([groups.index(marker_group(n)) for n in lib.labels],
                                     device=device)
        self.n_groups = len(groups)
        w = torch.tensor(lib.miss_rate, dtype=torch.float32, device=device) + 0.01
        self.drop_w = w / w.sum()

    @property
    def n_points(self) -> int:
        return self.M + self.cfg.max_ghosts

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g).to(self.dev)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g).to(self.dev)

    def _randint(self, hi, *shape):
        return torch.randint(0, hi, shape, generator=self.g).to(self.dev)

    def sample(self, B: int, *, mix: bool = True, augment: bool = True,
               src: torch.Tensor | None = None):
        """Generate B clouds. ``mix=False, augment=False`` with ``src`` given
        reproduces library frames exactly (round-trip test)."""
        cfg, L, M = self.cfg, self.hp.shape[0], self.M
        if src is None:
            src = self._randint(L, B, self.n_groups)
            same = self._rand(B) >= (cfg.p_mix if mix else 0.0)
            src = torch.where(same[:, None], src[:, :1].expand(-1, self.n_groups), src)
        src = src.to(self.dev)
        # 1. pose: finger blocks from their own source frames
        hp = self.hp[src[:, 0]].clone()
        for gi, (f, (a, b)) in enumerate(FINGER_BLOCKS.items(), start=1):
            hp[:, a:b] = self.hp[src[:, gi], a:b]
        if augment:
            hp = hp + cfg.pose_noise_rad * self._randn(B, 45)
        # 2. shape
        if augment:
            subj = self._rand(B, 1) < cfg.p_subject_shape
            betas = torch.where(subj, self.betas0 + cfg.subject_shape_sd * self._randn(B, 10),
                                cfg.prior_shape_sd * self._randn(B, 10))
        else:
            betas = self.betas0.expand(B, -1)
        with torch.no_grad():
            out = self.model(global_orient=torch.zeros(B, 3, device=self.dev), hand_pose=hp,
                             betas=betas, transl=torch.zeros(B, 3, device=self.dev))
        V = out.vertices * 1000.0
        # 3. markers = attachment + face-frame residual from each marker's source frame
        P, R = face_frames(V, self.F, self.face_idx, self.bary)          # (B,M,3), (B,M,3,3)
        rsrc = src.gather(1, self.group_of[None].expand(B, -1))          # (B, M)
        res = self.res[rsrc, torch.arange(M, device=self.dev)[None]]     # (B, M, 3)
        X = P + torch.einsum("bmij,bmj->bmi", R, torch.nan_to_num(res))
        present = torch.isfinite(res).all(-1)
        if augment:
            X = X + cfg.jitter_mm * self._randn(B, M, 3)
            # 4. extra drops, weighted by real miss rates, capped in total
            k_extra = self._randint(cfg.max_missing + 1, B)
            keys = -torch.log(self._rand(B, M)) / self.drop_w            # exponential race
            keys = torch.where(present, keys, torch.full_like(keys, -1.0))
            order = keys.argsort(1)                                       # absent first
            # top up the source's own absences to k_extra; never resurrect a
            # marker with no measured residual (it would sit on the bare mesh)
            n_abs = (~present).sum(1)
            want = torch.maximum(n_abs, k_extra)
            rank = order.argsort(1)
            present = present & (rank >= want[:, None])
        # ghosts near random present markers
        G = cfg.max_ghosts
        n_gh = self._randint(G + 1, B) if (augment and G) else torch.zeros(B, dtype=torch.long,
                                                                         device=self.dev)
        anchor = X[torch.arange(B, device=self.dev)[:, None], self._randint(M, B, max(G, 1))]
        ghosts = anchor + cfg.ghost_spread_mm * self._randn(B, max(G, 1), 3)
        # assemble
        N = M + G
        pts = torch.zeros(B, N, 3, device=self.dev)
        lab = torch.full((B, N), PAD, dtype=torch.long, device=self.dev)
        pts[:, :M] = X
        lab[:, :M] = torch.where(present, torch.arange(M, device=self.dev)[None], PAD)
        if G:
            gm = torch.arange(G, device=self.dev)[None] < n_gh[:, None]
            pts[:, M:] = ghosts[:, :G]
            lab[:, M:] = torch.where(gm, GHOST, PAD)
        valid = lab != PAD
        # 5. rotate, median-centre, shuffle
        if augment:
            pts = pts @ random_rotations(B, self.g, cfg.full_rotation).to(self.dev).transpose(1, 2)
        med = torch.stack([torch.nanmedian(torch.where(valid[..., None], pts, torch.nan)[..., d], 1)
                           .values for d in range(3)], -1)
        pts = torch.where(valid[..., None], pts - med[:, None], torch.zeros_like(pts))
        perm = torch.argsort(self._rand(B, N) + (~valid) * 2.0, 1)       # valid points first
        bi = torch.arange(B, device=self.dev)[:, None]
        return pts[bi, perm], valid[bi, perm], lab[bi, perm]


def random_rotations(B: int, g: torch.Generator, full: bool = True) -> torch.Tensor:
    """(B, 3, 3) uniform rotations (Shoemake quaternions), or about z only."""
    if not full:
        th = 2 * np.pi * torch.rand(B, generator=g)
        c, s = th.cos(), th.sin()
        R = torch.zeros(B, 3, 3); R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s
        R[:, 1, 1] = c; R[:, 2, 2] = 1
        return R
    u1, u2, u3 = torch.rand(3, B, generator=g)
    q = torch.stack([(1 - u1).sqrt() * (2 * np.pi * u2).sin(), (1 - u1).sqrt() * (2 * np.pi * u2).cos(),
                     u1.sqrt() * (2 * np.pi * u3).sin(), u1.sqrt() * (2 * np.pi * u3).cos()], 1)
    x, y, z, w = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).view(B, 3, 3)
