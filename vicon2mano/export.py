"""Export FitResult to common formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fitter import FitResult


def save_npz(result: FitResult, path: str) -> None:
    """Save all parameters and joints to a compressed .npz file."""
    np.savez_compressed(
        path,
        global_orient=result.global_orient,
        hand_pose=result.hand_pose,
        transl=result.transl,
        betas=result.betas,
        joints=result.joints,
        assignment=result.assignment,
    )


def save_obj_sequence(result: FitResult, out_dir: str, *, every: int = 1) -> None:
    """Write one .obj mesh file per frame (or every N frames) to out_dir."""
    import trimesh

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    T = result.vertices.shape[0]
    for t in range(0, T, every):
        mesh = trimesh.Trimesh(
            vertices=result.vertices[t],
            process=False,
        )
        mesh.export(str(out_path / f"frame_{t:06d}.obj"))


def to_dict(result: FitResult) -> dict:
    """Convert to a plain dict (e.g. for JSON serialisation)."""
    return {
        "global_orient": result.global_orient.tolist(),
        "hand_pose": result.hand_pose.tolist(),
        "transl": result.transl.tolist(),
        "betas": result.betas.tolist(),
        "joints": result.joints.tolist(),
        "assignment": result.assignment.tolist(),
    }
