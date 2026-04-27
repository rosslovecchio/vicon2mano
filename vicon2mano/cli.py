"""Command-line entry point: vicon2mano <input> <output> [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vicon2mano",
        description="Fit MANO hand model to Vicon 3D marker data.",
    )
    parser.add_argument("input", help=".c3d or .csv Vicon file")
    parser.add_argument("output", help="output .npz path")
    parser.add_argument("--mano-dir", default="data/mano", help="MANO model directory")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--no-pca", action="store_true", help="optimise full 45-dim pose")
    parser.add_argument("--n-pca", type=int, default=6)
    parser.add_argument("--iters-shape", type=int, default=100)
    parser.add_argument("--iters-pose", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--obj-dir", default=None, help="also export per-frame .obj meshes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from .fitter import FitConfig, MANOFitter
    from .export import save_npz, save_obj_sequence

    cfg = FitConfig(
        mano_model_path=args.mano_dir,
        hand_side=args.side,
        use_pca=not args.no_pca,
        n_pca_comps=args.n_pca,
        n_iters_shape=args.iters_shape,
        n_iters_pose=args.iters_pose,
        lr=args.lr,
    )

    ext = Path(args.input).suffix.lower()
    if ext == ".c3d":
        from .loader import load_c3d
        markers, labels, fps = load_c3d(args.input)
        print(f"Loaded {markers.shape[0]} frames, {markers.shape[1]} markers @ {fps} Hz")
    elif ext == ".csv":
        from .loader import load_csv
        markers, labels = load_csv(args.input)
        print(f"Loaded {markers.shape[0]} frames, {markers.shape[1]} markers")
    else:
        sys.exit(f"Unsupported input format: {ext}")

    fitter = MANOFitter(cfg)
    result = fitter.fit(markers, labels, verbose=not args.quiet)

    save_npz(result, args.output)
    print(f"Saved parameters to {args.output}")

    if args.obj_dir:
        save_obj_sequence(result, args.obj_dir)
        print(f"Saved mesh sequence to {args.obj_dir}/")


if __name__ == "__main__":
    main()
