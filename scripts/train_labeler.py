#!/usr/bin/env python3
"""Train the deep marker labeler (Han et al., SIGGRAPH 2018).

Sub-commands
------------
generate    Generate synthetic training data and write to HDF5.
train       Train MarkerLabeler on a previously generated dataset.

Examples
--------
# Step 1: generate 500 k synthetic frames
python scripts/train_labeler.py generate \\
    --mano-dir ../clean_kinematics/mano_v1_2/models \\
    --n-samples 500000 \\
    --out data/synth_labeler.h5

# Step 2: train the CNN
python scripts/train_labeler.py train \\
    --data data/synth_labeler.h5 \\
    --out-weights vicon2mano/weights/deep_labeler.pt \\
    --epochs 20 --batch-size 2 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the package is importable even when invoked from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Focal BCE loss
# ---------------------------------------------------------------------------

def focal_bce_loss(logits, targets, *, gamma: float = 2.0, alpha: float = 0.75):
    """Sigmoid focal binary cross-entropy.

    Args:
        logits:  (B, 21, res, res, res) raw network outputs.
        targets: (B, 21, res, res, res) Gaussian heatmap targets ∈ [0, 1].
        gamma:   Focusing exponent (suppresses easy negatives).
        alpha:   Weight for positive (foreground) voxels.

    Returns:
        Scalar mean loss.
    """
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    # BCE per-element
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # Focal weighting
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce
    return loss.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(args):
    import torch
    import torch.utils.data

    from vicon2mano.deep_labeler import MarkerLabeler

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={device}")

    # Datasets and loaders
    print("[train] loading dataset …", flush=True)
    if args.real_data:
        from vicon2mano.real_data import RealDataConfig, make_real_dataset
        cfg = RealDataConfig(val_frac=args.val_frac, side=args.side)
        print(f"[train] parsing CSV: {args.real_data}  side={args.side}", flush=True)
        train_ds = make_real_dataset(args.real_data, split="train", cfg=cfg)
        val_ds = make_real_dataset(args.real_data, split="val", cfg=cfg)
        print(f"[train] real data: {args.real_data}", flush=True)
    else:
        if not args.data:
            raise SystemExit("Error: provide --data <synth.h5> or --real-data <labeled.csv>")
        from vicon2mano.synth_data import make_synth_dataset
        train_ds = make_synth_dataset(args.data, split="train", val_frac=args.val_frac)
        val_ds = make_synth_dataset(args.data, split="val", val_frac=args.val_frac)
    if args.max_samples:
        from torch.utils.data import Subset
        total = min(args.max_samples, len(train_ds) + len(val_ds))
        n_val = max(1, int(total * args.val_frac))
        n_train = total - n_val
        train_ds = Subset(train_ds, range(min(n_train, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(n_val,   len(val_ds))))
        print(f"[train] capped to {len(train_ds)} train / {len(val_ds)} val samples", flush=True)
    else:
        print(f"[train] {len(train_ds)} train / {len(val_ds)} val samples", flush=True)

    print(f"[train] creating DataLoaders (num_workers={args.num_workers}) …", flush=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print("[train] DataLoaders ready", flush=True)

    # Model, optimiser, scheduler
    print("[train] building model …", flush=True)
    model = MarkerLabeler().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model parameters: {n_params:,}", flush=True)

    print("[train] creating optimiser and scheduler …", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * steps_per_epoch
    )

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[train] AMP={'on' if use_amp else 'off'}  steps_per_epoch={steps_per_epoch}", flush=True)

    out_path = Path(args.out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    log_every = max(1, steps_per_epoch // 10)  # ~10 log lines per epoch
    epoch_log: list[dict] = []
    log_path = out_path.with_name(out_path.stem + "_log.json")

    print(f"[train] starting training — {args.epochs} epochs, log every {log_every} steps", flush=True)

    for epoch in range(1, args.epochs + 1):
        # ---- Training ----
        print(f"[train] epoch {epoch}/{args.epochs} — fetching first batch …", flush=True)
        model.train()
        train_loss_sum = 0.0
        t0 = time.time()

        for step, (grids, heatmaps) in enumerate(train_loader, 1):
            if step == 1:
                print(f"[train] epoch {epoch}  first batch loaded — grid shape {tuple(grids.shape)}", flush=True)

            grids = grids.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            optimiser.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(grids)
                    loss = focal_bce_loss(logits, heatmaps)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(grids)
                loss = focal_bce_loss(logits, heatmaps)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            scheduler.step()
            train_loss_sum += loss.item()

            if step <= 3 or step % log_every == 0:
                avg = train_loss_sum / step
                lr_now = scheduler.get_last_lr()[0]
                elapsed_s = time.time() - t0
                sps = step / elapsed_s
                eta_s = (steps_per_epoch - step) / sps if sps > 0 else 0
                print(
                    f"  epoch {epoch:3d}  step {step:5d}/{steps_per_epoch}"
                    f"  loss={avg:.5f}  lr={lr_now:.2e}"
                    f"  {sps:.1f} steps/s  ETA {eta_s/60:.1f} min",
                    flush=True,
                )

        elapsed = time.time() - t0
        train_avg = train_loss_sum / steps_per_epoch

        # ---- Validation ----
        print(f"[train] epoch {epoch}  training done ({elapsed:.0f}s)  running validation …", flush=True)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for grids, heatmaps in val_loader:
                grids = grids.to(device, non_blocking=True)
                heatmaps = heatmaps.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(grids)
                        loss = focal_bce_loss(logits, heatmaps)
                else:
                    logits = model(grids)
                    loss = focal_bce_loss(logits, heatmaps)
                val_loss_sum += loss.item()

        val_avg = val_loss_sum / max(1, len(val_loader))
        improved = val_avg < best_val_loss

        print(
            f"epoch {epoch:3d}  train={train_avg:.5f}  val={val_avg:.5f}"
            f"  {'*SAVED*' if improved else '      '}  {elapsed:.0f}s",
            flush=True,
        )

        epoch_log.append({
            "epoch": epoch,
            "train_loss": train_avg,
            "val_loss": val_avg,
            "lr": scheduler.get_last_lr()[0],
        })
        with open(log_path, "w") as _f:
            json.dump(epoch_log, _f, indent=2)

        if improved:
            best_val_loss = val_avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_avg,
                    "args": vars(args),
                },
                out_path,
            )

    print(f"\n[train] done.  Best val loss: {best_val_loss:.5f}  Weights → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate(args):
    from vicon2mano.synth_data import SynthConfig, generate_dataset

    cfg = SynthConfig(
        mano_model_path=args.mano_dir,
        hand_side=args.side,
        n_samples=args.n_samples,
        output_path=args.out,
        seed=args.seed,
        mano_batch_size=args.mano_batch_size,
    )
    generate_dataset(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_labeler",
        description="Generate training data and/or train the deep marker labeler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    gen = sub.add_parser("generate", help="Generate synthetic HDF5 dataset.")
    gen.add_argument("--mano-dir", default="../clean_kinematics/mano_v1_2/models",
                     help="Path to directory containing MANO_RIGHT.pkl / MANO_LEFT.pkl")
    gen.add_argument("--side", default="right", choices=["right", "left"])
    gen.add_argument("--n-samples", type=int, default=500_000)
    gen.add_argument("--out", default="data/synth_labeler.h5")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--mano-batch-size", type=int, default=512)

    # ---- train ----
    trn = sub.add_parser("train", help="Train MarkerLabeler on a generated dataset.")
    trn.add_argument("--data", default=None,
                     help="Path to synth_labeler.h5 generated by the 'generate' command.")
    trn.add_argument("--real-data", default=None,
                     help="Path to a labeled Vicon CSV (frame,marker,x_mm,y_mm,z_mm). "
                          "Use instead of --data to train on real recordings.")
    trn.add_argument("--out-weights", default="vicon2mano/weights/deep_labeler.pt")
    trn.add_argument("--epochs", type=int, default=20)
    trn.add_argument("--batch-size", type=int, default=2)
    trn.add_argument("--lr", type=float, default=3e-4)
    trn.add_argument("--device", default=None,
                     help="'cuda', 'cpu', or 'cuda:N'.  Auto-detected if omitted.")
    trn.add_argument("--num-workers", type=int, default=2)
    trn.add_argument("--val-frac", type=float, default=0.02)
    trn.add_argument("--max-samples", type=int, default=None,
                     help="Cap training+val to this many frames (useful for quick smoke-tests).")
    trn.add_argument("--side", default="left", choices=["left", "right"],
                     help="Which hand to load from wide-format CSVs (default: left).")
    trn.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                     help="Disable automatic mixed precision (AMP).")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        _generate(args)
    elif args.command == "train":
        _train(args)


if __name__ == "__main__":
    main()
