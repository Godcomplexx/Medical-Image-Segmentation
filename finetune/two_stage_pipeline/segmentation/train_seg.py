"""Train the 2.5D U-Net lesion segmentation detector.

Best checkpoint is selected on validation **lesion recall** (fraction of ground-truth
lesion components that receive at least one predicted pixel) — the quantity that
gates the end-to-end target — not pixel Dice.

    python -m finetune.two_stage_pipeline.segmentation.train_seg
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LesionSegDataset
from .losses import combo_loss, dice_score
from .unet import UNet


def _lesion_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """(matched, missed) ground-truth lesion components for one slice."""
    count, labels = cv2.connectedComponents(gt.astype(np.uint8))
    matched = missed = 0
    for component in range(1, count):
        if pred[labels == component].any():
            matched += 1
        else:
            missed += 1
    return matched, missed


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    dice_total = 0.0
    batches = 0
    tp = fn = 0
    for images, masks in loader:
        images = images.to(device)
        masks_device = masks.to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
        dice_total += float(dice_score(logits, masks_device))
        batches += 1
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        gt = masks.numpy().astype(np.uint8)
        for sample in range(pred.shape[0]):
            matched, missed = _lesion_counts(pred[sample, 0], gt[sample, 0])
            tp += matched
            fn += missed
    dice = dice_total / max(batches, 1)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return dice, recall


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.gpu_mem_fraction:
        # Cap how much VRAM this run may allocate so the rest of the GPU stays
        # free for other work while training.
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device.index or 0)
    root = args.data
    train_labels = args.train_labels or (root / "labels" / "train")
    train_ds = LesionSegDataset(
        root / "images" / "train", train_labels,
        imgsz=args.imgsz, augment=True, background_ratio=args.background_ratio,
    )
    val_ds = LesionSegDataset(
        root / "images" / "val", root / "labels" / "val", imgsz=args.imgsz, augment=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    print(f"train {len(train_ds)} | val {len(val_ds)} | device {device}")

    if args.encoder:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(device)
        print(f"model: smp U-Net, encoder={args.encoder} (ImageNet-pretrained)")
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)
        print("model: from-scratch U-Net (GroupNorm)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = args.output
    (out / "weights").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["epoch", "seconds", "train_loss", "val_dice", "val_lesion_recall"])

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        for images, masks in train_dl:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = combo_loss(logits, masks)
            scaler.scale(loss).backward()
            # Clip after unscaling: prevents a single large step from driving all
            # logits negative and collapsing the model to an empty mask.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss)
        scheduler.step()
        train_loss = running / max(len(train_dl), 1)
        val_dice, val_recall = evaluate(model, val_dl, device)
        elapsed = time.time() - start
        print(
            f"epoch {epoch}/{args.epochs}  loss {train_loss:.4f}  "
            f"val_dice {val_dice:.4f}  val_lesion_recall {val_recall:.4f}  ({elapsed:.0f}s)"
        )
        with results_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [epoch, round(elapsed, 1), round(train_loss, 5), round(val_dice, 5), round(val_recall, 5)]
            )
        torch.save(model.state_dict(), out / "weights" / "last.pt")
        # Select the best checkpoint on dice + recall so a pure over-predictor
        # (huge recall, ~0 dice) cannot win.
        score = val_dice + val_recall
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out / "weights" / "best.pt")
    print(f"best val dice+recall: {best_score:.4f}")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base / "detector_2p5d")
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=None,
        help="Override the train label dir (e.g. LST-AI-completed labels); val is unchanged.",
    )
    parser.add_argument("--output", type=Path, default=base / "seg_runs" / "unet_2p5d")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.0,
        help="Background slices kept per lesion slice (balances the collapse-prone data).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--encoder",
        default=None,
        help="smp encoder name (e.g. resnet34) for an ImageNet-pretrained U-Net; "
        "omit to use the from-scratch GroupNorm U-Net.",
    )
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=None,
        help="Cap VRAM usage to this fraction (e.g. 0.55) to keep the GPU free for other work.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
