from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(
        Path(__file__).resolve().parents[1]
        / "fileStorage"
        / "two_stage"
        / "ultralytics"
    ),
)

import torch
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    default_data_root = (
        project_root / "finetune" / "fileStorage" / "two_stage"
    )
    parser = argparse.ArgumentParser(description="Train the binary lesion detector.")
    parser.add_argument("--model", default="yolo11m-seg.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_root / "detector" / "dataset.yaml",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=default_data_root / "runs",
    )
    parser.add_argument("--name", default="binary_detector")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--mask-ratio", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for detector training")
    if not args.data.is_file():
        raise FileNotFoundError(
            f"{args.data} does not exist; run the data preparation first"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.1,
        warmup_epochs=3.0,
        deterministic=True,
        seed=42,
        cache=False,
        amp=True,
        cos_lr=True,
        mask_ratio=args.mask_ratio,
        overlap_mask=True,
        save_period=5,
        plots=True,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
    )


if __name__ == "__main__":
    main()
