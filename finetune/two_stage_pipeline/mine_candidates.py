from __future__ import annotations

import argparse
import csv
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

from PIL import Image
from ultralytics import YOLO

from .data import (
    MEDICAL_CLASSES,
    annotation_box,
    box_iou,
    contextual_square_box,
    find_image,
    parse_yolo_segmentation,
)


def mine_candidates(
    detector_weights: Path,
    source_root: Path,
    output_root: Path,
    *,
    split: str = "train",
    confidence: float = 0.05,
    match_iou: float = 0.3,
    context: float = 2.0,
    minimum_side: int = 48,
    maximum_per_image: int = 20,
    maximum_background: int = 5000,
    device: str = "0",
) -> dict[str, int]:
    source_split = "finetune" if split == "train" else "val"
    image_dir = source_root / "images" / source_split
    label_dir = source_root / "labels" / source_split
    labels = sorted(label_dir.glob("*.txt"))
    images = [find_image(image_dir, label.stem) for label in labels]
    labels_by_stem = {label.stem: label for label in labels}
    model = YOLO(detector_weights)
    summary = {"matched": 0, "background": 0, "images": len(images)}
    manifest_rows: list[dict[str, object]] = []

    results = model.predict(
        source=[str(path) for path in images],
        conf=confidence,
        device=device,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path)
        label_path = labels_by_stem[image_path.stem]
        with Image.open(image_path) as image:
            annotations = parse_yolo_segmentation(label_path)
            truth_boxes = [
                annotation_box(annotation, image.width, image.height)
                for annotation in annotations
            ]
            predicted_boxes = (
                result.boxes.xyxy.detach().cpu().tolist()
                if result.boxes is not None
                else []
            )
            confidences = (
                result.boxes.conf.detach().cpu().tolist()
                if result.boxes is not None
                else []
            )
            for candidate_index, (predicted_box, detector_confidence) in enumerate(
                zip(predicted_boxes[:maximum_per_image], confidences[:maximum_per_image])
            ):
                overlaps = [box_iou(predicted_box, truth_box) for truth_box in truth_boxes]
                best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
                if best_index is not None and overlaps[best_index] >= match_iou:
                    annotation = annotations[best_index]
                    class_name = MEDICAL_CLASSES[annotation.class_id]
                    source_class_id: int | None = annotation.class_id
                    matched_iou = overlaps[best_index]
                    summary["matched"] += 1
                else:
                    if summary["background"] >= maximum_background:
                        continue
                    class_name = "background"
                    source_class_id = None
                    matched_iou = max(overlaps, default=0.0)
                    summary["background"] += 1

                crop_box = contextual_square_box(
                    predicted_box,
                    image.width,
                    image.height,
                    context=context,
                    minimum_side=minimum_side,
                )
                destination = (
                    output_root
                    / "classifier"
                    / "images"
                    / split
                    / class_name
                    / f"{image_path.stem}__detector_{candidate_index:04d}.png"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.crop(crop_box).convert("RGB").save(destination, format="PNG")
                manifest_rows.append(
                    {
                        "split": split,
                        "crop_path": str(destination),
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "source_class_id": source_class_id,
                        "class_name": class_name,
                        "candidate_index": candidate_index,
                        "detector_confidence": float(detector_confidence),
                        "matched_iou": matched_iou,
                        "x1": crop_box[0],
                        "y1": crop_box[1],
                        "x2": crop_box[2],
                        "y2": crop_box[3],
                    }
                )

    manifest = output_root / "classifier" / f"detector_candidates_{split}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else [
            "split",
            "crop_path",
            "source_image",
            "source_label",
            "source_class_id",
            "class_name",
            "candidate_index",
            "detector_confidence",
            "matched_iou",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "finetune" / "fileStorage"
    parser = argparse.ArgumentParser(
        description="Mine matched and hard-negative detector crops for stage two."
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=data_root / "train_set")
    parser.add_argument("--output", type=Path, default=data_root / "two_stage")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--minimum-side", type=int, default=48)
    parser.add_argument("--maximum-per-image", type=int, default=20)
    parser.add_argument(
        "--maximum-background",
        type=int,
        default=5000,
        help="Maximum hard-negative crops added to this split.",
    )
    parser.add_argument("--device", default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = mine_candidates(
        args.detector,
        args.source,
        args.output,
        split=args.split,
        confidence=args.confidence,
        match_iou=args.match_iou,
        context=args.context,
        minimum_side=args.minimum_side,
        maximum_per_image=args.maximum_per_image,
        maximum_background=args.maximum_background,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
