"""End-to-end evaluation of the U-Net segmentation cascade.

U-Net (2.5D) proposes lesion candidates -> anatomy classifier assigns I/P/S/U ->
lesion-level per-class precision/recall/F1 on the untouched validation patients.
The segmenter is fed the 2.5D images it was trained on; ground truth is the
original multiclass labels.

    python -m finetune.two_stage_pipeline.segmentation.evaluate_seg \
        --seg-weights .../seg_runs/unet_2p5d/weights/best.pt \
        --anatomy-model .../anatomy/anatomy_classifier.joblib \
        --synthseg-dir .../anatomy/segmentations --centroid-match
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from ..data import MEDICAL_CLASSES, annotation_box, parse_yolo_segmentation
from ..evaluate import _match_image
from ..predict import AnatomyClassifier, classify_boxes
from .fp_filter import FalsePositiveFilter
from .predict_seg import LesionSegmenter

CLASS_ORDER = list(MEDICAL_CLASSES.values())


def evaluate_seg(
    seg_weights: Path,
    anatomy_model: Path,
    synthseg_dir: Path,
    images_2p5d: Path,
    labels_dir: Path,
    *,
    imgsz: int = 640,
    threshold: float = 0.5,
    min_area: int = 4,
    seg_confidence: float = 0.5,
    match_iou: float = 0.1,
    centroid_match: bool = True,
    fp_filter_path: Path | None = None,
    fp_threshold: float = 0.5,
    device: str = "cuda:0",
    encoder: str | None = None,
) -> dict:
    segmenter = LesionSegmenter(
        seg_weights, imgsz=imgsz, threshold=threshold, min_area=min_area,
        device=device, encoder=encoder,
    )
    classifier = AnatomyClassifier(anatomy_model)
    fp_filter = FalsePositiveFilter(fp_filter_path) if fp_filter_path else None
    seg_cache: dict = {}
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_ORDER}
    images = accepted = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_2p5d / f"{label_path.stem}.png"
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.width, image.height
        truth = [
            (MEDICAL_CLASSES[a.class_id], annotation_box(a, width, height))
            for a in parse_yolo_segmentation(label_path)
        ]
        candidates = [
            c
            for c in segmenter.boxes(image_path, fp_filter=fp_filter, fp_threshold=fp_threshold)
            if c["score"] >= seg_confidence
        ]
        boxes = [c["box_xyxy"] for c in candidates]
        scores = [c["score"] for c in candidates]
        predictions = classify_boxes(
            image_path, boxes, scores, classifier, synthseg_dir, seg_cache
        )
        tp, fp, fn = _match_image(predictions, truth, match_iou, centroid_match)
        for name in totals:
            totals[name]["tp"] += tp[name]
            totals[name]["fp"] += fp[name]
            totals[name]["fn"] += fn[name]
        images += 1
        accepted += len(predictions)

    per_class = {}
    for name, counts in totals.items():
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {**counts, "precision": precision, "recall": recall, "f1": f1}

    return {
        "images": images,
        "accepted_candidates": accepted,
        "seg_confidence": seg_confidence,
        "threshold": threshold,
        "match_iou": match_iou,
        "centroid_match": centroid_match,
        "macro_f1": sum(per_class[c]["f1"] for c in CLASS_ORDER) / len(CLASS_ORDER),
        "per_class": per_class,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-weights", type=Path, default=base / "seg_runs" / "unet_2p5d" / "weights" / "best.pt")
    parser.add_argument("--anatomy-model", type=Path, default=base / "anatomy" / "anatomy_classifier.joblib")
    parser.add_argument("--synthseg-dir", type=Path, default=base / "anatomy" / "segmentations")
    parser.add_argument("--images-2p5d", type=Path, default=base / "detector_2p5d" / "images" / "val")
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root / "finetune" / "fileStorage" / "train_set" / "labels" / "val",
    )
    parser.add_argument("--output", type=Path, default=base / "seg_runs" / "unet_2p5d" / "cascade_metrics.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--seg-confidence", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--no-centroid-match", dest="centroid_match", action="store_false")
    parser.add_argument("--fp-filter", type=Path, default=None, help="fp_filter.joblib for FP rejection.")
    parser.add_argument("--fp-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder", default=None, help="smp encoder name (e.g. resnet34) matching the trained weights.")
    args = parser.parse_args()

    metrics = evaluate_seg(
        args.seg_weights, args.anatomy_model, args.synthseg_dir, args.images_2p5d, args.labels,
        imgsz=args.imgsz, threshold=args.threshold, min_area=args.min_area,
        seg_confidence=args.seg_confidence, match_iou=args.match_iou,
        centroid_match=args.centroid_match, fp_filter_path=args.fp_filter,
        fp_threshold=args.fp_threshold, device=args.device, encoder=args.encoder,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["per_class"], ensure_ascii=False, indent=2))
    print("macro_f1", metrics["macro_f1"], "accepted", metrics["accepted_candidates"])


if __name__ == "__main__":
    main()
