from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
    find_image,
    parse_yolo_segmentation,
)
from .predict import AnatomyClassifier, predict_image, predict_image_anatomy


def _center_inside(inner: tuple[float, ...], outer: tuple[float, ...]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _lesion_hit(
    pred_box: tuple[float, ...],
    truth_box: tuple[float, ...],
    iou_threshold: float,
    centroid_match: bool,
) -> bool:
    """A hit if IoU clears the threshold, or (looser) the boxes' centres coincide.

    Centroid matching is the clinically sensible criterion for 5-15px lesions,
    where a few pixels of localisation error collapses IoU even on a real hit.
    """
    if box_iou(pred_box, truth_box) >= iou_threshold:
        return True
    if centroid_match and (
        _center_inside(pred_box, truth_box) or _center_inside(truth_box, pred_box)
    ):
        return True
    return False


def _match_image(
    predictions: list[dict[str, Any]],
    truth: list[tuple[str, tuple[float, float, float, float]]],
    iou_threshold: float,
    centroid_match: bool = False,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    true_positives = {name: 0 for name in MEDICAL_CLASSES.values()}
    false_positives = {name: 0 for name in MEDICAL_CLASSES.values()}
    false_negatives = {name: 0 for name in MEDICAL_CLASSES.values()}
    matched_truth: set[int] = set()

    ordered_predictions = sorted(
        predictions,
        key=lambda prediction: prediction["joint_score"],
        reverse=True,
    )
    for prediction in ordered_predictions:
        predicted_class = prediction["class"]
        if predicted_class not in false_positives:
            continue
        eligible = [
            (truth_index, box_iou(prediction["box_xyxy"], truth_box))
            for truth_index, (truth_class, truth_box) in enumerate(truth)
            if truth_index not in matched_truth
            and truth_class == predicted_class
            and _lesion_hit(
                prediction["box_xyxy"], truth_box, iou_threshold, centroid_match
            )
        ]
        if eligible:
            best_index, _ = max(eligible, key=lambda item: item[1])
            matched_truth.add(best_index)
            true_positives[predicted_class] += 1
        else:
            false_positives[predicted_class] += 1

    for truth_index, (truth_class, _) in enumerate(truth):
        if truth_index not in matched_truth:
            false_negatives[truth_class] += 1
    return true_positives, false_positives, false_negatives


def evaluate_validation(
    detector_weights: Path,
    classifier_weights: Path | None,
    source_root: Path,
    *,
    detector_confidence: float = 0.05,
    classifier_confidence: float = 0.5,
    match_iou: float = 0.5,
    context: float = 2.0,
    minimum_side: int = 48,
    device: str = "0",
    anatomy_model: Path | None = None,
    synthseg_dir: Path | None = None,
    centroid_match: bool = False,
) -> dict[str, Any]:
    image_dir = source_root / "images" / "val"
    label_dir = source_root / "labels" / "val"
    detector = YOLO(detector_weights)
    use_anatomy = anatomy_model is not None
    if use_anatomy:
        anatomy_classifier = AnatomyClassifier(anatomy_model)
        seg_cache: dict[str, Any] = {}
    else:
        classifier = YOLO(classifier_weights)
    totals = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in MEDICAL_CLASSES.values()
    }
    image_count = 0
    accepted_candidates = 0
    rejected_candidates = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image(image_dir, label_path.stem)
        with Image.open(image_path) as image:
            truth = [
                (
                    MEDICAL_CLASSES[annotation.class_id],
                    annotation_box(annotation, image.width, image.height),
                )
                for annotation in parse_yolo_segmentation(label_path)
            ]
        if use_anatomy:
            prediction = predict_image_anatomy(
                image_path,
                detector,
                anatomy_classifier,
                synthseg_dir,
                detector_confidence=detector_confidence,
                device=device,
                seg_cache=seg_cache,
                include_rejected=True,
            )
        else:
            prediction = predict_image(
                image_path,
                detector,
                classifier,
                detector_confidence=detector_confidence,
                classifier_confidence=classifier_confidence,
                context=context,
                minimum_side=minimum_side,
                device=device,
                include_rejected=True,
            )
        tp, fp, fn = _match_image(
            prediction["lesions"], truth, match_iou, centroid_match
        )
        for class_name in totals:
            totals[class_name]["tp"] += tp[class_name]
            totals[class_name]["fp"] += fp[class_name]
            totals[class_name]["fn"] += fn[class_name]
        image_count += 1
        accepted_candidates += len(prediction["lesions"])
        rejected_candidates += len(prediction["rejected"])

    metrics: dict[str, dict[str, float | int]] = {}
    for class_name, counts in totals.items():
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = (
            counts["tp"] / precision_denominator if precision_denominator else 0.0
        )
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[class_name] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "images": image_count,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "detector_confidence": detector_confidence,
        "classifier_confidence": classifier_confidence,
        "match_iou": match_iou,
        "centroid_match": centroid_match,
        "per_class": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate the complete cascade on unchanged validation patients."
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="YOLO crop classifier weights (omit when using --anatomy-model).",
    )
    parser.add_argument(
        "--anatomy-model",
        type=Path,
        default=None,
        help="Fitted anatomy classifier (.joblib) for the anatomy cascade.",
    )
    parser.add_argument(
        "--synthseg-dir",
        type=Path,
        default=None,
        help="Aligned SynthSeg segmentations ({study}.nii.gz) for the anatomy cascade.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "finetune" / "fileStorage" / "train_set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "finetune"
        / "fileStorage"
        / "two_stage"
        / "cascade_metrics.json",
    )
    parser.add_argument("--detector-confidence", type=float, default=0.05)
    parser.add_argument("--classifier-confidence", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--centroid-match",
        action="store_true",
        help="Count a hit when box centres coincide even below the IoU threshold "
        "(clinically sensible for tiny lesions).",
    )
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--minimum-side", type=int, default=48)
    parser.add_argument("--device", default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.anatomy_model is None and args.classifier is None:
        raise SystemExit("Provide --classifier (YOLO) or --anatomy-model (+ --synthseg-dir)")
    if args.anatomy_model is not None and args.synthseg_dir is None:
        raise SystemExit("--anatomy-model requires --synthseg-dir")
    metrics = evaluate_validation(
        args.detector,
        args.classifier,
        args.source,
        detector_confidence=args.detector_confidence,
        classifier_confidence=args.classifier_confidence,
        match_iou=args.match_iou,
        context=args.context,
        minimum_side=args.minimum_side,
        device=args.device,
        anatomy_model=args.anatomy_model,
        synthseg_dir=args.synthseg_dir,
        centroid_match=args.centroid_match,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
