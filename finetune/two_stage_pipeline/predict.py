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

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .anatomy import lesion_features, parse_study_slice
from .data import contextual_square_box
from .synthseg_anatomy import (
    brain_z_extent,
    lesion_features_from_seg,
    z_norm_from_extent,
)


def predict_image(
    image_path: Path,
    detector: YOLO,
    classifier: YOLO,
    *,
    detector_confidence: float = 0.05,
    classifier_confidence: float = 0.5,
    context: float = 2.0,
    minimum_side: int = 48,
    device: str = "0",
    include_rejected: bool = False,
) -> dict[str, Any]:
    detector_result = detector.predict(
        source=str(image_path),
        conf=detector_confidence,
        device=device,
        verbose=False,
    )[0]
    boxes = (
        detector_result.boxes.xyxy.detach().cpu().tolist()
        if detector_result.boxes is not None
        else []
    )
    detector_scores = (
        detector_result.boxes.conf.detach().cpu().tolist()
        if detector_result.boxes is not None
        else []
    )
    if not boxes:
        return {"image": str(image_path), "lesions": [], "rejected": []}

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop_boxes = [
            contextual_square_box(
                box,
                image.width,
                image.height,
                context=context,
                minimum_side=minimum_side,
            )
            for box in boxes
        ]
        # Ultralytics accepts a list of HWC numpy images.
        crops = [np.asarray(image.crop(crop_box)) for crop_box in crop_boxes]

    classification_results = classifier.predict(
        source=crops,
        device=device,
        verbose=False,
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for box, crop_box, detector_score, classification in zip(
        boxes,
        crop_boxes,
        detector_scores,
        classification_results,
    ):
        if classification.probs is None:
            raise RuntimeError("Classifier returned no class probabilities")
        class_index = int(classification.probs.top1)
        class_score = float(classification.probs.top1conf)
        class_name = str(classifier.names[class_index])
        record = {
            "class": class_name,
            "detector_confidence": float(detector_score),
            "classifier_confidence": class_score,
            "joint_score": float(detector_score) * class_score,
            "box_xyxy": [round(float(value), 3) for value in box],
            "context_box_xyxy": list(crop_box),
        }
        if class_name.casefold() == "background":
            record["rejection_reason"] = "classified_as_background"
            rejected.append(record)
        elif class_score < classifier_confidence:
            record["rejection_reason"] = "classifier_confidence_below_threshold"
            rejected.append(record)
        else:
            accepted.append(record)

    response: dict[str, Any] = {"image": str(image_path), "lesions": accepted}
    if include_rejected:
        response["rejected"] = rejected
    return response


class AnatomyClassifier:
    """Wraps the fitted anatomy model (joblib bundle) for cascade inference."""

    def __init__(self, bundle_path: Path):
        import joblib

        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

    def classify(self, vectors: list[list[float]]) -> list[tuple[str, float]]:
        if not vectors:
            return []
        matrix = np.asarray(vectors, dtype=np.float64)
        probabilities = self.model.predict_proba(matrix)
        classes = list(self.model.classes_)
        best = probabilities.argmax(axis=1)
        return [(str(classes[i]), float(probabilities[row, i])) for row, i in enumerate(best)]


def _load_segmentation(seg_dir: Path, study: str):
    import nibabel as nib

    for suffix in (".nii.gz", ".nii"):
        candidate = seg_dir / f"{study}{suffix}"
        if candidate.is_file():
            volume = np.asarray(nib.load(str(candidate)).get_fdata()).astype(np.int32)
            return volume, brain_z_extent(volume)
    return None, None


def classify_boxes(
    image_path: Path,
    boxes: list,
    scores: list,
    classifier: AnatomyClassifier,
    seg_dir: Path,
    seg_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Classify candidate boxes by anatomical context (shared by both cascades).

    Uses the study's SynthSeg ventricle/cortex/infratentorial masks indexed by
    slice level, falling back to the FLAIR proxy when a study has no segmentation.
    ``scores`` are the detector/segmenter confidences, folded into ``joint_score``.
    """
    if not boxes:
        return []
    study, z = parse_study_slice(image_path)
    if study not in seg_cache:
        seg_cache[study] = _load_segmentation(seg_dir, study)
    seg_volume, extent = seg_cache[study]
    use_seg = seg_volume is not None and 0 <= z < seg_volume.shape[2]
    gray = None
    if not use_seg:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))
    z_norm = z_norm_from_extent(z, extent) if use_seg else 0.5

    vectors = []
    for box in boxes:
        if use_seg:
            features = lesion_features_from_seg(seg_volume, z, box, z_norm=z_norm)
        else:
            features = lesion_features(gray, box, z_norm=z_norm)
        vectors.append(features.vector())

    lesions = []
    for box, detector_score, (class_name, class_score) in zip(
        boxes, scores, classifier.classify(vectors)
    ):
        lesions.append(
            {
                "class": class_name,
                "detector_confidence": float(detector_score),
                "classifier_confidence": class_score,
                "joint_score": float(detector_score) * class_score,
                "box_xyxy": [round(float(value), 3) for value in box],
                "context": "synthseg" if use_seg else "flair_proxy",
            }
        )
    return lesions


def predict_image_anatomy(
    image_path: Path,
    detector: YOLO,
    classifier: AnatomyClassifier,
    seg_dir: Path,
    *,
    detector_confidence: float = 0.05,
    device: str = "0",
    seg_cache: dict[str, Any] | None = None,
    include_rejected: bool = False,
) -> dict[str, Any]:
    """
    YOLO detection followed by anatomy-based I/P/S/U classification.

    The detector is the only lesion/background gate, so tune
    ``detector_confidence`` on validation.
    """
    detector_result = detector.predict(
        source=str(image_path), conf=detector_confidence, device=device, verbose=False
    )[0]
    boxes = (
        detector_result.boxes.xyxy.detach().cpu().tolist()
        if detector_result.boxes is not None
        else []
    )
    scores = (
        detector_result.boxes.conf.detach().cpu().tolist()
        if detector_result.boxes is not None
        else []
    )
    lesions = classify_boxes(
        image_path, boxes, scores, classifier, seg_dir,
        seg_cache if seg_cache is not None else {},
    )
    response = {"image": str(image_path), "lesions": lesions}
    if include_rejected:
        response["rejected"] = []
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run binary lesion detection followed by candidate classification."
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-confidence", type=float, default=0.05)
    parser.add_argument("--classifier-confidence", type=float, default=0.5)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--minimum-side", type=int, default=48)
    parser.add_argument("--device", default="0")
    parser.add_argument("--include-rejected", action="store_true")
    return parser


def _source_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.iterdir()
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg"}
        )
    raise FileNotFoundError(source)


def main() -> None:
    args = build_parser().parse_args()
    detector = YOLO(args.detector)
    classifier = YOLO(args.classifier)
    results = [
        predict_image(
            image,
            detector,
            classifier,
            detector_confidence=args.detector_confidence,
            classifier_confidence=args.classifier_confidence,
            context=args.context,
            minimum_side=args.minimum_side,
            device=args.device,
            include_rejected=args.include_rejected,
        )
        for image in _source_images(args.source)
    ]
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
