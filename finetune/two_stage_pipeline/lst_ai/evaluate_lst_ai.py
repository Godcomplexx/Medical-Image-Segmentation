"""
End-to-end evaluation of LST-AI on the untouched validation patients.

LST-AI writes a zone-annotated lesion volume in FLAIR native space
(``space-flair_desc-annotated_seg-lst.nii.gz``) with integer labels
``1=Periventricular 2=Juxtacortical 3=Subcortical 4=Infratentorial`` — a 1:1
match to our ``P/U/S/I`` classes. Because that volume is the same reconstructed
FLAIR grid used to generate the slice PNGs, ``annotated[:, :, z][y, x]``
corresponds to PNG pixel ``(x, y)`` of ``slice_{z}_{study}.png`` — the identical
convention the SynthSeg cascade uses (``seg_volume[:, :, z]`` indexed ``[y, x]``).

For every validation slice we turn each connected component of the annotated
plane into a predicted ``(class, box)`` and score it against the ground-truth
YOLO polygons with the shared :func:`_match_image` (lesion-level per-class
precision/recall/F1). This is the same val set and matcher used by
``segmentation.evaluate_seg``, so the macro-F1 is directly comparable to the
U-Net cascade baseline.

    python -m finetune.two_stage_pipeline.lst_ai.evaluate_lst_ai --centroid-match
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..data import MEDICAL_CLASSES, annotation_box, parse_yolo_segmentation
from ..evaluate import _match_image

CLASS_ORDER = list(MEDICAL_CLASSES.values())
# LST-AI annotation labels -> our class codes.
ZONE_TO_CLASS = {1: "P", 2: "U", 3: "S", 4: "I"}
_STEM = re.compile(r"^slice_(\d+)_(.+)$")


def _study_slice(stem: str) -> tuple[str, int] | None:
    match = _STEM.match(stem)
    if not match:
        return None
    return match.group(2), int(match.group(1))


def _load_annotated(lst_root: Path, study: str, cache: dict[str, Any]):
    if study in cache:
        return cache[study]
    import nibabel as nib

    path = lst_root / study / "output" / "space-flair_desc-annotated_seg-lst.nii.gz"
    volume = None
    if path.is_file():
        volume = np.asarray(nib.load(str(path)).get_fdata()).astype(np.int32)
    cache[study] = volume
    return volume


def _plane_predictions(plane: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    """Connected components per zone -> predicted (class, box_xyxy) records."""
    import cv2

    predictions: list[dict[str, Any]] = []
    for zone, class_name in ZONE_TO_CLASS.items():
        mask = (plane == zone).astype(np.uint8)
        if not mask.any():
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp in range(1, count):
            x, y, w, h, area = stats[comp]
            if area < min_area:
                continue
            predictions.append(
                {
                    "class": class_name,
                    "box_xyxy": [float(x), float(y), float(x + w - 1), float(y + h - 1)],
                    "joint_score": float(area),  # larger lesions matched first
                }
            )
    return predictions


def evaluate_lst_ai(
    lst_root: Path,
    images_dir: Path,
    labels_dir: Path,
    *,
    min_area: int = 4,
    match_iou: float = 0.1,
    centroid_match: bool = True,
    studies: list[str] | None = None,
) -> dict:
    cache: dict[str, Any] = {}
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_ORDER}
    evaluated = skipped_no_output = images = 0
    covered_studies: set[str] = set()

    for label_path in sorted(labels_dir.glob("*.txt")):
        parsed = _study_slice(label_path.stem)
        if parsed is None:
            continue
        study, z = parsed
        if studies is not None and study not in studies:
            continue
        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.is_file():
            continue

        annotated = _load_annotated(lst_root, study, cache)
        if annotated is None:
            skipped_no_output += 1
            continue
        if not 0 <= z < annotated.shape[2]:
            skipped_no_output += 1
            continue
        covered_studies.add(study)

        with Image.open(image_path) as image:
            width, height = image.width, image.height
        truth = [
            (MEDICAL_CLASSES[a.class_id], annotation_box(a, width, height))
            for a in parse_yolo_segmentation(label_path)
        ]
        predictions = _plane_predictions(annotated[:, :, z], min_area)

        tp, fp, fn = _match_image(predictions, truth, match_iou, centroid_match)
        for name in totals:
            totals[name]["tp"] += tp[name]
            totals[name]["fp"] += fp[name]
            totals[name]["fn"] += fn[name]
        images += 1
        evaluated += 1

    per_class = {}
    for name, counts in totals.items():
        pd = counts["tp"] + counts["fp"]
        rd = counts["tp"] + counts["fn"]
        precision = counts["tp"] / pd if pd else 0.0
        recall = counts["tp"] / rd if rd else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {**counts, "precision": precision, "recall": recall, "f1": f1}

    return {
        "images_evaluated": images,
        "skipped_no_output": skipped_no_output,
        "covered_studies": sorted(covered_studies),
        "match_iou": match_iou,
        "centroid_match": centroid_match,
        "min_area": min_area,
        "macro_f1": sum(per_class[c]["f1"] for c in CLASS_ORDER) / len(CLASS_ORDER),
        "per_class": per_class,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lst-root", type=Path, default=base / "lst_ai")
    parser.add_argument("--images", type=Path, default=base / "detector_2p5d" / "images" / "val")
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root / "finetune" / "fileStorage" / "train_set" / "labels" / "val",
    )
    parser.add_argument("--output", type=Path, default=base / "lst_ai" / "lst_ai_metrics.json")
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--no-centroid-match", dest="centroid_match", action="store_false")
    parser.add_argument("--studies", nargs="*", default=None, help="Restrict to these study names.")
    args = parser.parse_args()

    metrics = evaluate_lst_ai(
        args.lst_root, args.images, args.labels,
        min_area=args.min_area, match_iou=args.match_iou,
        centroid_match=args.centroid_match, studies=args.studies,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["per_class"], ensure_ascii=False, indent=2))
    print("macro_f1", round(metrics["macro_f1"], 4),
          "| images", metrics["images_evaluated"],
          "| studies", len(metrics["covered_studies"]))


if __name__ == "__main__":
    main()
