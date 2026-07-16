"""
Hybrid MS-lesion pipeline: LST-AI detection + our anatomy classifier.

LST-AI (a pretrained 3D-U-Net ensemble on T1+FLAIR) is the *detector* — it finds
lesions with markedly higher recall than our 2.5D U-Net across all classes
(I 0.41, P 0.67, S 0.61, U 0.67 vs 0.16/0.47/0.55/0.44). We then relabel each
detected lesion I/P/S/U with our own anatomy classifier trained on the *relabeled*
ground truth (the ">=50% of lesion in cortex => U" convention the annotators
confirmed), because LST-AI's built-in McDonald zones do not match our S/U boundary.

Two alignment/robustness details are handled here:

* **Resolution fix** — for studies whose reconstructed FLAIR (hence LST-AI output
  and SynthSeg) differs in in-plane size from the annotation PNG (14/41 val
  studies, e.g. CAMMS/LPS 288 vs 560), the LST-AI plane is nearest-neighbour
  resized to the PNG grid; z already matches, so this restores alignment
  (detection recall on those studies 0.00 -> 0.79).
* **Posterior-fossa filter** — the classifier otherwise labels any low-``z`` blob
  ``I``; LST-AI over-detects in the cerebellum/brainstem, giving hundreds of false
  ``I``. A prediction is only kept as ``I`` when the lesion box actually overlaps
  the infratentorial mask by :data:`INFRA_MIN_OVERLAP` (removes ~570 false I,
  keeps essentially all true I).

    python -m finetune.two_stage_pipeline.lst_ai.hybrid            # evaluate on val
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import nibabel as nib
from PIL import Image

from ..data import MEDICAL_CLASSES, annotation_box, parse_yolo_segmentation
from ..evaluate import _match_image
from ..predict import AnatomyClassifier
from ..synthseg_anatomy import (
    INFRATENTORIAL_IDS,
    brain_z_extent,
    lesion_features_from_seg,
    z_norm_from_extent,
)

CLASS_ORDER = list(MEDICAL_CLASSES.values())
INFRA_MIN_OVERLAP = 0.5  # fraction of an "I" lesion box that must sit in cerebellum/brainstem
_STEM = re.compile(r"^slice_(\d+)_(.+)$")
_SEG_NAME = "space-flair_seg-lst.nii.gz"


def _study_slice(stem: str) -> tuple[str, int] | None:
    m = _STEM.match(stem)
    return (m.group(2), int(m.group(1))) if m else None


class HybridPipeline:
    """LST-AI binary mask -> boxes -> anatomy classifier -> I/P/S/U, with the two fixes."""

    def __init__(self, lst_root: Path, synthseg_dir: Path, classifier: Path, *, min_area: int = 4):
        self.lst_root = lst_root
        self.synthseg_dir = synthseg_dir
        self.classifier = AnatomyClassifier(classifier)
        self.min_area = min_area
        self._bin: dict[str, Any] = {}
        self._seg: dict[str, Any] = {}
        self._ext: dict[str, Any] = {}

    def _lst_binary(self, study: str):
        if study not in self._bin:
            p = self.lst_root / study / "output" / _SEG_NAME
            self._bin[study] = (
                np.asarray(nib.load(str(p)).get_fdata()).astype(np.uint8) if p.is_file() else None
            )
        return self._bin[study]

    def _segmentation(self, study: str):
        if study not in self._seg:
            p = self.synthseg_dir / f"{study}.nii.gz"
            v = np.asarray(nib.load(str(p)).get_fdata()).astype(np.int32) if p.is_file() else None
            self._seg[study] = v
            self._ext[study] = brain_z_extent(v) if v is not None else None
        return self._seg[study], self._ext[study]

    def predict_slice(self, study: str, z: int, width: int, height: int) -> list[dict[str, Any]]:
        """Predicted lesions (class + box in PNG pixel coords) for one slice."""
        binary = self._lst_binary(study)
        seg, extent = self._segmentation(study)
        if binary is None or not 0 <= z < binary.shape[2]:
            return []
        plane = binary[:, :, z]
        if plane.shape != (height, width):  # resolution fix (z already matches)
            plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_NEAREST)
        count, _, stats, _ = cv2.connectedComponentsWithStats(plane, connectivity=8)
        boxes, areas = [], []
        for comp in range(1, count):
            x, y, w, h, area = stats[comp]
            if area >= self.min_area:
                boxes.append([float(x), float(y), float(x + w - 1), float(y + h - 1)])
                areas.append(float(area))
        if not boxes or seg is None or not 0 <= z < seg.shape[2]:
            return []

        sh, sw = seg.shape[0], seg.shape[1]
        sx, sy = sw / width, sh / height
        z_norm = z_norm_from_extent(z, extent)
        infra = np.isin(seg[:, :, z], list(INFRATENTORIAL_IDS)).astype(np.uint8)
        boxes_seg = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in boxes]
        vectors = [
            lesion_features_from_seg(seg, z, bs, z_norm=z_norm).vector() for bs in boxes_seg
        ]
        classes = self.classifier.classify(vectors)

        lesions: list[dict[str, Any]] = []
        for box, area, bs, (name, conf) in zip(boxes, areas, boxes_seg, classes):
            if name == "I" and not self._in_infratentorial(infra, bs):
                continue  # posterior-fossa false positive
            lesions.append(
                {"class": name, "box_xyxy": box, "joint_score": area, "classifier_confidence": conf}
            )
        return lesions

    @staticmethod
    def _in_infratentorial(infra: np.ndarray, box_seg: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = (int(box_seg[0]), int(box_seg[1]), int(box_seg[2]), int(box_seg[3]))
        region = infra[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        return region.size > 0 and float(region.mean()) >= INFRA_MIN_OVERLAP


def evaluate(
    pipeline: HybridPipeline,
    images_dir: Path,
    labels_dir: Path,
    *,
    match_iou: float = 0.1,
    centroid_match: bool = True,
) -> dict:
    # A study is "covered" only if its LST-AI volume spans the annotation's z-range.
    # 6 studies (mostly the Cyrillic set + 689) were reconstructed from a different
    # acquisition — their PNGs index z up to ~160 while the LST-AI volume has ~23
    # slices, so alignment cannot be recovered by in-plane resizing; they are
    # excluded and reported separately (they need re-reconstruction).
    max_z: dict[str, int] = {}
    for label_path in labels_dir.glob("*.txt"):
        parsed = _study_slice(label_path.stem)
        if parsed is not None:
            max_z[parsed[0]] = max(max_z.get(parsed[0], 0), parsed[1])
    covered: dict[str, bool] = {}
    for study, mz in max_z.items():
        binary = pipeline._lst_binary(study)
        covered[study] = binary is not None and binary.shape[2] > mz

    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_ORDER}
    studies: set[str] = set()
    excluded: set[str] = {s for s, ok in covered.items() if not ok and pipeline._lst_binary(s) is not None}
    slices = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        parsed = _study_slice(label_path.stem)
        if parsed is None:
            continue
        study, z = parsed
        if not covered.get(study, False):
            continue
        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.width, image.height
        predictions = pipeline.predict_slice(study, z, width, height)
        if pipeline._lst_binary(study) is None:
            continue
        truth = [
            (MEDICAL_CLASSES[a.class_id], annotation_box(a, width, height))
            for a in parse_yolo_segmentation(label_path)
        ]
        tp, fp, fn = _match_image(predictions, truth, match_iou, centroid_match)
        for c in CLASS_ORDER:
            totals[c]["tp"] += tp[c]
            totals[c]["fp"] += fp[c]
            totals[c]["fn"] += fn[c]
        studies.add(study)
        slices += 1

    per_class = {}
    for c, t in totals.items():
        pr = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
        rc = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per_class[c] = {**t, "precision": pr, "recall": rc, "f1": f1}
    return {
        "studies": len(studies),
        "excluded_studies": sorted(excluded),
        "slices": slices,
        "infra_min_overlap": INFRA_MIN_OVERLAP,
        "macro_f1": sum(per_class[c]["f1"] for c in CLASS_ORDER) / len(CLASS_ORDER),
        "macro_recall": sum(per_class[c]["recall"] for c in CLASS_ORDER) / len(CLASS_ORDER),
        "per_class": per_class,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lst-root", type=Path, default=base / "lst_ai")
    parser.add_argument("--synthseg-dir", type=Path, default=base / "anatomy" / "segmentations")
    parser.add_argument(
        "--classifier",
        type=Path,
        default=base / "anatomy" / "classifier_relabeled" / "anatomy_classifier.joblib",
    )
    parser.add_argument("--images", type=Path, default=base / "detector_2p5d" / "images" / "val")
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "finetune" / "fileStorage" / "train_set" / "labels_relabeled" / "val",
    )
    parser.add_argument("--output", type=Path, default=base / "lst_ai" / "hybrid_metrics.json")
    args = parser.parse_args()

    pipeline = HybridPipeline(args.lst_root, args.synthseg_dir, args.classifier)
    metrics = evaluate(pipeline, args.images, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in CLASS_ORDER:
        d = metrics["per_class"][c]
        print(f"  {c}: prec={d['precision']:.2f} rec={d['recall']:.2f} f1={d['f1']:.2f} "
              f"(tp={d['tp']} fp={d['fp']} fn={d['fn']})")
    print(f"  macro-F1={metrics['macro_f1']:.3f}  macro-recall={metrics['macro_recall']:.3f} "
          f"| {metrics['studies']} studies")


if __name__ == "__main__":
    main()
