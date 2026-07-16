"""
Train and evaluate the anatomy-aware I/P/S/U classifier on ground-truth lesions.

This isolates the *classification* question from detection: given a correctly
located lesion, can anatomical context separate the four classes? It reads the
ground-truth crops recorded in a ``two_stage`` / ``two_stage_full`` manifest,
derives anatomy features from the full slices (see :mod:`anatomy`), fits a
gradient-boosted classifier, and reports per-class precision / recall / F1 plus a
confusion matrix on the untouched validation patients. A deterministic clinical
rule is scored alongside as a baseline.

Run from the repository root in the training environment:

    python -m finetune.two_stage_pipeline.train_classifier_anatomy \
        --manifest finetune/fileStorage/two_stage_full/classifier/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .anatomy import (
    FEATURE_NAMES,
    build_slice_context,
    classify_by_rule,
    compute_z_extent,
    features_from_context,
    parse_study_slice,
)
from .data import MEDICAL_CLASSES, annotation_box, parse_yolo_segmentation
from .synthseg_anatomy import (
    brain_z_extent,
    lesion_features_from_seg,
    z_norm_from_extent,
)

CLASS_ORDER = list(MEDICAL_CLASSES.values())  # I, P, S, U


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _z_norm_lookup(rows: list[dict[str, str]]) -> dict[str, float]:
    """Normalized slice level per source image, using per-study min/max extent."""
    names = [row["source_image"] for row in rows]
    extent = compute_z_extent(names)
    lookup: dict[str, float] = {}
    for name in names:
        try:
            study, z = parse_study_slice(name)
        except ValueError:
            lookup[name] = 0.5
            continue
        low, high = extent[study]
        lookup[name] = 0.5 if high == low else (z - low) / (high - low)
    return lookup


def _load_segmentation(synthseg_dir: Path, study: str) -> np.ndarray | None:
    import nibabel as nib

    seg_path = synthseg_dir / f"{study}.nii.gz"
    if not seg_path.is_file():
        seg_path = synthseg_dir / f"{study}.nii"
    if not seg_path.is_file():
        return None
    return np.asarray(nib.load(str(seg_path)).get_fdata()).astype(np.int32)


def build_features(
    rows: list[dict[str, str]],
    z_lookup: dict[str, float],
    *,
    synthseg_dir: Path | None = None,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """
    Compute features for every ground-truth lesion row, caching per slice/study.

    With ``synthseg_dir`` the anatomy context comes from real SynthSeg structure
    masks (``{study}.nii.gz`` aligned to the slice grid); otherwise the
    polarity-robust FLAIR proxy is used. Lesions whose study has no segmentation
    are skipped in the SynthSeg path.
    """
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("generated_by") != "ground_truth":
            continue
        if row["class_name"] not in CLASS_ORDER:
            continue
        by_image[row["source_image"]].append(row)

    vectors: list[list[float]] = []
    labels: list[str] = []
    meta: list[dict[str, Any]] = []
    seg_cache: dict[str, np.ndarray | None] = {}
    extent_cache: dict[str, tuple[int, int]] = {}
    for source_image, image_rows in by_image.items():
        source_label = image_rows[0]["source_label"]
        annotations = parse_yolo_segmentation(Path(source_label))
        with Image.open(source_image) as image:
            width, height = image.width, image.height
            gray = None if synthseg_dir is not None else np.asarray(image.convert("L"))
        z_norm = z_lookup.get(source_image, 0.5)

        seg_volume: np.ndarray | None = None
        z_index = 0
        if synthseg_dir is not None:
            study, z_index = parse_study_slice(source_image)
            if study not in seg_cache:
                seg_cache[study] = _load_segmentation(synthseg_dir, study)
                if seg_cache[study] is not None:
                    extent_cache[study] = brain_z_extent(seg_cache[study])
            seg_volume = seg_cache[study]
            if seg_volume is None or not 0 <= z_index < seg_volume.shape[2]:
                continue
            # Consistent, study-normalized slice level from the brain span.
            z_norm = z_norm_from_extent(z_index, extent_cache[study])
        else:
            context = build_slice_context(gray)

        for row in image_rows:
            box = annotation_box(annotations[int(row["object_index"])], width, height)
            if synthseg_dir is not None:
                features = lesion_features_from_seg(
                    seg_volume, z_index, box, z_norm=z_norm
                )
            else:
                features = features_from_context(context, box, z_norm=z_norm)
            vectors.append(features.vector())
            labels.append(row["class_name"])
            meta.append({"source_image": source_image, "features": features})
    return np.asarray(vectors, dtype=np.float64), labels, meta


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return {
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in CLASS_ORDER
        },
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "confusion_matrix": {
            "labels": CLASS_ORDER,
            "rows_true_cols_pred": matrix.tolist(),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root
        / "finetune"
        / "fileStorage"
        / "two_stage_full"
        / "classifier"
        / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for the fitted model and metrics (defaults next to the manifest).",
    )
    parser.add_argument(
        "--synthseg-dir",
        type=Path,
        default=None,
        help="Directory of aligned SynthSeg segmentations ({study}.nii.gz). When "
        "given, features use real ventricle/cortex masks instead of the FLAIR proxy.",
    )
    args = parser.parse_args()

    rows = _read_manifest(args.manifest)
    z_lookup = _z_norm_lookup(rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    x_train, y_train, _ = build_features(train_rows, z_lookup, synthseg_dir=args.synthseg_dir)
    x_val, y_val, _ = build_features(val_rows, z_lookup, synthseg_dir=args.synthseg_dir)
    print(f"train lesions: {len(y_train)}  val lesions: {len(y_val)}")

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=400,
        max_depth=4,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred_model = list(model.predict(x_val))

    # Deterministic clinical rule baseline over the same features.
    from .anatomy import LesionFeatures

    pred_rule = [
        classify_by_rule(LesionFeatures(*vector)) for vector in x_val.tolist()
    ]

    metrics = {
        "manifest": str(args.manifest),
        "feature_source": "synthseg" if args.synthseg_dir else "flair_proxy",
        "features": list(FEATURE_NAMES),
        "counts": {"train": len(y_train), "val": len(y_val)},
        "learned_model": _per_class_metrics(y_val, pred_model),
        "rule_baseline": _per_class_metrics(y_val, pred_rule),
    }

    output_dir = args.output or args.manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "anatomy_classifier_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        import joblib

        joblib.dump(
            {"model": model, "feature_names": list(FEATURE_NAMES)},
            output_dir / "anatomy_classifier.joblib",
        )
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        print(f"model not persisted: {exc}")

    print(json.dumps(metrics["learned_model"]["per_class"], ensure_ascii=False, indent=2))
    print("macro F1 (learned):", metrics["learned_model"]["macro_f1"])
    print("macro F1 (rule):   ", metrics["rule_baseline"]["macro_f1"])
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
