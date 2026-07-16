"""
Mine U-Net candidates on train and fit the lesion / not-lesion filter.

    python -m finetune.two_stage_pipeline.segmentation.train_fp_filter
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .fp_filter import FP_FEATURE_NAMES, mine
from .predict_seg import LesionSegmenter


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-weights", type=Path, default=base / "seg_runs" / "unet_2p5d" / "weights" / "best.pt")
    parser.add_argument("--images", type=Path, default=base / "detector_2p5d" / "images" / "train")
    parser.add_argument("--labels", type=Path, default=base / "detector_2p5d" / "labels" / "train")
    parser.add_argument("--output", type=Path, default=base / "seg_runs" / "unet_2p5d")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Mine only N images (debug).")
    parser.add_argument("--encoder", default=None, help="smp encoder (e.g. resnet34) matching the seg weights.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    segmenter = LesionSegmenter(
        args.seg_weights, threshold=args.threshold, min_area=args.min_area,
        device=args.device, encoder=args.encoder,
    )
    candidates_csv = args.output / "fp_candidates_train.csv"
    print("mining candidates on train...")
    counts = mine(segmenter, args.images, args.labels, candidates_csv, limit=args.limit)
    print("mined:", counts)

    rows = list(csv.DictReader(candidates_csv.open(encoding="utf-8")))
    x = np.asarray([[float(r[f]) for f in FP_FEATURE_NAMES] for r in rows], dtype=np.float64)
    y = np.asarray([int(r["is_lesion"]) for r in rows])
    print(f"candidates: {len(y)}  lesion={int(y.sum())}  background={int((1 - y).sum())}")

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=400, max_depth=4,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )
    model.fit(x_tr, y_tr)
    report = classification_report(
        y_te, model.predict(x_te), target_names=["background", "lesion"], output_dict=True, zero_division=0
    )
    print("held-out candidate classification:")
    for name in ("background", "lesion"):
        d = report[name]
        print(f"  {name}: precision={d['precision']:.3f} recall={d['recall']:.3f} f1={d['f1-score']:.3f}")

    import joblib

    joblib.dump({"model": model, "feature_names": list(FP_FEATURE_NAMES)}, args.output / "fp_filter.joblib")
    (args.output / "fp_filter_metrics.json").write_text(
        json.dumps({"counts": counts, "held_out": report}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output / 'fp_filter.joblib'}")


if __name__ == "__main__":
    main()
