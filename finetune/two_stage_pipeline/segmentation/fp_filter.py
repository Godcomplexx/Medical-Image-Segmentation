"""
False-positive rejection for the U-Net cascade.

The anatomy classifier cannot tell a true lesion from a false detection at the same
location (identical anatomy), so precision needs an *appearance* gate. This module
mines the U-Net's own candidates on the training split — matched to ground truth =
lesion, unmatched = hard negative — and fits a "lesion vs not-lesion" classifier on
shape / intensity / 2.5D-persistence / mask-confidence features. Real MS lesions are
compact, hyperintense and persist across neighbouring slices; vessels and artefacts
are elongated or slice-thin, which these features capture.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from .dataset import imread_rgb, rasterize_polygons
from .predict_seg import LesionSegmenter

FP_FEATURE_NAMES: tuple[str, ...] = (
    "area_norm",
    "prob_mean",
    "prob_max",
    "compactness",
    "eccentricity",
    "extent",
    "aspect",
    "intensity_mean",
    "intensity_contrast",
    "persist_prev",
    "persist_next",
)


def _eccentricity(moments: dict) -> float:
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    denom = moments["m00"] + 1e-6
    a, b, c = mu20 / denom, mu02 / denom, mu11 / denom
    common = math.sqrt(max((a - b) ** 2 + 4 * c * c, 0.0))
    l1, l2 = (a + b + common) / 2, (a + b - common) / 2
    if l1 <= 0:
        return 0.0
    return float(math.sqrt(max(1 - l2 / l1, 0.0)))


def blob_features(prob: np.ndarray, image_rgb: np.ndarray, comp: np.ndarray) -> list[float]:
    """Appearance features for one candidate blob (all arrays at model resolution)."""
    height, width = comp.shape
    area = float(comp.sum())
    ys, xs = np.nonzero(comp)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    aspect = bw / bh if bh else 1.0
    extent = area / (bw * bh) if bw * bh else 0.0

    mask_u8 = comp.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
    compactness = (4 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    eccentricity = _eccentricity(cv2.moments(mask_u8))

    green = image_rgb[:, :, 1].astype(np.float32)  # current slice
    red = image_rgb[:, :, 0].astype(np.float32)    # previous slice
    blue = image_rgb[:, :, 2].astype(np.float32)   # next slice
    blob_green = green[comp]
    ring = (cv2.dilate(mask_u8, np.ones((7, 7), np.uint8)) > 0) & (~comp)
    ring_mean = float(green[ring].mean()) if ring.any() else float(blob_green.mean())

    green_mean = float(blob_green.mean()) + 1e-6
    return [
        area / (height * width),
        float(prob[comp].mean()),
        float(prob[comp].max()),
        float(min(compactness, 1.5)),
        eccentricity,
        extent,
        float(min(aspect, 5.0)),
        green_mean / 255.0,
        (green_mean - ring_mean) / 255.0,
        float(red[comp].mean()) / green_mean,
        float(blue[comp].mean()) / green_mean,
    ]


def mine(
    segmenter: LesionSegmenter,
    images_dir: Path,
    labels_dir: Path,
    output_csv: Path,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Run the segmenter over a split and label each blob matched / unmatched."""
    images = sorted(p for p in images_dir.iterdir() if p.suffix.casefold() == ".png")
    if limit:
        images = images[:limit]
    rows: list[list[float]] = []
    labels: list[int] = []
    counts = {"lesion": 0, "background": 0, "images": 0}
    imgsz = segmenter.imgsz

    for image_path in images:
        rgb = imread_rgb(image_path)
        prob, _, _ = segmenter.probability_map(rgb)
        rgb_r = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        gt = rasterize_polygons(labels_dir / f"{image_path.stem}.txt", rgb.shape[1], rgb.shape[0])
        gt = cv2.resize(gt, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)

        mask = (prob > segmenter.threshold).astype(np.uint8)
        count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for c in range(1, count):
            if stats[c, cv2.CC_STAT_AREA] < segmenter.min_area:
                continue
            comp = comp_labels == c
            is_lesion = int(bool((gt[comp] > 0).any()))
            rows.append(blob_features(prob, rgb_r, comp))
            labels.append(is_lesion)
            counts["lesion" if is_lesion else "background"] += 1
        counts["images"] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*FP_FEATURE_NAMES, "is_lesion"])
        for feats, label in zip(rows, labels):
            writer.writerow([*feats, label])
    return counts


class FalsePositiveFilter:
    """Loads a fitted lesion/not-lesion model and scores candidate blobs."""

    def __init__(self, bundle_path: Path):
        import joblib

        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]

    def is_lesion_prob(self, prob: np.ndarray, image_rgb: np.ndarray, comp: np.ndarray) -> float:
        features = np.asarray([blob_features(prob, image_rgb, comp)], dtype=np.float64)
        return float(self.model.predict_proba(features)[0, 1])
