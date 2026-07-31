"""Dataset of (2.5D image, binary lesion mask) pairs for the U-Net.

Reuses the ``detector_2p5d`` inputs (prev/cur/next channels) and rasterises the
binary YOLO polygon labels into per-pixel masks on the fly, so no extra mask files
are stored. Optional light flips provide cheap augmentation.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def imread_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB, tolerant of non-ASCII (Cyrillic) Windows paths.

    ``cv2.imread`` cannot open Unicode paths on Windows, so decode from bytes.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rasterize_polygons(
    label_path: Path, width: int, height: int, *, min_box_area: float = 0.0
) -> np.ndarray:
    """Fill every YOLO polygon (any class id) into a {0,1} mask of shape (H, W).

    ``min_box_area`` drops lesions whose bounding-box area (native px) is below the
    threshold, so the detector can be trained on the same >=100 px size regime it is
    evaluated on (tiny sub-threshold lesions become background instead of positives).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.is_file():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coordinates = [float(value) for value in parts[1:]]
        points = np.array(
            [
                (coordinates[k] * width, coordinates[k + 1] * height)
                for k in range(0, len(coordinates) - 1, 2)
            ],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue
        if min_box_area > 0:
            box_area = (points[:, 0].max() - points[:, 0].min()) * (
                points[:, 1].max() - points[:, 1].min()
            )
            if box_area < min_box_area:
                continue
        cv2.fillPoly(mask, [points], 1)
    return mask


class LesionSegDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        *,
        imgsz: int = 640,
        augment: bool = False,
        background_ratio: Optional[float] = None,
        seed: int = 42,
        min_box_area: float = 0.0,
    ):
        self.labels_dir = labels_dir
        self.imgsz = imgsz
        self.augment = augment
        self.min_box_area = min_box_area
        all_images = sorted(
            p for p in images_dir.iterdir() if p.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if background_ratio is None:
            self.images = all_images
        else:
            # Most slices are pure background; training on them unbalanced makes the
            # net collapse to predicting empty masks. Keep every lesion slice and
            # only ``background_ratio`` as many background slices.
            positives, backgrounds = [], []
            for path in all_images:
                label = labels_dir / f"{path.stem}.txt"
                has_lesion = label.is_file() and bool(label.read_text(encoding="utf-8").strip())
                (positives if has_lesion else backgrounds).append(path)
            rng = random.Random(seed)
            rng.shuffle(backgrounds)
            keep = backgrounds[: int(len(positives) * background_ratio)]
            self.images = sorted(positives + keep)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        image = imread_rgb(image_path)
        height, width = image.shape[:2]
        mask = rasterize_polygons(
            self.labels_dir / f"{image_path.stem}.txt", width, height,
            min_box_area=self.min_box_area,
        )

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1])
                mask = np.ascontiguousarray(mask[::-1])

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return image_tensor, mask_tensor
