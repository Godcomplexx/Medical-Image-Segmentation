from pathlib import Path

import numpy as np
from PIL import Image

from finetune.config.logs_config import *


def normalize_mri_volume(
    volume: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Normalize a complete MRI volume with one shared intensity window."""
    data = np.asarray(volume, dtype=np.float32)
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError("MRI volume contains no finite intensity values")

    finite_values = data[finite_mask]
    lower, upper = np.percentile(
        finite_values, [lower_percentile, upper_percentile]
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
    if upper <= lower:
        return np.zeros(data.shape, dtype=np.uint8)

    normalized = np.nan_to_num(
        (data - lower) / (upper - lower),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def generate_images(
    epi_img_data: np.ndarray,
    yolo_path: Path,
    base_filename: str = "slice",
    cmap: str = "gray",
) -> None:
    """Save normalized MRI slices as grayscale PNG images."""
    del cmap  # Retained in the signature for backward compatibility.
    logger_main.info("Starting generate_yolo_images")
    images_dir = yolo_path / "Yolo" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if epi_img_data.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, received shape {epi_img_data.shape}"
        )

    normalized_volume = normalize_mri_volume(epi_img_data)
    num_slices = normalized_volume.shape[2]
    failed_slices: list[int] = []
    logger_main.info(f"Saving {num_slices} slices to {images_dir}")

    for index in range(num_slices):
        output_path = images_dir / f"{base_filename}_{index}.png"
        try:
            image = Image.fromarray(
                np.ascontiguousarray(normalized_volume[:, :, index]),
                mode="L",
            )
            image.save(output_path, format="PNG")
            logger_main.debug(f"Saved slice {index} to {output_path}")
        except Exception as exc:
            failed_slices.append(index)
            logger_main.error(f"Failed to save slice {index}: {exc}")

    if failed_slices:
        raise OSError(
            f"Failed to write {len(failed_slices)}/{num_slices} MRI slices"
        )

    logger_main.info(
        f"YOLO image generation complete — {num_slices} slices processed."
    )
