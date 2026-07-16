from pathlib import Path

import numpy as np
from PIL import Image

from finetune.src.yolo.image_generator import generate_images, normalize_mri_volume


def test_normalization_uses_one_window_for_the_complete_volume() -> None:
    volume = np.zeros((16, 16, 2), dtype=np.float32)
    volume[:, :, 0] = np.linspace(0, 100, 256).reshape(16, 16)
    volume[:, :, 1] = np.linspace(1, 2, 256).reshape(16, 16)

    normalized = normalize_mri_volume(volume, 0, 100)

    assert normalized.dtype == np.uint8
    assert normalized[:, :, 0].max() == 255
    assert normalized[:, :, 1].max() <= 6


def test_generate_images_preserves_voxel_dimensions(tmp_path: Path) -> None:
    volume = np.arange(12 * 10 * 3, dtype=np.float32).reshape(12, 10, 3)
    unicode_output = tmp_path / "исследование"

    generate_images(volume, unicode_output)

    images = sorted((unicode_output / "Yolo" / "images").glob("*.png"))
    assert len(images) == 3
    with Image.open(images[0]) as image:
        assert image.size == (10, 12)
