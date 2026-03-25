from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from config.logs_config import *


def generate_images(
    epi_img_data: np.ndarray,
    yolo_path: Path,
    base_filename: str = "slice",
    cmap: str = "gray"
) -> None:
    """
    Saves each slice of a 3D NIfTI image as a PNG image for YOLO training or prediction.

    Args:
        epi_img_data (np.ndarray): 3D numpy array of the NIfTI image data.
        yolo_path (Path): Output directory for YOLO dataset.
        base_filename (str, optional): Base name prefix for saved image slices. Defaults to "slice".
        cmap (str, optional): Colormap used for displaying the image slices. Defaults to 'gray'.
    """
    logger_main.info("Starting generate_yolo_images")
    images_dir: Path = yolo_path / 'Yolo' / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    num_slices: int = epi_img_data.shape[2]
    logger_main.info(f"Saving {num_slices} slices to {images_dir}")

    for i in range(num_slices):
        try:

            plt.imshow(epi_img_data[:, :, i], cmap=cmap)
            plt.axis('off')
            output_path = images_dir / f"{base_filename}_{i}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            logger_main.debug(f"Saved slice {i} to {output_path}")

        except Exception as e:
            logger_main.error(f"Failed to save slice {i}: {str(e)}")

    logger_main.info(f"YOLO image generation complete — {num_slices} slices processed.")
