from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import json
import nibabel as nib
from finetune.config.logs_config import *
from finetune.config.labeling_config import labeling


def create_demyelination_areas_markups(
    epi_img: nib.Nifti1Image,
    survey_path: Path
) -> Optional[List[Tuple[str, List[np.ndarray]]]]:
    """
    Loads JSON markup files, transforms demyelination points from LPS to voxel space,
    and returns transformed points.

    Args:
        epi_img (nib.Nifti1Image): NIfTI image object with affine matrix.
        survey_path (Path): Directory containing markup JSON files.

    Returns:
        Optional[List[Tuple[str, List[np.ndarray]]]]:
            List of tuples with file names and corresponding transformed points,
            or None if an error occurs.
    """
    logger_main.info("Starting create_demyelination_areas_markups")

    points_lps: List[Tuple[str, List[np.ndarray]]] = []

    try:
        json_files = list(survey_path.rglob('*.json'))
        if not json_files:
            logger_main.warning(f"No JSON files found in {survey_path}")
            return None
    except Exception as e:
        logger_main.error(f"Failed to list JSON files: {e}")
        return None

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            local_point_list: List[np.ndarray] = []

            for point in json_data["markups"][0]["controlPoints"]:
                orientation_matrix = np.reshape(np.array(point['orientation']), (3, 3))
                position_matrix = np.reshape(np.array(point["position"]), (3, 1))
                transformed_point = np.vstack((orientation_matrix @ position_matrix, [[1]]))
                local_point_list.append(transformed_point)

            points_lps.append((json_file.name, local_point_list))
            logger_main.info(f"Processed {json_file.name} with {len(local_point_list)} points")

        except Exception as e:
            logger_main.error(f"Error processing {json_file.name}: {e}")
            continue

    try:
        new_points_lps: List[Tuple[str, List[np.ndarray]]] = []
        inv_affine = np.linalg.inv(epi_img.affine)

        for filename, points in points_lps:
            transformed_points = [np.round(inv_affine @ point) for point in points]
            new_points_lps.append((filename, transformed_points))

        logger_main.info("Point transformation completed successfully")

    except Exception as e:
        logger_main.error(f"Failed to transform points: {e}")
        return None

    return new_points_lps


def generate_labels(
        epi_img: nib.Nifti1Image,
        new_points_lps: List[Tuple[str, List[np.ndarray]]],
        yolo_path: Path
) -> None:
    """
    Generates YOLO-compatible labels (annotations) from markups for fine-tuning.

    Args:
        epi_img: NIfTI image containing 3D volume and affine info.
        new_points_lps: List of tuples (filename, list of annotated points in LPS coordinates).
        yolo_path: Path where YOLO-compatible label files will be saved.
    """
    logger_main.info("Starting YOLO label generation.")

    try:
        labels_dir = yolo_path / "Yolo" / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        epi_data = epi_img.get_fdata()
        height, width, depth = epi_data.shape

        for slice_index in range(depth):
            annotations: List[str] = []

            for file_name, points in new_points_lps:
                if not points or int(points[0][2]) != slice_index:
                    continue

                try:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    contour = [(int(p[1]), int(p[0])) for p in points]

                    color = labeling.color_dict.get(file_name[0], (255, 255, 255))
                    color_name = ''.join(map(str, color))
                    class_id = labeling.color_to_class.get(color_name, -1)

                    if class_id == -1:
                        logger_main.warning(f"Unknown color '{color_name}' in file '{file_name}'")
                        continue

                    cv2.fillPoly(mask, [np.array(contour, dtype=np.int32)], 255)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        polygon = [f"{x[0][0] / width:.6f} {x[0][1] / height:.6f}" for x in cnt]
                        if polygon:
                            annotations.append(f"{class_id} " + " ".join(polygon))

                except Exception as e:
                    logger_main.error(f"Error processing file '{file_name}' on slice {slice_index}: {e}")
                    continue

            if annotations:
                label_path = labels_dir / f"slice_{slice_index}.txt"
                try:
                    label_path.write_text('\n'.join(annotations))
                except Exception as e:
                    logger_main.error(f"Failed to write label file {label_path}: {e}")

        logger_main.info("YOLO label generation complete.")

    except Exception as e:
        logger_main.exception(f"Unhandled exception in generate_labels: {e}")


def ensure_empty_labels_exist(yolo_path: Path) -> None:
    """
    Ensures that every image in the 'images' directory has a corresponding label file in the 'labels' directory.

    If a label file does not exist for an image, an empty .txt file will be created. This is required for
    YOLO training, which expects every image to have a corresponding label file, even if it's empty.

    Args:
        yolo_path (Path): The base directory containing 'images' and 'labels' subdirectories.
    """
    image_dir = yolo_path / 'images'
    label_dir = yolo_path / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = image_dir.glob('*.png')
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            try:
                label_path.write_text('')
                logger_processing_manager.debug(f"Created empty label file: {label_path}")
            except Exception as e:
                logger_processing_manager.error(f"Failed to create label file {label_path}: {str(e)}")
