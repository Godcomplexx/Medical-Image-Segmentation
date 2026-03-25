from config.logs_config import *
from config.labeling_config import labeling
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import nibabel as nib
import numpy as np
import cv2
import csv


def load_freesurfer_lut(path: Path) -> dict:
    """
    Загружает LUT из FreeSurferColorLUT.txt.
    Возвращает словарь {label_index: (label_name, (R, G, B))}
    """
    lut = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    label_id = int(parts[0])
                    label_name = parts[1]
                    color = parts[2:5]
                    lut[label_id] = (label_name,tuple(map(int, color)))
                except ValueError:
                    continue
    return lut

def process_image_and_mask(image: np.array, mask_path: Path) -> dict:
    """
    Сопоставляет цвета между двумя изображениями и считает количество совпавших пикселей
    по LUT. Возвращает словарь {label_index: matched_pixel_count}.
    """
    # image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))

    if image is None or mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Конвертируем в RGB

    # Загружаем LUT и строим обратный словарь: {(R, G, B): label_index}
    lut = load_freesurfer_lut(settings.color_settings)

    # Сравнение по пикселям
    height, width = image.shape
    result = defaultdict(int)

    for y in range(height):
        for x in range(width):
            img_pixel = image[y, x]
            mask_pixel = mask[y, x]
            if any(mask_pixel) != 0 and img_pixel in lut:
                result[img_pixel] += 1

    return result


def extract_slice_number(filename: str) -> int | None:
    try:
        return int(filename.split('_')[1])
    except IndexError:
        logger_main.warning(f"Unexpected filename format: {filename}")
        return None



def count_colors_in_masks(
    image_dir: dict,
    mask_dir: Path,
    docker_subject_id: Path,
    nifti_path: Path
) -> (dict, bool):
    logger_main.info(f"Running color count for subject: {docker_subject_id}")

    lut = load_freesurfer_lut(settings.color_settings)

    zooms = nib.load(str(nifti_path)).header.get_zooms()
    pixel_area = zooms[0] * zooms[1]
    logger_main.info(f"Pixel spacing: {zooms[:2]}, pixel area: {pixel_area}")

    study_number = docker_subject_id.name
    data_rows = []
    result_json = {}
    accumulated_data = {}

    mask_paths = list(mask_dir.glob('*.png'))
    transpose_dict = {
        1: 'I',
        2: 'P',
        3: 'S',
        4: 'U',
    }
    for mask_path in mask_paths:
        slice_number = extract_slice_number(mask_path.name)
        image_path = image_dir.get(slice_number)

        if image_path is None:
            logger_main.warning(f"No matching image found for slice: {slice_number}")
            continue

        color_counts = process_image_and_mask(image_path, mask_path)

        try:
            class_number = int(mask_path.name.split('_')[2].split('.')[0])
        except (IndexError, ValueError):
            logger_main.warning(f"Invalid mask filename format: {mask_path.name}")
            continue

        for color_rgb, count in color_counts.items():
            if color_rgb == 0:
                continue

            label_id = lut.get(color_rgb)[0]
            color_name = f"LUT_{label_id}" if label_id is not None else 'Unknown'
            area = count * pixel_area
            key = (study_number, str(class_number), color_name)
            if key in accumulated_data:
                accumulated_data[key] += area
            else:
                accumulated_data[key] = area

    for (study_num, class_num, color_name), area in accumulated_data.items():
        if study_num not in result_json:
            result_json[study_num] = []
        result_json[study_num].append([transpose_dict.get(int(class_num), 'I'), color_name, f"{area:.2f}"])


    return result_json, True
