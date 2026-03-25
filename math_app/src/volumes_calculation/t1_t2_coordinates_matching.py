from pathlib import Path
from typing import List

import imageio
import numpy as np
import nibabel as nib
from collections  import defaultdict
from nibabel import filebasedimages
from skimage.color import label2rgb
import cv2
from scipy.ndimage import map_coordinates
from config.logs_config import logger_main
from config.settings import settings
from src.volumes_calculation.pixel_count import load_freesurfer_lut


def save_colored_wmparc_slice(
    wmparc_slice: np.ndarray,
    lut_path: Path,
    output_path: Path
) -> None:
    """
    Сохраняет раскрашенное изображение одного среза wmparc.

    Args:
        wmparc_slice (np.ndarray): 2D массив с метками (label ids).
        lut_path (Path): Путь к FreeSurferColorLUT.txt.
        output_path (Path): Куда сохранить PNG.
    """
    # Загружаем LUT
    lut = load_freesurfer_lut(lut_path)

    # Формируем массив цветов по индексам (от 0 до max)
    max_label = wmparc_slice.max()
    colors = np.zeros((max_label + 1, 3))
    for label_id, (_, rgb) in lut.items():
        if label_id <= max_label:
            colors[label_id] = np.array(rgb) / 255.0  # нормализация

    # Получаем цветное изображение
    rgb_image = label2rgb(wmparc_slice, colors=colors, bg_label=0)
    # Сохраняем
    imageio.imwrite(str(output_path), (rgb_image * 255).astype(np.uint8))


def register_parcellation_to_t2_space(
        t1_path: Path,
        t2_path: Path,
        wmparc_path: Path,
        to_wmparc_dir: Path
) -> dict:
    """
    Правильная регистрация парцелляции с T1 на T2 пространство.
    """
    try:
        # Загрузка изображений
        t1_nii = nib.load(str(t1_path))
        t2_nii = nib.load(str(t2_path))
        wmparc_nii = nib.load(str(wmparc_path))
    except Exception as e:
        logger_main.error(f"Error loading files: {e}")
        return {}

    # Получение данных
    t1_data = t1_nii.get_fdata()
    t2_data = t2_nii.get_fdata()
    wmparc_data = wmparc_nii.get_fdata().astype(np.int32)

    t1_affine = t1_nii.affine
    t2_affine = t2_nii.affine
    wmparc_affine = wmparc_nii.affine

    logger_main.info(f"T1 shape: {t1_data.shape}, T2 shape: {t2_data.shape}, WMPARC shape: {wmparc_data.shape}")
    logger_main.info(f"T1 affine:\n{t1_affine}")
    logger_main.info(f"T2 affine:\n{t2_affine}")
    logger_main.info(f"WMPARC affine:\n{wmparc_affine}")

    # Создание координатной сетки для T2 пространства
    t2_shape = t2_data.shape
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(t2_shape[0]),
        np.arange(t2_shape[1]),
        np.arange(t2_shape[2]),
        indexing='ij'
    )

    # Преобразование T2 пиксельных координат в мировые координаты
    # Формируем однородные координаты [x, y, z, 1]
    ones = np.ones(x_coords.shape)
    t2_world_coords = np.stack([
        x_coords.flatten(),
        y_coords.flatten(),
        z_coords.flatten(),
        ones.flatten()
    ])

    # Преобразуем в мировые координаты
    world_coords = t2_affine @ t2_world_coords  # [4, N]

    # Преобразуем мировые координаты в координаты парцелляции
    inv_wmparc_affine = np.linalg.inv(wmparc_affine)
    wmparc_coords = inv_wmparc_affine @ world_coords  # [4, N]

    # Извлекаем 3D координаты
    wmparc_coords_3d = wmparc_coords[:3, :]  # [3, N]

    # Ресэмплинг парцелляции на пространство T2
    logger_main.info("Resampling parcellation to T2 space...")

    # Изменяем форму координат для map_coordinates
    wmparc_coords_reshaped = wmparc_coords_3d.reshape(3, t2_shape[0], t2_shape[1], t2_shape[2])

    # Ресэмплинг с nearest neighbor интерполяцией
    resampled_wmparc = np.zeros(t2_shape, dtype=np.int32)

    for z in range(t2_shape[2]):
        if z % 10 == 0:
            logger_main.info(f"Processing slice {z}/{t2_shape[2]}")

        coords_for_slice = [
            wmparc_coords_reshaped[0, :, :, z],
            wmparc_coords_reshaped[1, :, :, z],
            wmparc_coords_reshaped[2, :, :, z]
        ]

        resampled_wmparc[:, :, z] = map_coordinates(
            wmparc_data,
            coords_for_slice,
            order=0,  # nearest neighbor
            mode='constant',
            cval=0,
            prefilter=False
        )

    # Сохранение результата
    to_wmparc_dir.mkdir(exist_ok=True, parents=True)
    toNIIparc = to_wmparc_dir.parent / 'ParcelationMask'
    toNIIparc.mkdir(exist_ok=True)

    # Создание NIfTI с правильным аффином (T2)
    result_nii = nib.Nifti1Image(resampled_wmparc, t2_affine)
    nib.save(result_nii, toNIIparc / "parcelation.nii")

    # Сохранение цветных срезов
    choosen_layers = {}
    lut_path = settings.color_settings

    for i in range(resampled_wmparc.shape[2]):  # Сохраняем первые 10 срезов для проверки
        slice_data = resampled_wmparc[:, :, i]
        choosen_layers[i] = slice_data
        save_colored_wmparc_slice(
            slice_data,
            lut_path,
            to_wmparc_dir / f'Slice_{i}.png'
        )

    logger_main.info(f"Registration completed. Saved to {toNIIparc / 'parcelation.nii'}")

    return choosen_layers


def debug_coordinate_transformation(
        t1_path: Path,
        t2_path: Path,
        wmparc_path: Path
):
    """
    Функция для отладки преобразования координат.
    """
    t1_nii = nib.load(str(t1_path))
    t2_nii = nib.load(str(t2_path))
    wmparc_nii = nib.load(str(wmparc_path))

    logger_main.info("=== Coordinate System Debug Info ===")
    logger_main.info(f"T1 affine determinant: {np.linalg.det(t1_nii.affine)}")
    logger_main.info(f"T2 affine determinant: {np.linalg.det(t2_nii.affine)}")
    logger_main.info(f"WMPARC affine determinant: {np.linalg.det(wmparc_nii.affine)}")

    # Проверка размеров вокселей
    t1_voxel_sizes = np.sqrt(np.sum(t1_nii.affine[:3, :3] ** 2, axis=0))
    t2_voxel_sizes = np.sqrt(np.sum(t2_nii.affine[:3, :3] ** 2, axis=0))
    logger_main.info(f"T1 voxel sizes: {t1_voxel_sizes}")
    logger_main.info(f"T2 voxel sizes: {t2_voxel_sizes}")


def wmparc(
        wmparc_path: Path,
        path_to_t1: Path,
        path_to_t2: Path,
        to_wmparc_dir: Path
) -> dict:
    """
    Основная функция для регистрации парцелляции.
    """
    # Отладочная информация
    debug_coordinate_transformation(path_to_t1, path_to_t2, wmparc_path)

    # Правильная регистрация
    return register_parcellation_to_t2_space(
        path_to_t1, path_to_t2, wmparc_path, to_wmparc_dir
    )





