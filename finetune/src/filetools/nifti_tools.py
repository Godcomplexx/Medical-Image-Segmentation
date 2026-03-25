from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Optional, Tuple
from finetune.config.logs_config import *


def get_nifti_meta(nifti_path: Path, volume_name: str) -> Tuple[Optional[nib.Nifti1Image], Optional[np.ndarray]]:
    """
    Loads the first NIfTI file found in a given directory and returns its image object and data array.

    Args:
        nifti_path (Path): Path to the directory or to a subdirectory named 'NIFTI'.
        volume_name (str): t1 or t2 NIFTI

    Returns:
        Tuple[Optional[nib.Nifti1Image], Optional[np.ndarray]]:
            (nibabel image object, numpy array of image data)
            or (None, None) if an error occurs.
    """
    logger_main.info(f"Starting get_nifti_meta for: {nifti_path}")

    if volume_name == 't2':
        nifti_dir = nifti_path if nifti_path.name == 'NIFTI' else nifti_path / 'NIFTI'
    else:
        nifti_dir = nifti_path if nifti_path.name == 'NIFTI_T1' else nifti_path / 'NIFTI_T1'

    try:
        nifti_file = next(nifti_dir.glob('*'))
        logger_main.info(f"Found NIfTI file: {nifti_file}")
        epi_img = nib.load(nifti_file)

        if not isinstance(epi_img, nib.Nifti1Image):
            logger_main.error(f"Unsupported image type: {type(epi_img)}")
            return None, None

    except StopIteration:
        logger_main.error("No NIfTI files found in the directory.")
        return None, None
    except Exception as e:
        logger_main.error(f"Failed to load NIfTI file: {e}")
        return None, None

    try:
        epi_img_data = epi_img.get_fdata()
        logger_main.info(f"NIfTI data loaded successfully: shape {epi_img.shape}")
    except Exception as e:
        logger_main.error(f"Failed to read NIfTI data: {e}")
        return None, None

    return epi_img, epi_img_data


def make_zero_affine(nifti_path: Path, volume_name: str) -> Optional[Path]:
    """
    Creates a copy of a NIfTI file with a zeroed affine matrix and saves it to a new file.

    Args:
        nifti_path (Path): Path to the directory or to a subdirectory named 'NIFTI'
        volume_name (str): t1 or t2 NIFTI

    Returns:
        Optional[Path]: Path to the new file with zeroed affine, or None if an error occurs.
    """
    logger_main.info(f"Starting make_zero_affine for: {nifti_path}")

    if volume_name == 'T2':
        nifti_dir = nifti_path if nifti_path.name == 'NIFTI' else nifti_path / 'NIFTI'
    else:
        nifti_dir = nifti_path if nifti_path.name == 'NIFTI_T1' else nifti_path / 'NIFTI_T1'

    try:
        nifti_file = next(nifti_dir.glob('*'))
        logger_main.info(f"Found NIfTI file: {nifti_file}")
        img = nib.load(nifti_file)
        if not isinstance(img, nib.Nifti1Image):
            logger_main.error(f"Unsupported image type: {type(img)}")
            return None

        zero_affine = np.zeros((4, 4))
        new_img = nib.Nifti1Image(img.get_fdata(), affine=img.header.get_base_affine(), header=img.header)

        new_path = nifti_path / f"{volume_name}.nii"

        nib.save(new_img, new_path)
        logger_main.info(f"Saved zero-affine NIfTI to: {new_path}")
        return new_path

    except Exception as e:
        logger_main.error(f"Failed to create zero-affine file: {e}")
        return None
