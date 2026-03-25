from pathlib import Path
import shutil
import dicom2nifti
import numpy as np
import pydicom
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from pydicom.errors import InvalidDicomError
from typing import Callable, Optional, Union
from finetune.config.logs_config import *
from dicom2nifti.exceptions import ConversionValidationError


def process_dicom_to_nifti(
    dicom_source: Path,
    dicom_temp_dir: Path,
    nifti_output_dir: Path,
    filter_func: Optional[Callable[[FileDataset], bool]] = None
) -> None:
    logger_processing_manager.info(f"Starting DICOM-to-NIfTI conversion from {dicom_source}")

    dicom_temp_dir.mkdir(parents=True, exist_ok=True)
    nifti_output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in dicom_source.rglob('*'):
        if not file_path.is_file():
            continue

        try:
            dicom_content: Union[FileDataset, DicomDir] = pydicom.dcmread(file_path)

            if filter_func is None or filter_func(dicom_content):
                shutil.copy(file_path, dicom_temp_dir / file_path.name)
                logger_processing_manager.debug(f"Copied: {file_path.name}")

        except pydicom.errors.InvalidDicomError:
            logger_processing_manager.warning(f"Invalid DICOM file: {file_path.name}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to read {file_path.name}: {str(e)}")

    try:
        if any(dicom_temp_dir.iterdir()):
            dicom2nifti.convert_directory(
                str(dicom_temp_dir),
                str(nifti_output_dir),
                compression=True,
                reorient=True
            )
            logger_processing_manager.info(f"Conversion successful, NIfTI files saved in {nifti_output_dir}")
        else:
            logger_processing_manager.warning(f"No DICOM files found for conversion in {dicom_source}")
    except ConversionValidationError as e:
        logger_processing_manager.warning(f"Skipped conversion: {e}")
    except Exception as e:
        logger_processing_manager.error(f"Conversion failed: {str(e)}")
    finally:
        if dicom_temp_dir.exists():
            shutil.rmtree(dicom_temp_dir, ignore_errors=True)
            logger_processing_manager.info(f"Temporary DICOM files removed from {dicom_temp_dir}")


def dicom_orientation_ok(orientation) -> bool:
    """
    Checks if the DICOM orientation matrix matches axial (TRA) orientation.
    """
    return all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])


def t1_filter(dicom_content: FileDataset) -> bool:
    """
    Filter function for T1-weighted images.
    """
    try:
        orientation = np.round(dicom_content.ImageOrientationPatient)
        echo_time = dicom_content.EchoTime < 20
        desc = dicom_content.SeriesDescription.lower()
        thickness = 0.8 < dicom_content.SliceThickness < 1.2

        orientation_ok = dicom_orientation_ok(orientation)
        mode_ok = 't1' in desc
        tra_ok = 'tra' in desc

        return (mode_ok or echo_time) and (orientation_ok or tra_ok) and thickness
    except AttributeError:
        logger_main.warning("Missing attribute in DICOM file while filtering T1.")
        return False


def t2_filter(dicom_content: FileDataset) -> bool:
    """
    Filter function for T2 FLAIR images.
    """
    try:
        orientation = np.round(dicom_content.ImageOrientationPatient)
        desc = dicom_content.SeriesDescription.lower()

        orientation_ok = dicom_orientation_ok(orientation)
        mode_ok = ('flair' in desc or 'fluid' in desc or 'IR' in dicom_content.ScanningSequence)
        tra_ok = 't2' in desc

        return mode_ok and (orientation_ok and tra_ok)
    except AttributeError:
        logger_main.warning("Missing attribute in DICOM file while filtering T2.")
        return False


def create_nifti_series(
    source_root: Path,
    target_root: Path,
    temp_subdir: str,
    nifti_subdir: str,
    dicom_filter: Callable[[FileDataset], bool],
    match_subfolders: bool = False
) -> None:
    """
    Generic function to convert DICOM series to NIfTI format.

    Args:
        source_root (Path): Path to root directory/tree containing DICOM data.
        target_root (Path): Path to output directory for NIfTI files.
        temp_subdir (str): Subdirectory name for temporary DICOM files.
        nifti_subdir (str): Subdirectory name for NIfTI output files.
        dicom_filter (Callable): Filter function to select appropriate DICOM files.
        match_subfolders (bool): Whether to process only matching subfolders between source and target.
    """
    logger_main.info(f"Starting create_nifti_series: {source_root}, {target_root}")

    try:
        if match_subfolders:
            target_folders = [
                folder for folder in target_root.iterdir()
                if folder.is_dir() and (source_root / folder.name).exists()
            ]
        else:
            target_folders = [source_root]

    except Exception as e:
        logger_main.error(f"Failed to list folders: {e}")
        return

    for folder in target_folders:
        logger_main.info(f"Processing folder: {folder.name}")

        try:
            process_dicom_to_nifti(
                dicom_source=(
                    source_root / folder.name / 'DICOM'
                    if match_subfolders
                    else folder
                ),
                dicom_temp_dir=(
                    target_root / folder.name / temp_subdir
                    if match_subfolders
                    else target_root / folder.name / temp_subdir
                ),
                nifti_output_dir=(
                    target_root / folder.name / nifti_subdir
                    if match_subfolders
                    else target_root / folder.name / nifti_subdir
                ),
                filter_func=dicom_filter
            )

            logger_main.info(f"Successfully processed: {folder.name}")
        except Exception as e:
            logger_main.error(f"Error while processing folder {folder.name}: {e}")


def create_t1_nifti(
        survey_path: Path,
        postpr_path: Path = settings.processed_data_path
) -> None:
    """
    Converts T1 DICOM to NIFTI
    :param survey_path: Path to directory/tree that contains DICOM data
    :param postpr_path: Output path
    :return:
    """
    create_nifti_series(
        source_root=survey_path,
        target_root=postpr_path,
        temp_subdir='DICOM_T1_TEMP',
        nifti_subdir='NIFTI_T1',
        dicom_filter=t1_filter,
        match_subfolders=False
    )


def create_t2_nifti(
        survey_path: Path,
        postpr_path: Path = settings.processed_data_path
) -> None:
    """
    Converts T2 DICOM to NIFTI
    :param survey_path: Path to directory/tree that contains DICOM data
    :param postpr_path: Output path
    :return:
    """
    create_nifti_series(
        source_root=survey_path,
        target_root=postpr_path,
        temp_subdir='DICOM_TEMP',
        nifti_subdir='NIFTI',
        dicom_filter=t2_filter,
        match_subfolders=False
    )
