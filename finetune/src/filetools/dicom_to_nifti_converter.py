from pathlib import Path
import logging
import os
import shutil
import tempfile
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings
import numpy as np
import pydicom
from pydicom import FileDataset
try:
    # DicomDir удалён в pydicom >= 3.0; используется только в аннотации типа.
    from pydicom.dicomdir import DicomDir
except ImportError:  # pragma: no cover
    DicomDir = None
from pydicom.errors import InvalidDicomError
from typing import Callable, Optional, Union
from finetune.config.logs_config import *
from dicom2nifti.exceptions import ConversionValidationError


class DicomConversionError(RuntimeError):
    """Raised when a DICOM study cannot be converted without data loss."""


class _Dicom2NiftiFailureHandler(logging.Handler):
    """Collect failures that dicom2nifti logs instead of raising."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.failures: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(("Unable to read:", "Unable to convert:")):
            self.failures.append(message)


def configure_gdcmconv() -> None:
    """Use the real GDCM executable bundled by python-gdcm on Windows.

    The console wrapper installed in ``Scripts/gdcmconv`` has no ``.exe``
    suffix and cannot be launched by CreateProcess (WinError 193).
    """
    if os.name != "nt":
        return
    try:
        import _gdcm
    except ImportError:
        return

    executable = Path(_gdcm.__file__).resolve().parent / "gdcmconv.exe"
    if executable.is_file():
        dicom2nifti_settings.gdcmconv_path = str(executable)


configure_gdcmconv()


def _ascii_staging_root() -> Path:
    """Return a case-independent ASCII directory suitable for Windows GDCM."""
    configured_root = os.getenv("FINETUNE_DICOM_TEMP")
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else settings.database_path / "_conversion_temp"
    ).resolve()

    try:
        str(root).encode("ascii")
    except UnicodeEncodeError as exc:
        raise DicomConversionError(
            "GDCM requires an ASCII-only temporary path on Windows. "
            "Set FINETUNE_DICOM_TEMP to a writable ASCII-only directory."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    return root


def process_dicom_to_nifti(
    dicom_source: Path,
    dicom_temp_dir: Path,
    nifti_output_dir: Path,
    filter_func: Optional[Callable[[FileDataset], bool]] = None
) -> None:
    logger_processing_manager.info(f"Starting DICOM-to-NIfTI conversion from {dicom_source}")

    # The legacy temporary path is nested under the case name. Do not pass it
    # to GDCM because Windows GDCM cannot reliably open paths with Cyrillic or
    # other non-ASCII characters.
    staging_root = _ascii_staging_root()
    if dicom_temp_dir.exists():
        shutil.rmtree(dicom_temp_dir, ignore_errors=True)

    if nifti_output_dir.exists():
        shutil.rmtree(nifti_output_dir)
    nifti_output_dir.mkdir(parents=True, exist_ok=True)

    failure_handler = _Dicom2NiftiFailureHandler()
    dicom2nifti_logger = logging.getLogger("dicom2nifti.convert_dir")
    previous_dicom2nifti_level = dicom2nifti_logger.level
    previous_dicom2nifti_disabled = dicom2nifti_logger.disabled
    copied_files = 0

    try:
        with tempfile.TemporaryDirectory(prefix="dicom_", dir=staging_root) as temp_dir:
            ascii_temp_dir = Path(temp_dir)

            for file_path in dicom_source.rglob('*'):
                if not file_path.is_file():
                    continue
                if any(parent.name.casefold() == "markups" for parent in file_path.parents):
                    continue

                try:
                    dicom_content: Union[FileDataset, DicomDir] = pydicom.dcmread(file_path)

                    if filter_func is None or filter_func(dicom_content):
                        # Numeric names prevent both Unicode issues and basename
                        # collisions between nested DICOM directories.
                        staged_path = ascii_temp_dir / f"{copied_files:08d}.dcm"
                        shutil.copy2(file_path, staged_path)
                        copied_files += 1
                        logger_processing_manager.debug(f"Copied: {file_path.name}")

                except InvalidDicomError:
                    logger_processing_manager.debug(f"Skipped non-DICOM file: {file_path.name}")
                except Exception as exc:
                    raise DicomConversionError(
                        f"Failed to read DICOM file {file_path.name}: {exc}"
                    ) from exc

            if copied_files == 0:
                raise DicomConversionError(
                    f"No DICOM files matched the requested series in {dicom_source}"
                )

            dicom2nifti_logger.setLevel(logging.INFO)
            dicom2nifti_logger.disabled = False
            dicom2nifti_logger.addHandler(failure_handler)
            dicom2nifti.convert_directory(
                str(ascii_temp_dir),
                str(nifti_output_dir),
                compression=True,
                reorient=True
            )

            if failure_handler.failures:
                details = "; ".join(failure_handler.failures[:3])
                raise DicomConversionError(
                    f"dicom2nifti reported partial conversion failure: {details}"
                )

            nifti_files = list(nifti_output_dir.glob("*.nii")) + list(
                nifti_output_dir.glob("*.nii.gz")
            )
            if not nifti_files:
                raise DicomConversionError(
                    "DICOM conversion produced no NIfTI files"
                )

            logger_processing_manager.info(
                f"Conversion successful, NIfTI files saved in {nifti_output_dir}"
            )
    except (ConversionValidationError, DicomConversionError) as exc:
        logger_processing_manager.error(f"Conversion failed: {exc}")
        raise
    except Exception as exc:
        logger_processing_manager.error(f"Conversion failed: {exc}")
        raise DicomConversionError(f"DICOM conversion failed: {exc}") from exc
    finally:
        dicom2nifti_logger.removeHandler(failure_handler)
        dicom2nifti_logger.setLevel(previous_dicom2nifti_level)
        dicom2nifti_logger.disabled = previous_dicom2nifti_disabled
        logger_processing_manager.info("Temporary DICOM files removed")


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
    Отбор аксиальных T2/FLAIR-серий для YOLO-датасета.

    Аксиальность проверяется по ImageOrientationPatient, а НЕ по тексту описания
    (прежний фильтр требовал слово ``t2`` в SeriesDescription и отбрасывал серии,
    названные просто ``FLAIR``).

    Принимаются:
    - аксиальный FLAIR (``flair``/``fluid`` в описании либо ``IR`` в ScanningSequence) —
      берётся целиком;
    - серии ``DUAL_TSE`` — только T2-эхо (``EchoTime >= 70`` мс). Дуальное эхо хранит
      PD- и T2-изображения в одной серии (один SeriesInstanceUID) на одинаковых позициях
      срезов; без фильтра по эху dicom2nifti получает по два изображения на позицию и
      падает на несогласованном объёме.
    """
    try:
        orientation = np.round(dicom_content.ImageOrientationPatient)
    except AttributeError:
        logger_main.warning("Missing ImageOrientationPatient in DICOM file while filtering T2.")
        return False

    if not dicom_orientation_ok(orientation):
        return False

    desc = (getattr(dicom_content, "SeriesDescription", "") or "").lower()
    scanning_seq = getattr(dicom_content, "ScanningSequence", "") or ""

    is_flair = "flair" in desc or "fluid" in desc or "IR" in scanning_seq

    # DUAL_TSE: оставляем только длинноэховые (T2) срезы.
    try:
        echo_time = float(getattr(dicom_content, "EchoTime", 0) or 0)
    except (TypeError, ValueError):
        echo_time = 0.0
    is_dual_tse_t2 = ("dual" in desc) and (echo_time >= 70)

    return is_flair or is_dual_tse_t2


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
        raise

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
            raise


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
