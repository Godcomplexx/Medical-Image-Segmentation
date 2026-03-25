import shutil
from pathlib import Path
import multiprocessing
from finetune.config.logs_config import *

from finetune.src.filetools.dicom_to_nifti_converter import create_t2_nifti
from finetune.src.filetools.memory_check import check_disk_space
from finetune.src.filetools.nifti_tools import get_nifti_meta
from finetune.src.yolo.image_generator import generate_images
from finetune.src.yolo.label_generator import (create_demyelination_areas_markups,
                                      generate_labels,
                                      ensure_empty_labels_exist)


def process_survey(survey_path: Path) -> bool:
    """
    Creates yolo dataset (images and labels) from DICOM and Markups
    :param survey_path: Path to directory with DICOM and Markups inside
    :return:
    """
    try:
        logger_main.info(f"Processing {survey_path.name}")

        if not check_disk_space(settings.processed_data_path):
            logger_processing_manager.error(f"Skipping {survey_path.name} due to low disk space.")
            return False

        create_t2_nifti(
            survey_path=survey_path,
            postpr_path=settings.processed_data_path
        )

        epi_img, epi_img_data = get_nifti_meta(
            nifti_path=settings.processed_data_path / survey_path.name,
            volume_name='t2')

        generate_images(
            epi_img_data=epi_img_data,
            yolo_path=settings.processed_data_path / survey_path.name
        )

        new_points_lps = create_demyelination_areas_markups(
            epi_img=epi_img,
            survey_path=survey_path
        )

        generate_labels(
            epi_img=epi_img,
            new_points_lps=new_points_lps,
            yolo_path=settings.processed_data_path / survey_path.name
        )

        ensure_empty_labels_exist(yolo_path=settings.processed_data_path / survey_path.name / 'Yolo')

        shutil.rmtree(settings.processed_data_path / survey_path.name / 'NIFTI', ignore_errors=True)

        logger_processing_manager.info(f"Completed processing {survey_path.name}")
        return True
    except Exception as e:
        logger_processing_manager.error(f"Error processing {survey_path.name}: {str(e)}")
        return False


def run_pipeline() -> None:
    """
    Creating yolo dataset from all directories that contains DICOM and Markups in input data
    :return:
    """
    logger_main.info("Starting yolo_dataset pipeline.")

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    survey_paths: list[Path] = [
        p for p in settings.input_data_path.iterdir()
        if p.is_dir() and not p.name.startswith('.') and not (settings.processed_data_path / p.name).exists()
    ]
    if not survey_paths:
        logger_main.info("No new surveys to process. Everything is up-to-date.")
        return

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_survey, survey_paths)

    successful = sum(results)
    logger_main.info(f"\nProcessing complete. Successfully processed {successful}/{len(results)} surveys.")


if __name__ == '__main__':
    run_pipeline()
