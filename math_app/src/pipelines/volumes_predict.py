import multiprocessing
from pathlib import Path
import time

from config.logs_config import logger_main
from src.filetools.dicom_to_nifti_converter import create_t1_nifti, create_t2_nifti
from src.filetools.memory_check import check_disk_space
from src.filetools.nifti_tools import get_nifti_meta, make_zero_affine
from src.volumes_calculation.fastsurfer_morphometry import run_fastsurfer
from src.volumes_calculation.pixel_count import count_colors_in_masks
from src.volumes_calculation.t1_t2_coordinates_matching import wmparc
from src.yolo.predict import example_case, predict_on_folder
from src.yolo.image_generator import generate_images
import shutil
from config.settings import settings
from src.filetools.cleaner import minimal_fastsurfer_cleanup

def wait_for_fastsurfer_result(result_dir: Path, expected_file: str, timeout: int = 600):
    """
    Ожидает появления файла результата FastSurfer.
    """
    result_path = result_dir / expected_file
    start = time.time()
    while not result_path.exists():
        if time.time() - start > timeout:
            return False
        time.sleep(5)
    return True


def process_survey(survey_path: Path) -> (dict, bool):
    """
    Creates predicted data with dicom
    :param survey_path: Path to directory with DICOM inside
    :return: json file with results and bool type value
    """
    try:

        logger_main.info(f"Processing {survey_path.name}")

        # if not check_disk_space(settings.processed_data_path):
        #     logger_main.error(f"Skipping {survey_path.name} due to low disk space.")
        #     return {survey_path.name : f"Skipping {survey_path.name} due to low disk space."}, False

        create_t1_nifti(
            survey_path=survey_path,
            postpr_path=settings.processed_data_path
        )

        for_fastsurfer = make_zero_affine(
            nifti_path=settings.processed_data_path / survey_path.name,
            volume_name='T1')

        run_fastsurfer(t1_path=for_fastsurfer,
                                           output_dir=settings.morphometry_path,
                                           license_path=settings.lic_path,
                                           subject_id=survey_path.name)

        create_t2_nifti(
            survey_path=survey_path,
            postpr_path=settings.processed_data_path
        )
        for_correlation = make_zero_affine(
            nifti_path=settings.processed_data_path / survey_path.name,
            volume_name='T2'
        )

        epi_img, epi_img_data = get_nifti_meta(
            nifti_path=settings.processed_data_path / survey_path.name,
            volume_name='t2')

        generate_images(
            epi_img_data=epi_img_data,
            yolo_path=settings.processed_data_path / survey_path.name
        )

        predict_on_folder(
            path2t2=settings.processed_data_path / survey_path.name
        )

        choosen_wmparc = wmparc(
            wmparc_path=settings.morphometry_path / survey_path.name / 'mri' / 'wmparc.DKTatlas.mapped.mgz',
            path_to_t1=settings.morphometry_path / survey_path.name / 'mri' / 'T1.mgz',
            path_to_t2=for_correlation,
            to_wmparc_dir=settings.processed_data_path / survey_path.name / 'Morfometry'
        )

        result = count_colors_in_masks(
            image_dir=choosen_wmparc,
            mask_dir=settings.processed_data_path / survey_path.name / "Mask",
            docker_subject_id=survey_path,
            nifti_path=for_correlation
        )

        minimal_fastsurfer_cleanup(survey_path.name)


        shutil.rmtree(settings.processed_data_path / survey_path.name / 'NIFTI', ignore_errors=True)
        shutil.rmtree(settings.processed_data_path / survey_path.name / 'NIFTI_T1', ignore_errors=True)
        shutil.rmtree(settings.processed_data_path / survey_path.name / 'Mask', ignore_errors=True)
        shutil.rmtree(settings.processed_data_path / survey_path.name / 'Morfometry', ignore_errors=True)
        shutil.rmtree(survey_path, ignore_errors=True)

        logger_main.info(f"Completed processing {survey_path.name}")
        return result
    except Exception as e:
        logger_main.error(f"Error processing {survey_path.name}: {str(e)}")
        minimal_fastsurfer_cleanup(survey_path.name)
        return {survey_path.name : str(e)}, False


