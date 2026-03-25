from pathlib import Path
from typing import Optional
from config.logs_config import *
from src.volumes_calculation.send_to_fastsurfer import send_to_fastsurfer_api


def run_fastsurfer(
        t1_path: Path,
        output_dir: Path,
        license_path: Path,
        subject_id: Optional[str] = None,
        fastsurfer_url: str = "http://fastsurfer:8001"
) -> bool:

    """
    Отправляет T1-файл и параметры в FastSurfer API для сегментации.
    Все пути формируются относительно /fileStorage для совместимости с контейнером FastSurfer.
    """

    logger_main.info("FastSurfer start")
    subject_id = subject_id or t1_path.stem

    # Пути внутри контейнера FastSurfer
    output_dir_in_container = Path("/fileStorage/for_volume_calc")
    # Если у вас другой путь к license.txt, укажите его ниже
    license_path_in_container = Path("/fileStorage/fast_surfer_lic/license.txt")




    response = send_to_fastsurfer_api(
        t1_path=t1_path,
        subject_id=subject_id,
        license_path=license_path_in_container,
        output_dir=output_dir_in_container,
        fastsurfer_url=fastsurfer_url
    )
    if response.get("status") == "ok":
        logger_main.info(f"FastSurfer segmentation completed for {t1_path}")
        return True
    else:
        logger_processing_manager.error(f"FastSurfer error: {response}")
        return False
