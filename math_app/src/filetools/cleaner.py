import shutil
from pathlib import Path
from typing import List, Optional
from config.logs_config import logger_cleaner
from config.settings import settings


def minimal_fastsurfer_cleanup(subject_name: str) -> None:
    """
    Минимальная очистка - оставляет только wmparc.DKTatlas.mapped.mgz и T1.mgz, удаляет всё остальное.

    Args:
        subject_name: Имя субъекта (папка внутри settings.morphometry_path)
    """
    subject_dir = settings.morphometry_path / subject_name
    if not subject_dir.exists():
        logger_cleaner.warning(f"Subject directory not found: {subject_dir}")
        return

    mri_dir = subject_dir / "mri"
    if not mri_dir.exists():
        logger_cleaner.warning(f"MRI directory not found: {mri_dir}")
        return

    # Файлы, которые нужно сохранить
    files_to_save = [
        mri_dir / "wmparc.DKTatlas.mapped.mgz",
        mri_dir / "T1.mgz"
    ]

    temp_files = []

    # Копируем нужные файлы во временную директорию
    try:
        for file_path in files_to_save:
            if not file_path.exists():
                logger_cleaner.warning(f"File not found: {file_path}")
                continue
            temp_file = subject_dir / f"temp_{file_path.name}"
            shutil.copy2(file_path, temp_file)
            temp_files.append((temp_file, file_path))
        if not temp_files:
            logger_cleaner.warning("No files were copied for backup.")
            return
        logger_cleaner.info("Created temporary copies of required files")
    except Exception as e:
        logger_cleaner.error(f"Failed to create temporary copies: {e}")
        return

    # Удаляем всё содержимое subject_dir
    try:
        for item in subject_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        logger_cleaner.info(f"Deleted all contents of {subject_dir}")
    except Exception as e:
        logger_cleaner.error(f"Failed to clean subject directory: {e}")
        # Удаляем временные файлы при ошибке
        for temp_file, _ in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        return

    # Создаем mri директорию заново
    mri_dir.mkdir(parents=True, exist_ok=True)

    # Восстанавливаем файлы
    try:
        for temp_file, original_path in temp_files:
            shutil.move(str(temp_file), str(original_path))
        logger_cleaner.info("Restored required files")
    except Exception as e:
        logger_cleaner.error(f"Failed to restore files: {e}")