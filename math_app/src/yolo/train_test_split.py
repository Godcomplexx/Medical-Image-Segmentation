import math
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from config.logs_config import *
from typing import List


def create_dirs(base_dir: Path, sub_dirs: List[str]) -> None:
    """
    Creates subdirectories inside a base directory.

    Args:
        base_dir (Path): The root directory where subdirectories will be created.
        sub_dirs (List[str]): List of subdirectory names to create.
    """
    for sub_dir in sub_dirs:
        try:
            dir_path = base_dir / sub_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            logger_processing_manager.debug(f"Created directory: {dir_path}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to create directory: {str(e)}")


def move_and_rename_files(src_root: Path, dst_root: Path) -> None:
    """
    Moves and renames image and label files from the source to the destination directory.

    The new name includes the grandparent directory's name to avoid collisions.

    Args:
        src_root (Path): The source directory containing 'images' and 'labels' folders.
        dst_root (Path): The destination directory where files will be moved and renamed.
    """
    for file_path in src_root.rglob('*.*'):
        try:
            parent_name = file_path.parent.name
            if parent_name in ['images', 'labels']:
                new_name = f"{file_path.stem}_{file_path.parents[2].name}{file_path.suffix}"
                dst_path = dst_root / parent_name / new_name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dst_path))
                logger_processing_manager.debug(f"Moved and renamed {file_path} to {dst_path}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to move {file_path}: {str(e)}")


def split_dataset(base_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.2) -> None:
    """
    Splits dataset into train, validation, and test subsets.

    Args:
        base_dir (Path): Directory containing 'images' and 'labels' folders.
        train_ratio (float): Proportion of data to include in the training set.
        val_ratio (float): Proportion of data to include in the validation set.
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1")
    if not math.isclose(train_ratio + val_ratio, 1.0, rel_tol=1e-5):
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    exts = {'images': ['.png', '.jpg', '.jpeg'],
            'labels': ['.txt']
            }

    for category in ['images', 'labels']:
        try:
            valid_exts = exts[category]
            files = sorted((base_dir / category).glob('*.*'))
            basenames = [f.stem for f in files if f.suffix in valid_exts]

            # Single split into train and val (no test)
            train, val = train_test_split(
                basenames,
                test_size=val_ratio,
                random_state=42
            )

            for split_name, split_list in {'train': train, 'val': val}.items():
                for name in split_list:
                    for ext in valid_exts:
                        src = base_dir / category / f"{name}{ext}"
                        if src.exists():
                            dst_dir = base_dir / category / split_name
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            dst_path = dst_dir / src.name
                            shutil.move(str(src), dst_path)
                            logger_processing_manager.debug(f"Moved {src} to {dst_path}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to split category '{category}': {str(e)}")


def create_train_set() -> None:
    """
    Prepares the training dataset by creating necessary directories,
    copying and renaming files, and splitting them into train, validation, and test sets.
    """

    logger_processing_manager.info(
        f"Creating train set in {settings.train_set_path} from {settings.processed_data_path}"
    )

    try:
        create_dirs(settings.train_set_path, ['images', 'labels'])
        move_and_rename_files(settings.processed_data_path, settings.train_set_path)
        split_dataset(settings.train_set_path, train_ratio=0.8, val_ratio=0.2)
        logger_processing_manager.info("Train set creation completed.")
    except Exception as e:
        logger_processing_manager.error(f"Failed to create train set: {str(e)}")


if __name__ == '__main__':
    create_train_set()
