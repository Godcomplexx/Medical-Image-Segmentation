from pathlib import Path
import shutil
from finetune.config.logs_config import *


def check_disk_space(path: Path, min_required_gb: float = 10.0) -> bool:
    """Check if there is enough disk space at the given path."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_required_gb:
        logger_main.warning(f"Low disk space: {free_gb:.2f} GB left at {path}. Required: {min_required_gb} GB.")
        return False
    return True
