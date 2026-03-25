from __future__ import annotations

import logging
import logging.config
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "math_worker_file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": str(LOG_DIR / "math_worker.log"),
            "encoding": "utf-8",
        },
        "db_worker_file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": str(LOG_DIR / "db_worker.log"),
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "math_worker": {
            "handlers": ["math_worker_file"],
            "level": "INFO",
            "propagate": False,
        },
        "db_worker": {
            "handlers": ["db_worker_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


logging.config.dictConfig(LOGGING_CONFIG)

logger_math_worker = logging.getLogger("math_worker")
logger_db_worker = logging.getLogger("db_worker")


