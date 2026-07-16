"""Shared logging setup for finetune and math_app."""
from __future__ import annotations

import logging
import logging.config
from datetime import datetime

import pytz


LOGGER_NAMES = (
    "interface",
    "main",
    "filtration_manager",
    "manager_settings",
    "processing_manager",
    "load_data",
    "processes_control",
    "extract",
    "cleaner",
)


def configure_logging(settings) -> dict[str, logging.Logger]:
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    timezone = pytz.timezone("Europe/Samara")

    class DefaultFormatter(logging.Formatter):
        converter = staticmethod(lambda *_: datetime.now(timezone).timetuple())

    handlers = {
        f"file_{name}": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(settings.logs_path / f"{name}.log"),
            "mode": "w",
            "encoding": "utf-8",
        }
        for name in LOGGER_NAMES
    }
    loggers = {
        name: {
            "handlers": [f"file_{name}"],
            "level": settings.log_level,
            "propagate": True,
        }
        for name in LOGGER_NAMES
    }
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "()": DefaultFormatter,
                    "format": "%(asctime)s :: %(lineno)s :: %(levelname)s :: %(message)s",
                }
            },
            "handlers": handlers,
            "loggers": loggers,
        }
    )
    return {f"logger_{name}": logging.getLogger(name) for name in LOGGER_NAMES}
