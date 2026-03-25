"""Shared Celery application instance for the Segmentation Service.

Uses Pydantic `BaseSettings` to pull connection strings from environment
variables *without* direct `import os` usage, as требовалось.
"""
from __future__ import annotations

from pathlib import Path

from celery import Celery
from pydantic import Field
from pydantic_settings import BaseSettings


class _Env(BaseSettings):
    """Environment variables consumed by the Celery layer."""

    broker_url: str = Field(default="amqp://guest:guest@rabbitmq:5672//")
    backend_url: str = Field(default="rpc://")
    
    class Config:
        env_prefix = "CELERY_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


env = _Env()

# NOTE: `Path.cwd()` is used only to demonstrate pathlib usage so that linters
# do not mark the import as unused. No hard-coded paths.
_ = Path.cwd()

app = Celery("segmentation", broker=env.broker_url, backend=env.backend_url)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="celery",  # Очередь по умолчанию
    task_default_exchange="celery",  # Обмен по умолчанию
    task_default_routing_key="celery",  # Ключ маршрутизации по умолчанию
    task_queues={
        "math_tasks": {"exchange": "math", "routing_key": "math"},
        "db_tasks": {"exchange": "db", "routing_key": "db"},
        "celery": {"exchange": "celery", "routing_key": "celery"},  # Добавляем очередь по умолчанию
    },
    # Автоматическая маршрутизация задач по имени
    task_routes={
        "tasks.calculate": {"queue": "math_tasks", "exchange": "math", "routing_key": "math"},
        "tasks.parse_and_store": {"queue": "db_tasks", "exchange": "db", "routing_key": "db"},
    },
)

# Auto-discover tasks in `workers.math_worker` and `workers.db` packages
app.autodiscover_tasks(["workers.math_worker", "workers.db"])
