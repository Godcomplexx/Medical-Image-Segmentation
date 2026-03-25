"""Celery client для отправки задач в MQ систему."""
from celery import Celery
from pydantic import Field
from pydantic_settings import BaseSettings


class CelerySettings(BaseSettings):
    """Настройки для подключения к Celery брокеру."""
    
    broker_url: str = Field(default="amqp://guest:guest@localhost:5672//")
    backend_url: str = Field(default="rpc://")
    
    class Config:
        env_prefix = "CELERY_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Создаем настройки
celery_settings = CelerySettings()

# Создаем Celery клиент только для отправки задач
celery_client = Celery(
    "math_app_client",
    broker=celery_settings.broker_url,
    backend=celery_settings.backend_url
)

# Настройки для клиента
celery_client.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="celery",
    task_routes={
        "tasks.process_survey": {"queue": "math_tasks"},
        "tasks.parse_and_store": {"queue": "db_tasks"},
    }
)
