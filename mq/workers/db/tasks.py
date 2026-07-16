"""
Celery задачи DB-воркера.
Слушаем очередь  db_tasks  и сохраняем результаты вычислений в JSON-файлы.
"""

from __future__ import annotations
import os
import requests
import json
import time
from pathlib import Path

from celery import shared_task
try:
    from mq.config.logging_config import logger_db_worker
except ImportError:  # Container layout: /app/config
    from config.logging_config import logger_db_worker

INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
INTERNAL_HEADERS = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}


def _mark_failed(research_id, message: str) -> None:
    """Сообщает API, что обработка исследования завершилась ошибкой."""
    if not research_id:
        return
    try:
        url = f"http://api_container:8000/researches/{research_id}/fail"
        requests.post(url, json={"error_message": str(message)[:2000]}, headers=INTERNAL_HEADERS, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger_db_worker.error(f"Не удалось выставить статус 'Ошибка': {exc}")

@shared_task(name="tasks.save_json")          # очередь задаётся при вызове/роутинге
def save_json(result_dict: dict):
    """
    Сохраняет result_dict в файл  <folder_name>_YYYYMMDD_HHMMSS.json.

    Args:
        result_dict: словарь, полученный от math-воркера
        folder_name: имя исходной папки (для понятного имени файла)

    Returns:
        str — путь к сохранённому файлу
    """
    research_id = None
    try:
        research_id = list(result_dict["result"][0].keys())[0]
        data_to_send = result_dict["result"][0][research_id]
        api_url = f"http://api_container:8000/researches/{research_id}/classes_areas"
        response = requests.post(api_url, json=data_to_send, headers=INTERNAL_HEADERS, timeout=60)
        response.raise_for_status()

        # Статус исследования обновится на стороне API после успешного сохранения классов-областей

    except requests.exceptions.RequestException as e:
        logger_db_worker.exception(f"Ошибка при отправке в API-контейнер: {e}")
        # Не оставляем исследование "В обработке" навсегда — фиксируем ошибку.
        _mark_failed(research_id, f"Не удалось сохранить результаты: {e}")
    except (KeyError, IndexError) as e:
        logger_db_worker.error(f"Ошибка при извлечении research_id: {e}")

    
