"""
Celery tasks для математической обработки (Math-Worker).
После получения результата от math_app мы сразу публикуем его
в очередь db_tasks → tasks.save_json (DB-Worker).
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
from celery import shared_task, current_app
import requests as _requests
try:
    from mq.config.logging_config import logger_math_worker
except ImportError:  # Container layout: /app/config
    from config.logging_config import logger_math_worker



MATH_APP_URL = os.getenv("MATH_APP_URL", "http://math_app:8000")  # можно переопределить переменной окружения
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
INTERNAL_HEADERS = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}


from typing import Optional


def _mark_failed(research_id: Optional[str], message: str) -> None:
    """Сообщает API, что обработка исследования завершилась ошибкой."""
    if not research_id:
        logger_math_worker.warning("research_id не передан — не можем выставить статус 'Ошибка'")
        return
    try:
        url = f"http://api_container:8000/researches/{research_id}/fail"
        _requests.post(url, json={"error_message": str(message)[:2000]}, headers=INTERNAL_HEADERS, timeout=10)
    except Exception as exc:  # noqa: BLE001 — обработка ошибок не должна падать
        logger_math_worker.error(f"Не удалось выставить статус 'Ошибка': {exc}")


@shared_task(name="tasks.calculate", bind=True, max_retries=3)
def calculate(self, folder_name: str, research_id: Optional[str] = None) -> dict:
    """
    Вызывает энд-поинт /run_process у сервиса math_app и
    пересылает полученный JSON в DB-воркер.
    """
    logger_math_worker.info(f"Получена задача: {folder_name}")

    try:
        # Не запускаем ML, пока API атомарно не подтвердит Queued -> Processing.
        if research_id:
            processing_status_id = _get_status_id("В обработке")
            if not processing_status_id:
                raise requests.RequestException("Processing status is unavailable")
            api_url = f"http://api_container:8000/researches/{research_id}/status"
            status_response = _requests.put(
                api_url,
                json={"status_id": processing_status_id},
                headers=INTERNAL_HEADERS,
                timeout=10,
            )
            status_response.raise_for_status()
        else:
            logger_math_worker.warning("research_id не передан — пропускаем обновление статуса на 'В обработке'")

        url = f"{MATH_APP_URL}/run_process"
        logger_math_worker.info(f"POST → {url}")

        response = requests.post(
            url,
            data={"folder_name": folder_name},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=None,        # большая обработка — ждём сколько нужно
        )

        logger_math_worker.info(f"Ответ: {response.status_code} {response.text}")
        response.raise_for_status()

        result_json = response.json()          # ← то, что вернём API-клиенту

        # ─── НОВОЕ: публикуем JSON в очередь db_tasks ────────────────────────
        current_app.send_task(
            "tasks.save_json",                 # задача DB-воркера
            args=[result_json],   # JSON + имя папки (для имени файла)
            queue="db_tasks"
        )
        # ─────────────────────────────────────────────────────────────────────

        return result_json

    except requests.RequestException as exc:
        # сетевые ошибки / ошибки обработки — повторим позже
        logger_math_worker.exception(f"Ошибка запроса: {exc}")
        if exc.response is not None:
            logger_math_worker.error(f"Тело ответа: {exc.response.text}")
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            # Повторы исчерпаны — фиксируем неустранимый сбой.
            _mark_failed(research_id, f"Обработка не удалась после повторов: {exc}")
            raise


def _get_status_id(status_name: str) -> str:
    try:
        url = f"http://api_container:8000/statuses"
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        items = resp.json()
        for item in items:
            if item.get("status_name") == status_name:
                return item.get("id")
    except Exception as _e:
        logger_math_worker.error(f"Ошибка получения id статуса '{status_name}': {_e}")
    return ""

