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
from config.logging_config import logger_math_worker



MATH_APP_URL = os.getenv("MATH_APP_URL", "http://math_app:8000")  # можно переопределить переменной окружения


from typing import Optional


@shared_task(name="tasks.calculate", bind=True)
def calculate(self, folder_name: str, research_id: Optional[str] = None) -> dict:
    """
    Вызывает энд-поинт /run_process у сервиса math_app и
    пересылает полученный JSON в DB-воркер.
    """
    logger_math_worker.info(f"Получена задача: {folder_name}")

    try:
        # Перед запуском обработки — обновим статус исследования на "обрабатывается"
        try:
            if research_id:
                api_url = f"http://api_container:8000/researches/{research_id}/status"
                _ = _requests.put(api_url, json={"status_id": _get_status_id("В обработке")}, timeout=10)
            else:
                logger_math_worker.warning("research_id не передан — пропускаем обновление статуса на 'В обработке'")
        except Exception as _e:
            logger_math_worker.error(f"Не удалось обновить статус на 'В обработке': {_e}")

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
        # сетевые ошибки — повторим позже
        logger_math_worker.exception(f"Ошибка запроса: {exc}")
        if exc.response is not None:
            logger_math_worker.error(f"Тело ответа: {exc.response.text}")
        raise self.retry(exc=exc, countdown=60)


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

