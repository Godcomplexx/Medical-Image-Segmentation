"""End-to-end test of Celery queue flow using pytest-celery fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from celery import Celery
from pytest_celery import celery_worker

from workers.celery_app import app as celery_app  # noqa: WPS433 — used in test


@pytest.fixture(scope="session")
def celery_app_instance() -> Celery:  # type: ignore[return-type]
    """Provide the shared Celery app to pytest-celery plugin."""

    return celery_app


def test_calculate_roundtrip(celery_session_worker: Any) -> None:  # noqa: ANN401
    """Ensure payload -> calculate -> db_stub chain succeeds."""

    payload: Dict[str, Any] = {"user_id": 1, "payload": {"dicom_path": "/data/sample.dcm"}}

    # Call task synchronously for test speed; in real life use delay/apply_async
    result = celery_app.send_task("tasks.calculate", args=[payload])

    output = result.get(timeout=10)
    assert output["user_id"] == payload["user_id"]
    assert isinstance(output["score"], (int, float))

    # Next task: simulate DB persistence (executes in same worker process during tests)
    db_result = celery_app.send_task("tasks.parse_and_store", args=[output])
    db_result.get(timeout=10)

