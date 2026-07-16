"""Unit-level verification of the current Celery task chain."""
from __future__ import annotations

from dataclasses import dataclass

import requests
import pytest

from workers.db import tasks as db_tasks
from workers.math_worker import tasks as math_tasks


@dataclass
class FakeResponse:
    payload: dict
    status_code: int = 200
    text: str = "ok"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def json(self) -> dict:
        return self.payload


class FakeCeleryApp:
    def __init__(self) -> None:
        self.calls = []

    def send_task(self, name, args, queue):
        self.calls.append((name, args, queue))


def test_calculate_dispatches_current_save_task(monkeypatch):
    research_id = "2f8e9c82-315f-4ad8-acb7-d58fd752d63c"
    result_payload = {
        "result": [{research_id: [["zone", "class", "12.5"]]}]
    }
    status_updates = []
    fake_app = FakeCeleryApp()

    monkeypatch.setattr(math_tasks, "current_app", fake_app)
    monkeypatch.setattr(math_tasks, "_get_status_id", lambda _: "processing-status-id")
    monkeypatch.setattr(
        math_tasks._requests,
        "put",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or FakeResponse({}),
    )
    monkeypatch.setattr(
        math_tasks.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(result_payload),
    )

    result = math_tasks.calculate.run("2f8e9", research_id)

    assert result == result_payload
    assert status_updates
    assert fake_app.calls == [
        ("tasks.save_json", [result_payload], "db_tasks")
    ]


def test_calculate_does_not_run_ml_when_status_transition_is_rejected(monkeypatch):
    ml_calls = []

    monkeypatch.setattr(math_tasks, "_get_status_id", lambda _: "processing-status-id")
    monkeypatch.setattr(
        math_tasks._requests,
        "put",
        lambda *args, **kwargs: FakeResponse({}, status_code=409),
    )
    monkeypatch.setattr(
        math_tasks.requests,
        "post",
        lambda *args, **kwargs: ml_calls.append((args, kwargs)),
    )

    # Direct task invocation re-raises the original HTTP error; a worker
    # invocation schedules the configured retry.
    with pytest.raises(requests.HTTPError):
        math_tasks.calculate.run("2f8e9", "2f8e9c82-315f-4ad8-acb7-d58fd752d63c")

    assert ml_calls == []


def test_save_json_posts_current_result_contract(monkeypatch):
    research_id = "2f8e9c82-315f-4ad8-acb7-d58fd752d63c"
    areas = [["zone", "class", "12.5"]]
    payload = {"result": [{research_id: areas}]}
    calls = []

    monkeypatch.setattr(
        db_tasks.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or FakeResponse({}),
    )

    db_tasks.save_json.run(payload)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].endswith(f"/researches/{research_id}/classes_areas")
    assert kwargs["json"] == areas
    assert kwargs["timeout"] == 60
