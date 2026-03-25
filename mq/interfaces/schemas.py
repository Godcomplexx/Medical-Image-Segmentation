"""Shared Pydantic schemas for payloads passed through the queues."""
from __future__ import annotations

from typing import Dict, Any

from pydantic import BaseModel, Field


class MathInput(BaseModel):
    """Message placed on *math_tasks* queue by API layer."""

    user_id: int = Field(..., ge=1)
    payload: Dict[str, Any]


class MathOutput(BaseModel):
    """Result produced by `tasks.calculate` and consumed by DB task."""

    folder_name: str
    status: str
    result: Dict[str, Any]
    task_id: str
