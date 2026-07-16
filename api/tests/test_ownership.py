"""Проверка владельца исследования."""
import uuid

import pytest
from fastapi import HTTPException

from routes.ResearchRoute import _require_research_owner, _enforce_owner_unless_internal
from security import Principal


class _User:
    def __init__(self, uid):
        self.id = uid


class _Research:
    def __init__(self, uid):
        self.user_id = uid


def test_owner_allowed():
    uid = uuid.uuid4()
    _require_research_owner(_Research(uid), _User(uid))  # не бросает


def test_non_owner_forbidden():
    with pytest.raises(HTTPException) as exc:
        _require_research_owner(_Research(uuid.uuid4()), _User(uuid.uuid4()))
    assert exc.value.status_code == 403


def test_internal_principal_bypasses_ownership():
    # Внутренний сервис (user=None) не обязан быть владельцем.
    _enforce_owner_unless_internal(Principal(user=None), _Research(uuid.uuid4()))


def test_user_principal_enforced():
    with pytest.raises(HTTPException) as exc:
        _enforce_owner_unless_internal(Principal(user=_User(uuid.uuid4())), _Research(uuid.uuid4()))
    assert exc.value.status_code == 403
