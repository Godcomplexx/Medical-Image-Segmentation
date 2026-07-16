"""JWT и распознавание внутреннего сервисного токена."""
import uuid

import jwt
import pytest
from starlette.requests import Request

import security


def _request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def test_access_token_roundtrip():
    uid = uuid.uuid4()
    token = security.create_access_token(subject=uid)
    payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
    assert payload["sub"] == str(uid)
    assert "exp" in payload


def test_tampered_token_is_rejected():
    with pytest.raises(jwt.PyJWTError):
        jwt.decode("not.a.valid.token", security.SECRET_KEY, algorithms=[security.ALGORITHM])


def test_internal_token_constant_time_match(monkeypatch):
    monkeypatch.setattr(security, "INTERNAL_API_TOKEN", "the-internal-token")
    assert security._is_internal_request(_request({"X-Internal-Token": "the-internal-token"})) is True
    assert security._is_internal_request(_request({"X-Internal-Token": "wrong"})) is False
    assert security._is_internal_request(_request({})) is False


def test_internal_token_disabled_when_unset(monkeypatch):
    monkeypatch.setattr(security, "INTERNAL_API_TOKEN", "")
    assert security._is_internal_request(_request({"X-Internal-Token": "anything"})) is False
