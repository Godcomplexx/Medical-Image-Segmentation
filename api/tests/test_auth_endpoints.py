"""Эндпоинты аутентификации: login отдаёт JWT (без пароля), защита требует токен.

TestClient инстанцируется без контекстного менеджера, поэтому lifespan
(проверка БД) не запускается — БД для этих тестов не нужна.
"""
import uuid

from fastapi.testclient import TestClient


def _client(monkeypatch):
    from main import app
    from db.db import get_db

    async def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


def test_login_returns_jwt_and_no_password(monkeypatch):
    from crud import crud_user

    class FakeUser:
        id = uuid.uuid4()
        email = "doc@example.com"
        password = "hashed"

    async def fake_get_user_by_email(db, email):
        return FakeUser()

    monkeypatch.setattr(crud_user, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(crud_user, "verify_password", lambda plain, hashed: True)

    client = _client(monkeypatch)
    resp = client.post("/users/login", json={"email": "doc@example.com", "password": "x"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert "password" not in body


def test_login_rejects_bad_credentials(monkeypatch):
    from crud import crud_user

    async def fake_get_user_by_email(db, email):
        return None

    monkeypatch.setattr(crud_user, "get_user_by_email", fake_get_user_by_email)

    client = _client(monkeypatch)
    resp = client.post("/users/login", json={"email": "x@example.com", "password": "x"})
    assert resp.status_code == 401


def test_protected_endpoint_requires_token(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/researches/")
    assert resp.status_code == 401
