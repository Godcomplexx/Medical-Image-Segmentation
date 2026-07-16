"""Схемы: пароль не должен попадать в ответы API."""
import uuid

from schemas.schemas import User, UserSchema


def test_user_response_excludes_password():
    class ORMUser:
        id = uuid.uuid4()
        email = "doc@example.com"
        password = "bcrypt$hashed$secret"

    dumped = User.model_validate(ORMUser(), from_attributes=True).model_dump()
    assert "password" not in dumped
    assert dumped["email"] == "doc@example.com"


def test_input_schema_still_accepts_password():
    s = UserSchema(email="doc@example.com", password="plaintext")
    assert s.password == "plaintext"
