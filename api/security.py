"""JWT-аутентификация и зависимости для проверки доступа.

Две модели доступа:
- Пользователь: JWT в заголовке `Authorization: Bearer <token>` (эндпоинты UI).
- Внутренний сервис (Celery-воркеры): заголовок `X-Internal-Token`, равный
  секрету INTERNAL_API_TOKEN. Используется для server-to-server вызовов
  внутри сети (обновление статуса, сохранение площадей).

SECRET_KEY обязателен: без него защищённые эндпоинты возвращают 500,
чтобы нельзя было случайно запустить API без подписи токенов.
"""
import hmac
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from db.db import get_db
from crud import crud_user
from models.models import User as UserModel

SECRET_KEY = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET", "")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

# tokenUrl используется Swagger-кнопкой Authorize; реальный логин — POST /users/login.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/login")

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def _require_secret() -> None:
    if not SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret is not configured.",
        )


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    """Создаёт подписанный JWT с идентификатором пользователя в поле `sub`."""
    _require_secret()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def _user_from_token(token: str, db: AsyncSession) -> UserModel:
    _require_secret()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = payload.get("sub")
        if subject is None:
            raise credentials_exception
        user_id = uuid.UUID(subject)
    except (jwt.PyJWTError, ValueError):
        raise credentials_exception

    user = await crud_user.get_user(db, user_id=user_id)
    if user is None:
        raise credentials_exception
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> UserModel:
    """Декодирует JWT и возвращает соответствующего пользователя из БД."""
    return await _user_from_token(token, db)


def _is_internal_request(request: Request) -> bool:
    """Проверяет внутренний сервисный токен в постоянном по времени сравнении."""
    if not INTERNAL_API_TOKEN:
        return False
    provided = request.headers.get("X-Internal-Token", "")
    return hmac.compare_digest(provided, INTERNAL_API_TOKEN)


@dataclass
class Principal:
    """Аутентифицированный субъект запроса: пользователь либо доверенный сервис."""
    user: Optional[UserModel]  # None => внутренний сервис (воркер)

    @property
    def is_internal(self) -> bool:
        return self.user is None


async def get_principal(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Principal:
    """Допускает либо валидный внутренний токен, либо пользовательский JWT.

    Используется на эндпоинтах, которые вызываются и UI, и Celery-воркерами.
    Проверку владельца ресурса выполняет сам обработчик (только для пользователя).
    """
    if _is_internal_request(request):
        return Principal(user=None)

    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise credentials_exception
    user = await _user_from_token(token, db)
    return Principal(user=user)
