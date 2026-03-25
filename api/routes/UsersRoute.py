import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from db.db import get_db
from schemas.schemas import UserSchema, User, Research
from crud import crud_user

router = APIRouter(
    prefix="/users",
    tags=["users"]
)


@router.post("/", response_model=User)
async def create_user(user: UserSchema, db: AsyncSession = Depends(get_db)):
    """
    Создает нового пользователя.

    Входные данные:
    - **user**: Схема `UserSchema` с email и паролем, полученная из тела POST-запроса.

    Возвращает:
    - Созданный объект пользователя.
    """
    # Проверяем, не занят ли уже такой email
    db_user = await crud_user.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    # Вызываем CRUD-функцию для создания пользователя
    return await crud_user.create_user(db=db, user=user)


@router.get("/", response_model=List[User])
async def read_users(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех пользователей.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов пользователей.
    """
    return await crud_user.get_users(db)


@router.get("/{user_id}", response_model=User)
async def read_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Получает одного пользователя по его ID.

    Входные данные:
    - **user_id**: UUID пользователя, полученный из пути URL (например, /users/{user_id}).

    Возвращает:
    - Объект пользователя или ошибку 404, если пользователь не найден.
    """
    db_user = await crud_user.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/{user_id}/researches", response_model=List[Research])
async def read_user_researches(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Получает список всех проектов для конкретного пользователя.

    Входные данные:
    - **user_id**: UUID пользователя, полученный из пути URL.

    Возвращает:
    - Список объектов проектов, связанных с этим пользователем.
    """
    user = await crud_user.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    researches = await crud_user.get_user_researches(db, user_id=user_id)
    return researches


# ИЗМЕНЕНО: Эндпоинт для регистрации
@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserSchema, db: AsyncSession = Depends(get_db)):
    """Создает нового пользователя (регистрация)."""
    db_user = await crud_user.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return await crud_user.create_user(db=db, user=user)

# НОВЫЙ ЭНДПОИНТ для входа
@router.post("/login", response_model=User)
async def login_user(user_data: UserSchema, db: AsyncSession = Depends(get_db)):
    """Аутентифицирует пользователя и возвращает его данные в случае успеха."""
    user = await crud_user.get_user_by_email(db, email=user_data.email)
    if not user or not crud_user.verify_password(user_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    return user
