import uuid
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from passlib.context import CryptContext

from models.models import User as UserModel, Research as ResearchModel, ClassesArea as ClassesAreaModel
from schemas.schemas import UserSchema

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Проверяет, соответствует ли обычный пароль хешированному."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Создает хеш из обычного пароля."""
    return pwd_context.hash(password)

async def get_user(db: AsyncSession, user_id: uuid.UUID) -> UserModel | None:
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    return result.scalars().first()

async def get_user_by_email(db: AsyncSession, email: str) -> UserModel | None:
    result = await db.execute(select(UserModel).where(UserModel.email == email))
    return result.scalars().first()

async def get_users(db: AsyncSession) -> List[UserModel]:
    result = await db.execute(select(UserModel))
    return result.scalars().all()

async def create_user(db: AsyncSession, user: UserSchema) -> UserModel:
    """Создает нового пользователя с хешированным паролем."""
    # Хешируем пароль перед сохранением
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(email=user.email, password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
async def get_user_researches(db: AsyncSession, user_id: uuid.UUID) -> List[ResearchModel]:
    result = await db.execute(
        select(ResearchModel)
        .where(ResearchModel.user_id == user_id)
        .options(
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.localization),
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.classification),
            selectinload(ResearchModel.user),
            selectinload(ResearchModel.patient),
            selectinload(ResearchModel.status)
        )
    )
    return result.scalars().all()