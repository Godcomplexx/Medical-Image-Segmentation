from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List

from db.db import get_db
from models.models import Localization as LocalizationModel, User as UserModel
from schemas.schemas import LocalizationSchema, Localization
from security import get_current_user

router = APIRouter(
    prefix="/localizations",
    tags=["localizations"]
)


@router.post("/", response_model=Localization)
async def create_localization(
        localization: LocalizationSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Создает новую локализацию.

    Входные данные:
    - **localization**: Схема `LocalizationSchema` с именем зоны из тела POST-запроса.

    Возвращает:
    - Созданный объект локализации.
    """
    db_localization = LocalizationModel(**localization.model_dump())
    db.add(db_localization)
    await db.commit()
    await db.refresh(db_localization)
    return db_localization


@router.get("/", response_model=List[Localization])
async def read_localizations(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех локализаций.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов локализаций.
    """
    result = await db.execute(select(LocalizationModel))
    return result.scalars().all()


@router.get("/{zone_id}", response_model=Localization)
async def read_localization(
        zone_id: uuid.UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Получает одну локализацию по её ID.

    Входные данные:
    - **zone_id**: UUID локализации из пути URL.

    Возвращает:
    - Объект локализации или ошибку 404.
    """
    result = await db.execute(
        select(LocalizationModel)
        .where(LocalizationModel.zone_id == zone_id)
    )
    localization = result.scalars().first()
    if not localization:
        raise HTTPException(status_code=404, detail="Localization not found")
    return localization


@router.put("/{zone_id}", response_model=Localization)
async def update_localization(
        zone_id: uuid.UUID,
        localization: LocalizationSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Обновляет существующую локализацию.

    Входные данные:
    - **zone_id**: UUID локализации из пути URL.
    - **localization**: Схема `LocalizationSchema` с новыми данными из тела PUT-запроса.

    Возвращает:
    - Обновленный объект локализации.
    """
    result = await db.execute(
        select(LocalizationModel)
        .where(LocalizationModel.zone_id == zone_id)
    )
    db_localization = result.scalars().first()
    if not db_localization:
        raise HTTPException(status_code=404, detail="Localization not found")

    for field, value in localization.model_dump().items():
        setattr(db_localization, field, value)

    db.add(db_localization)
    await db.commit()
    await db.refresh(db_localization)
    return db_localization


@router.delete("/{zone_id}")
async def delete_localization(
        zone_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Удаляет локализацию по её ID.

    Входные данные:
    - **zone_id**: UUID локализации из пути URL.

    Возвращает:
    - Сообщение об успешном удалении.
    """
    result = await db.execute(
        select(LocalizationModel)
        .where(LocalizationModel.zone_id == zone_id)
    )
    localization = result.scalars().first()
    if not localization:
        raise HTTPException(status_code=404, detail="Localization not found")

    await db.delete(localization)
    await db.commit()
    return {"message": "Localization deleted successfully"}