from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import uuid
from typing import List

from db.db import get_db
from models.models import ClassesArea as ClassesAreaModel, User as UserModel
from schemas.schemas import ClassesAreaSchema, ClassesArea
from security import get_current_user

router = APIRouter(
    prefix="/classes_areas",
    tags=["classes_areas"]
)


@router.post("/", response_model=ClassesArea)
async def create_classes_area(
        classes_area: ClassesAreaSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Создает новую область классификации, предварительно проверяя
    существование связанных локализации и классификации.

    Входные данные:
    - **classes_area**: Схема `ClassesAreaSchema` с данными об области из тела POST-запроса.

    Возвращает:
    - Созданный объект области классификации.
    """
    from models.models import Localization as LocalizationModel, Classification as ClassificationModel

    localization = await db.execute(
        select(LocalizationModel)
        .where(LocalizationModel.zone_id == classes_area.localization_id)
    )
    if not localization.scalars().first():
        raise HTTPException(status_code=404, detail="Localization not found")

    classification = await db.execute(
        select(ClassificationModel)
        .where(ClassificationModel.classification_id == classes_area.class_id)
    )
    if not classification.scalars().first():
        raise HTTPException(status_code=404, detail="Classification not found")

    db_classes_area = ClassesAreaModel(**classes_area.model_dump())
    db.add(db_classes_area)
    await db.commit()
    await db.refresh(db_classes_area)

    result = await db.execute(
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id == db_classes_area.id)
        .options(
            selectinload(ClassesAreaModel.localization),
            selectinload(ClassesAreaModel.classification)
        )
    )
    return result.scalars().first()


@router.get("/", response_model=List[ClassesArea])
async def read_classes_areas(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех областей классификации.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов областей классификации.
    """
    result = await db.execute(
        select(ClassesAreaModel)
        .options(
            selectinload(ClassesAreaModel.localization),
            selectinload(ClassesAreaModel.classification)
        )
    )
    return result.scalars().all()


@router.get("/{area_id}", response_model=ClassesArea)
async def read_classes_area(
        area_id: uuid.UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Получает одну область классификации по её ID.

    Входные данные:
    - **area_id**: UUID области из пути URL.

    Возвращает:
    - Объект области или ошибку 404.
    """
    result = await db.execute(
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id == area_id)
        .options(
            selectinload(ClassesAreaModel.localization),
            selectinload(ClassesAreaModel.classification)
        )
    )
    area = result.scalars().first()
    if not area:
        raise HTTPException(status_code=404, detail="Classes area not found")
    return area


@router.put("/{area_id}", response_model=ClassesArea)
async def update_classes_area(
        area_id: uuid.UUID,
        classes_area: ClassesAreaSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Обновляет существующую область классификации.

    Входные данные:
    - **area_id**: UUID области из пути URL.
    - **classes_area**: Схема `ClassesAreaSchema` с новыми данными из тела PUT-запроса.

    Возвращает:
    - Обновленный объект области.
    """
    from models.models import Localization as LocalizationModel, Classification as ClassificationModel

    localization = await db.execute(
        select(LocalizationModel)
        .where(LocalizationModel.zone_id == classes_area.localization_id)
    )
    if not localization.scalars().first():
        raise HTTPException(status_code=404, detail="Localization not found")

    classification = await db.execute(
        select(ClassificationModel)
        .where(ClassificationModel.classification_id == classes_area.class_id)
    )
    if not classification.scalars().first():
        raise HTTPException(status_code=404, detail="Classification not found")

    result = await db.execute(
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id == area_id)
    )
    db_classes_area = result.scalars().first()
    if not db_classes_area:
        raise HTTPException(status_code=404, detail="Classes area not found")

    for field, value in classes_area.model_dump().items():
        setattr(db_classes_area, field, value)

    db.add(db_classes_area)
    await db.commit()

    result = await db.execute(
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id == area_id)
        .options(
            selectinload(ClassesAreaModel.localization),
            selectinload(ClassesAreaModel.classification)
        )
    )
    return result.scalars().first()


@router.delete("/{area_id}")
async def delete_classes_area(
        area_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Удаляет область классификации по её ID.

    Входные данные:
    - **area_id**: UUID области из пути URL.

    Возвращает:
    - Сообщение об успешном удалении.
    """
    result = await db.execute(
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id == area_id)
    )
    area = result.scalars().first()
    if not area:
        raise HTTPException(status_code=404, detail="Classes area not found")

    await db.delete(area)
    await db.commit()
    return {"message": "Classes area deleted successfully"}