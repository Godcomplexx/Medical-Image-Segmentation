from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List

from db.db import get_db
from models.models import Classification as ClassificationModel, User as UserModel
from schemas.schemas import ClassificationSchema, Classification
from security import get_current_user

router = APIRouter(
    prefix="/classifications",
    tags=["classifications"]
)


@router.post("/", response_model=Classification)
async def create_classification(
        classification: ClassificationSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Создает новую классификацию.

    Входные данные:
    - **classification**: Схема `ClassificationSchema` с именем класса из тела POST-запроса.

    Возвращает:
    - Созданный объект классификации.
    """
    db_classification = ClassificationModel(**classification.model_dump())
    db.add(db_classification)
    await db.commit()
    await db.refresh(db_classification)
    return db_classification


@router.get("/", response_model=List[Classification])
async def read_classifications(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех классификаций.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов классификаций.
    """
    result = await db.execute(select(ClassificationModel))
    return result.scalars().all()


@router.get("/{classification_id}", response_model=Classification)
async def read_classification(
        classification_id: uuid.UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Получает одну классификацию по её ID.

    Входные данные:
    - **classification_id**: UUID классификации из пути URL.

    Возвращает:
    - Объект классификации или ошибку 404.
    """
    result = await db.execute(
        select(ClassificationModel)
        .where(ClassificationModel.classification_id == classification_id)
    )
    classification = result.scalars().first()
    if not classification:
        raise HTTPException(status_code=404, detail="Classification not found")
    return classification


@router.put("/{classification_id}", response_model=Classification)
async def update_classification(
        classification_id: uuid.UUID,
        classification: ClassificationSchema,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Обновляет существующую классификацию.

    Входные данные:
    - **classification_id**: UUID классификации из пути URL.
    - **classification**: Схема `ClassificationSchema` с новыми данными из тела PUT-запроса.

    Возвращает:
    - Обновленный объект классификации.
    """
    result = await db.execute(
        select(ClassificationModel)
        .where(ClassificationModel.classification_id == classification_id)
    )
    db_classification = result.scalars().first()
    if not db_classification:
        raise HTTPException(status_code=404, detail="Classification not found")

    for field, value in classification.model_dump().items():
        setattr(db_classification, field, value)

    db.add(db_classification)
    await db.commit()
    await db.refresh(db_classification)
    return db_classification


@router.delete("/{classification_id}")
async def delete_classification(
        classification_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Удаляет классификацию по её ID.

    Входные данные:
    - **classification_id**: UUID классификации из пути URL.

    Возвращает:
    - Сообщение об успешном удалении.
    """
    result = await db.execute(
        select(ClassificationModel)
        .where(ClassificationModel.classification_id == classification_id)
    )
    classification = result.scalars().first()
    if not classification:
        raise HTTPException(status_code=404, detail="Classification not found")

    await db.delete(classification)
    await db.commit()
    return {"message": "Classification deleted successfully"}