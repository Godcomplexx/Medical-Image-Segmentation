from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List

from db.db import get_db
from models.models import Status as StatusModel
from schemas.schemas import StatusSchema, Status

router = APIRouter(
    prefix="/statuses",
    tags=["statuses"]
)


@router.post("/", response_model=Status)
async def create_status( status: StatusSchema, db: AsyncSession = Depends(get_db)):
    """
    Создает новый статус.

    Входные данные:
    - **status**: Схема `StatusSchema` с именем статуса из тела POST-запроса.

    Возвращает:
    - Созданный объект статуса.
    """
    db_status = StatusModel(**status.model_dump())
    db.add(db_status)
    await db.commit()
    await db.refresh(db_status)
    return db_status


@router.get("/", response_model=List[Status])
async def read_statuses(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех возможных статусов в системе.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов статусов.
    """
    result = await db.execute(select(StatusModel))
    return result.scalars().all()


@router.get("/{status_id}", response_model=Status)
async def read_status( status_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Получает один статус по его ID.

    Входные данные:
    - **status_id**: UUID статуса из пути URL.

    Возвращает:
    - Объект статуса или ошибку 404.
    """
    result = await db.execute(
        select(StatusModel)
        .where(StatusModel.id == status_id)
    )
    status = result.scalars().first()
    if not status:
        raise HTTPException(status_code=404, detail="Status not found")
    return status


@router.put("/{status_id}", response_model=Status)
async def update_status( status_id: uuid.UUID, status: StatusSchema, db: AsyncSession = Depends(get_db)):
    """
    Обновляет существующий статус.

    Входные данные:
    - **status_id**: UUID статуса из пути URL.
    - **status**: Схема `StatusSchema` с новыми данными из тела PUT-запроса.

    Возвращает:
    - Обновленный объект статуса.
    """
    result = await db.execute(
        select(StatusModel)
        .where(StatusModel.id == status_id)
    )
    db_status = result.scalars().first()
    if not db_status:
        raise HTTPException(status_code=404, detail="Status not found")

    for field, value in status.model_dump().items():
        setattr(db_status, field, value)

    db.add(db_status)
    await db.commit()
    await db.refresh(db_status)
    return db_status


@router.delete("/{status_id}")
async def delete_status( status_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Удаляет статус по его ID.

    Входные данные:
    - **status_id**: UUID статуса из пути URL.

    Возвращает:
    - Сообщение об успешном удалении.
    """
    result = await db.execute(
        select(StatusModel)
        .where(StatusModel.id == status_id)
    )
    status = result.scalars().first()
    if not status:
        raise HTTPException(status_code=404, detail="Status not found")

    await db.delete(status)
    await db.commit()
    return {"message": "Status deleted successfully"}