from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List

from db.db import get_db
from models.models import Patient as PatientModel
from schemas.schemas import Patient
from crud import crud_patient

router = APIRouter(
    prefix="/patients",
    tags=["patients"]
)


@router.post("/", response_model=Patient)
async def create_patient(db: AsyncSession = Depends(get_db)):
    """
    Создает новую запись пациента со сгенерированными UUID.

    Входные данные:
    - Нет. Тело запроса пустое.

    Возвращает:
    - Созданный объект пациента.
    """
    db_patient = PatientModel()
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    return db_patient


@router.get("/", response_model=List[Patient])
async def read_patients(db: AsyncSession = Depends(get_db)):
    """
    Получает список всех пациентов.

    Входные данные:
    - Нет.

    Возвращает:
    - Список объектов пациентов.
    """
    result = await db.execute(select(PatientModel))
    return result.scalars().all()


@router.get("/{patient_id}", response_model=Patient)
async def read_patient( patient_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Получает одного пациента по его ID.

    Входные данные:
    - **patient_id**: UUID пациента из пути URL.

    Возвращает:
    - Объект пациента или ошибку 404.
    """
    result = await db.execute(
        select(PatientModel)
        .where(PatientModel.id == patient_id)
    )
    patient = result.scalars().first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
        Удаляет пациента по его ID.

        Входные данные:
        - **patient_id**: UUID пациента из пути URL.

        Возвращает:
        - Ничего.
    """
    # Вызываем CRUD-функцию для удаления
    deleted = await crud_patient.delete_patient_by_id(db, patient_id=patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")

    return None  # Возвращаем пустой ответ