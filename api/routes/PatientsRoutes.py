from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from typing import List

from db.db import get_db
from models.models import User as UserModel
from schemas.schemas import Patient
from crud import crud_patient
from security import get_current_user

router = APIRouter(
    prefix="/patients",
    tags=["patients"]
)


async def _require_patient_owner(
    db: AsyncSession, patient_id: uuid.UUID, current_user: UserModel
) -> None:
    """Проверяет, что пациент связан с текущим пользователем через его исследования."""
    if not await crud_patient.user_owns_patient(db, user_id=current_user.id, patient_id=patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this patient.",
        )


@router.post("/", response_model=Patient)
async def create_patient(
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Создает новую запись пациента со сгенерированными UUID.

    Входные данные:
    - Нет. Тело запроса пустое.

    Возвращает:
    - Созданный объект пациента.
    """
    return await crud_patient.create_patient(db)


@router.get("/", response_model=List[Patient])
async def read_patients(
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Получает список пациентов, связанных с текущим пользователем.

    Возвращает:
    - Список объектов пациентов.
    """
    return await crud_patient.get_patients_for_user(db, user_id=current_user.id)


@router.get("/{patient_id}", response_model=Patient)
async def read_patient(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Получает одного пациента по его ID. Доступно только владельцу.

    Входные данные:
    - **patient_id**: UUID пациента из пути URL.

    Возвращает:
    - Объект пациента или ошибку 404.
    """
    await _require_patient_owner(db, patient_id, current_user)
    patient = await crud_patient.get_patient_by_id(db, patient_id=patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
        Удаляет пациента по его ID. Доступно только владельцу.

        Входные данные:
        - **patient_id**: UUID пациента из пути URL.

        Возвращает:
        - Ничего.
    """
    await _require_patient_owner(db, patient_id, current_user)
    deleted = await crud_patient.delete_patient_by_id(db, patient_id=patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")

    return None  # Возвращаем пустой ответ