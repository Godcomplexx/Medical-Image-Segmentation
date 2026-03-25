import uuid
import random
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.models import Patient as PatientModel

def generate_patient_code():
    number = random.randint(0, 99999)
    return str(number).zfill(5)

async def get_patient_by_code(db: AsyncSession, code:str) -> PatientModel | None:
    result = await db.execute(select(PatientModel).where(PatientModel.code == code))
    return result.scalars().first()


async def get_patient_by_id(db: AsyncSession, patient_id: uuid.UUID) -> PatientModel | None:
    """Ищет пациента по его основному ID."""
    result = await db.execute(select(PatientModel).where(PatientModel.id == patient_id))
    return result.scalars().first()


async def get_patients(db: AsyncSession) -> List[PatientModel]:
    """Возвращает список всех пациентов."""
    result = await db.execute(select(PatientModel))
    return result.scalars().all()


async def create_patient(db: AsyncSession) -> PatientModel:
    """Создает нового пациента (используется для эндпоинта POST /patients)."""

    new_patient = PatientModel()
    db.add(new_patient)
    await db.flush()

    while True:
        new_code = generate_patient_code()
        existing = await get_patient_by_code(db, code=new_code)
        if not existing:
            new_patient.code = new_code
            break

    await db.commit()
    await db.refresh(new_patient)
    return new_patient


async def get_or_create_patient(db: AsyncSession, patient_id: uuid.UUID) -> PatientModel:
    result = await db.execute(select(PatientModel).where(PatientModel.id == patient_id))
    patient = result.scalars().first()

    if patient:
        return patient

    new_patient = PatientModel(id=patient_id)

    while True:
        # 1. Генерируем новый код
        new_code = generate_patient_code()
        # 2. Проверяем, не занят ли этот код
        existing_patient = await get_patient_by_code(db, code=new_code)

        # 3. Если код свободен, выходим из цикла
        if not existing_patient:
            new_patient.code = new_code
            break

    # Добавляем и сохраняем пациента в БД
    db.add(new_patient)
    await db.flush()
    await db.refresh(new_patient)

    return new_patient

async def delete_patient_by_id(db: AsyncSession, patient_id: uuid.UUID) -> bool:
    """Находит пациента по ID и удаляет его."""
    result = await db.execute(select(PatientModel).where(PatientModel.id == patient_id))
    patient = result.scalars().first()
    if patient:
        await db.delete(patient)
        await db.commit()
        return True
    return False