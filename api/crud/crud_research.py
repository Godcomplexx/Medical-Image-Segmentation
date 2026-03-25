import uuid
from datetime import datetime
from sqlalchemy import select, cast, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.models import (
    Research as ResearchModel,
    User as UserModel,
    Patient as PatientModel,
    Status as StatusModel,
    ClassesArea as ClassesAreaModel,
    Localization as LocalizationModel,
    Classification as ClassificationModel
)
from schemas.schemas import ResearchCreateSchema
from . import crud_status, crud_patient


async def get_research(db: AsyncSession, research_id: uuid.UUID) -> ResearchModel | None:
    """Получает один проект из БД по его ID вместе со всеми связанными данными."""
    result = await db.execute(
        select(ResearchModel)
        .where(ResearchModel.id == research_id)
        .options(
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.localization),
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.classification),
            selectinload(ResearchModel.user),
            selectinload(ResearchModel.patient),
            selectinload(ResearchModel.status)
        )
    )
    return result.scalars().first()


async def get_researches(db: AsyncSession):
    """Получает все проекты"""
    result = await db.execute(
        select(ResearchModel).options(
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.localization),
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.classification),
            selectinload(ResearchModel.user),
            selectinload(ResearchModel.patient),
            selectinload(ResearchModel.status)
        )
    )
    return result.scalars().all()


async def get_research_by_short_id(db: AsyncSession, short_id: str) -> ResearchModel | None:
    """Ищет проект по префиксу UUID (первые символы UUID).

    Примечание: работает на PostgreSQL, где UUID можно привести к строке.
    """
    result = await db.execute(
        select(ResearchModel)
        .where(cast(ResearchModel.id, String).like(f"{short_id}%"))
        .options(
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.localization),
            selectinload(ResearchModel.classes_areas).selectinload(ClassesAreaModel.classification),
            selectinload(ResearchModel.user),
            selectinload(ResearchModel.patient),
            selectinload(ResearchModel.status)
        )
        .limit(1)
    )
    return result.scalars().first()

async def create_research(db: AsyncSession, research_data: ResearchCreateSchema, user: UserModel, research_id: uuid.UUID) -> ResearchModel:
    """Создает новую запись проекта в базе данных."""
    research_dict = research_data.model_dump(exclude={"user_email", "name"})
    research_dict["user_id"] = user.id

    research_dict["id"] = research_id
    research_short_name = str(research_id)[:5]
    research_dict["name"] = f"Исследование {research_short_name}"

    # Получаем или создаем пациента
    patient = await crud_patient.get_or_create_patient(db, research_data.patient_id)

    # Определяем статус: либо из запроса, либо по умолчанию "Загружен"
    if research_data.status_id:
        status = await crud_status.get_status_by_id(db, research_data.status_id)
        if not status:
            return None  # Статус не найден, обработается в роуте как 404
    else:
        status = await crud_status.get_status_by_name(db, "Загружен")
        if not status:
            return None  # Статус по умолчанию не найден

    research_dict["status_id"] = status.id

    if not research_dict.get("name"):
        date_str = datetime.now().strftime("%Y%m%d")
        research_dict["name"] = f"Исследование {user.email} - {patient.code} - {date_str}"

    if research_dict.get("upload_date") is None:
        research_dict["upload_date"] = datetime.now().replace(microsecond=0)

    # Создаем экземпляр модели SQLAlchemy и сохраняем его в БД
    db_project = ResearchModel(**research_dict)
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)

    # Возвращаем созданный объект со всеми подгруженными связями
    return await get_research(db, db_project.id)


async def update_research_status(db: AsyncSession, research: ResearchModel, status: StatusModel) -> ResearchModel:
    """Обновляет только статус существующего проекта."""
    research.status_id = status.id
    await db.commit()
    await db.refresh(research)

    # Возвращаем обновленный проект
    return await get_research(db, research.id)


async def delete_research(db: AsyncSession, research: ResearchModel):
    await db.delete(research)
    await db.commit()
    return True