import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.models import (
    ClassesArea as ClassesAreaModel,
    Localization as LocalizationModel,
    Classification as ClassificationModel,
    Research as ResearchModel
)
from schemas.schemas import ClassesAreaCreateSchema
from typing import List


async def get_or_create_localization_by_name(db: AsyncSession, name: str) -> LocalizationModel:
    """Ищет локализацию по имени. Если не находит, создает новую."""
    result = await db.execute(select(LocalizationModel).where(LocalizationModel.zone_name == name))
    instance = result.scalars().first()
    if not instance:
        instance = LocalizationModel(zone_name=name)
        db.add(instance)
        await db.flush()  # Получаем ID до коммита
        await db.refresh(instance)
    return instance


async def get_or_create_classification_by_name(db: AsyncSession, name: str) -> ClassificationModel:
    """Ищет классификацию по имени. Если не находит, создает новую."""
    result = await db.execute(select(ClassificationModel).where(ClassificationModel.class_name == name))
    instance = result.scalars().first()
    if not instance:
        instance = ClassificationModel(class_name=name)
        db.add(instance)
        await db.flush()
        await db.refresh(instance)
    return instance


async def add_classes_area_to_research(db: AsyncSession, data:List[List[str]], research_id: uuid.UUID) -> List[ClassesAreaModel]:
    """
    Заменяет области классификации исследования на переданные.

    Идемпотентность: перед добавлением удаляются уже существующие области
    исследования, чтобы повторная обработка (Celery redelivery / повторный
    запуск) не приводила к дублированию строк.
    """
    created_areas = []
    created_area_ids = []
    # 0. Удаляем ранее сохранённые области этого исследования (идемпотентность)
    await db.execute(
        delete(ClassesAreaModel).where(ClassesAreaModel.research_id == research_id)
    )
    # 1. Итерируемся по каждой строке из входного массива
    for item in data:
        # Распаковываем данные из строки
        classification_name, localization_name, area_str = item

        # 2. Находим или создаем связанные сущности по именам
        localization = await get_or_create_localization_by_name(db, name=localization_name)
        classification = await get_or_create_classification_by_name(db, name=classification_name)
        area_float = float(area_str)

        # 3. Создаем новый объект ClassesArea для этой строки
        db_classes_area = ClassesAreaModel(
            area=area_float,
            localization_id=localization.zone_id,
            class_id=classification.classification_id,
            research_id=research_id
        )
        db.add(db_classes_area)
        created_areas.append(db_classes_area)

    # 4. Сохраняем все созданные объекты одним коммитом
    await db.commit()

    created_area_ids = [area.id for area in created_areas]
    # 5. Выполняем ОДИН финальный запрос для получения всех созданных объектов
    if not created_area_ids:
        return []

    final_query = (
        select(ClassesAreaModel)
        .where(ClassesAreaModel.id.in_(created_area_ids))
        .options(
            selectinload(ClassesAreaModel.localization),
            selectinload(ClassesAreaModel.classification)
        )
    )
    result = await db.execute(final_query)
    return result.scalars().all()