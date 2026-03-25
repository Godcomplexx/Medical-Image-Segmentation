import asyncio
import uuid
from datetime import datetime
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Base, Localization, ClassesArea, User, Patient, Project, Status, Classification
from db import engine, AsyncSessionLocal


async def check_connection():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            print("Подключение к PostgreSQL успешно")
            return True
    except Exception as e:
        print(f" Ошибка подключения: {e}")
        return False


async def create_tables():
    if not await check_connection():
        return

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            print("Таблицы успешно созданы")
    except Exception as e:
        print(f"Ошибка при создании таблиц: {e}")

async def create_sample_data():
    async with AsyncSessionLocal() as session:
        # Создаем область
        loc = Localization(
            zone_name="Main Zone"
        )
        session.add(loc)
        await session.flush()

        classification = Classification(
            classification_id=uuid.uuid4(),
            class_name="Sample Classification"
        )
        session.add(classification)
        await session.flush()

        # Создаем классификации
        class_area = ClassesArea(
            localization_id=loc.zone_id,
            area=100.5,
            class_id=classification.classification_id

        session.add(class_area)
        await session.flush()
        
        # Создаем пользователя
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            password="securepassword"
        )
        session.add(user)
        await session.flush()
        
        # Создаем пациента
        patient = Patient()
        session.add(patient)
        await session.flush()

        # Создаём статус
        status = Status(
            status_name="загружено"
        )
        session.add(status)
        await session.flush()
        
        # Создаем исследование
        project = Project(
            user_id=user.id,
            patient_id=patient.id,
            upload_date=datetime.now(),
            classes_area_id=class_area.id,
            name="First Project",
            status_id=status.id

        )
        session.add(project)
        
        await session.commit()

async def get_projects_with_details():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Project)
            .options(
                selectinload(Project.classes_area).selectinload(ClassesArea.localization),
                selectinload(Project.user),
                selectinload(Project.patient),
                selectinload(Project.status)
            )
        )

        projects = result.scalars().unique().all()
          
        for project in projects:
            print(f"Project: {project.name}")
            print(f"Zone: {project.classes_area.localization.zone_name}")
            print(f"Area: {project.classes_area.area}")
            print(f"User: {project.user.email}")
            print(f"Patient Code: {project.patient.code}")
            print(f"Status: {project.status.status_name}")
            print("------")

async def main():
    try:
        await create_tables()
        await create_sample_data()
        await get_projects_with_details()
    finally:
        await engine.dispose()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Приложение остановлено")

