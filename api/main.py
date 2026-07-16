import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text

from routes.ClassAreaRoute import router as class_area_router
from routes.ClassificationRoute import router as classification_router
from routes.LocalizationRoute import router as localization_router
from routes.PatientsRoutes import router as patient_router
from routes.ResearchRoute import router as research_router
from routes.StatusesRoutes import router as status_router
from routes.UsersRoute import router as user_router

from db.db import engine, AsyncSessionLocal
from models.models import Status as StatusModel
from constants import ALL_STATUSES

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаем директорию для локального хранения файлов, если она не существует
    os.makedirs("file_storage", exist_ok=True)
    print("INFO:     Директория 'file_storage' готова к работе.")

    # Схема БД управляется Alembic (alembic upgrade head выполняется при старте контейнера).
    # Здесь только проверяем доступность БД и при недоступности — останавливаем запуск API.
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        # Без БД приложение работать не может — падаем явно, не запускаем API.
        raise RuntimeError(f"База данных недоступна, запуск API прерван: {e}") from e

    # Заполняем справочник статусов (идемпотентно).
    await initialize_application_data()
    print("INFO:     Соединение с БД установлено, статусы инициализированы.")

    yield   # Старт работы приложения

    await engine.dispose()  # Закрытие соединения с БД


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173", # Стандартный порт для Vite/React
    "http://localhost:3000", # Стандартный порт для Create React App
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # Разрешаем запросы с этих источников
    allow_credentials=True,   # Разрешаем передачу cookie
    allow_methods=["*"],      # Разрешаем все методы (GET, POST, PUT, DELETE)
    allow_headers=["*"],      # Разрешаем все заголовки
)

# Подключаем роутеры приложения
app.include_router(classification_router)
app.include_router(class_area_router)
app.include_router(localization_router)
app.include_router(patient_router)
app.include_router(research_router)
app.include_router(status_router)
app.include_router(user_router)


async def initialize_application_data():
    async with AsyncSessionLocal() as session:
        # Список статусов, которые должны быть в системе по умолчанию
        statuses_to_create = ALL_STATUSES
        # Проверяем каждый статус и создаем, если его нет
        for status_name in statuses_to_create:
            exists_query = select(StatusModel).where(StatusModel.status_name == status_name)
            existing_status = (await session.execute(exists_query)).scalars().first()
            if not existing_status:
                new_status = StatusModel(status_name=status_name)
                session.add(new_status)
        await session.commit()  # Сохраняем все изменения в БД
        print(f"INFO:     {len(statuses_to_create)} статусов проверено и создано при необходимости.")