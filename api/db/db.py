import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL_FROM_ENV = os.getenv("DATABASE_URL")

# Если переменная окружения задана, используем ее. Иначе — используем значение по умолчанию.
if DATABASE_URL_FROM_ENV:
    DATABASE_URL = DATABASE_URL_FROM_ENV
    print(f"INFO: Используется DATABASE_URL из переменной окружения: {DATABASE_URL}")
else:
    DATABASE_URL = "postgresql+asyncpg://postgres:3110@localhost:5432/patient"
    print(f"INFO: Переменная окружения DATABASE_URL не найдена. Используется значение по умолчанию: {DATABASE_URL}")

# Создаем асинхронный "движок" для взаимодействия с БД. echo=True выводит все SQL-запросы в консоль.
engine = create_async_engine(DATABASE_URL, echo=True)

# Создаем фабрику асинхронных сессий.
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Базовый класс для всех моделей
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session