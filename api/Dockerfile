# Используем образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей в рабочую директорию
COPY ./api/requirements.txt ./api/requirements.lock ./

# Устанавливаем зависимости из lock-файла (полностью зафиксированный набор) для воспроизводимости
RUN pip install --no-cache-dir -r requirements.lock

# Копируем весь код из папки api в рабочую директорию
COPY ./api .

# Применяем миграции БД (Alembic), затем запускаем приложение (production: без --reload).
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]