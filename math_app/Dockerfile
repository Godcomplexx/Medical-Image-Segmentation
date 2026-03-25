FROM python:3.10-slim

# Установить libGL для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HTTP_TIMEOUT=100 \
    POETRY_INSTALLER_PARALLEL=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" -i https://pypi.tuna.tsinghua.edu.cn/simple


WORKDIR /math_app

COPY pyproject.toml poetry.lock* /math_app/
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY ./src ./src
COPY ./fileStorage ./fileStorage
COPY ./config ./config
COPY ./api ./api
COPY ./best.pt .
COPY ./logs ./logs


CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]