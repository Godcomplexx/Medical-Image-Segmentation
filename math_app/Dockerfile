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
    PIP_DEFAULT_TIMEOUT=600 \
    POETRY_HTTP_TIMEOUT=600 \
    POETRY_INSTALLER_PARALLEL=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /math_app

# Установить PyTorch CPU-only (GPU не доступен в этом контейнере)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY math_app/pyproject.toml math_app/poetry.lock* /math_app/
RUN pip install --upgrade pip \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY math_app/src ./src
COPY math_app/fileStorage ./fileStorage
COPY math_app/config ./config
COPY math_app/api ./api
COPY math_app/best.pt .
COPY shared ./shared
RUN mkdir -p ./logs


CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]
