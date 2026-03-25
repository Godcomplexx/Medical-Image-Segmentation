FROM deepmi/fastsurfer:gpu-latest

USER root

WORKDIR /fastsurfer_app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml ./
COPY poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY main.py ./

ENTRYPOINT []

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]