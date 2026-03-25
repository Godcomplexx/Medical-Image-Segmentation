# Base image for Celery workers
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install minimal runtime deps (no build tools)
RUN pip install --no-cache-dir \
        "celery[rabbitmq]>=5.3,<6.0" \
        "pydantic>=2.7,<3.0" \
        "pydantic-settings>=2.0,<3.0" \
        "importlib-metadata>=7.0,<8.0" \
        "requests>=2.31.0,<3.0.0" \
        flower==2.0.*

# Copy only relevant sub-tree to keep image small
# Copy only source directories to avoid name clashes with stdlib modules
COPY workers/ /app/workers/
COPY interfaces/ /app/interfaces/
COPY config/ /app/config

# Ensure Python path includes app directory
ENV PYTHONPATH="/app:/app/workers:${PYTHONPATH}"

# Default command is overridden by docker-compose
CMD ["python", "-m", "celery", "--version"]
