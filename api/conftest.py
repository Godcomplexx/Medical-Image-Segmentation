import os
import sys
from pathlib import Path

# Делаем модули api (main, security, routes, ...) импортируемыми из тестов.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Значения по умолчанию для тестов — должны быть заданы ДО импорта security/db.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("INTERNAL_API_TOKEN", "test-internal-token")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
