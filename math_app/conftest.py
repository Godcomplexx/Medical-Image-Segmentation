import sys
from pathlib import Path

# Делаем пакет `src` импортируемым из тестов.
sys.path.insert(0, str(Path(__file__).resolve().parent))
