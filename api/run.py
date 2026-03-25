#!/usr/bin/env python3
"""
Скрипт для запуска внешнего клинического API.
"""

import uvicorn
import sys
from pathlib import Path

# Добавляем папку api в Python path
api_dir = Path(__file__).parent
sys.path.insert(0, str(api_dir))

# Импортируем app напрямую
from main import app

if __name__ == "__main__":
    print("🏥 Запуск внешнего клинического API...")
    print("📋 Доступные эндпоинты:")
    print("   GET  http://localhost:8081/          - Главная страница")
    print("   GET  http://localhost:8081/folders   - Список папок в fileStorage")
    print("   POST http://localhost:8081/process   - Отправить папку на обработку")
    print("   GET  http://localhost:8081/task/{id} - Статус задачи")
    print("   GET  http://localhost:8081/health    - Проверка здоровья")
    print("   GET  http://localhost:8081/docs      - Swagger документация")
    print()
    print("🚀 Запуск сервера...")
    
    # Запускаем с импортированным app
    uvicorn.run(
        app,  # Используем импортированный объект
        host="0.0.0.0", 
        port=8081,
        reload=False,  
        log_level="info"
    )
