# Внешний клинический API

Имитирует работу клинической системы для тестирования интеграции с системой сегментации.

### Запустить API:
```bash
python run.py
```

## Что происходит

1. **Clinical API** → отправляет задачу `tasks.calculate` в очередь `math_tasks`
2. **Math Worker** → получает задачу и вызывает `math_app`
3. **Math App** → обрабатывает папку через `process_survey()`
4. **Math Worker** → отправляет результат в `db_worker`
5. **DB Worker** → сохраняет результат в JSON файл

## 🌐 Swagger документация

Доступна по адресу: http://localhost:8080/docs

## Эндпоинт на вызов который нужно взять 

```python 
    @app.post("/process")
    async def process_folder(folder_name: str = Form(...)):
        """
        Отправить папку на обработку через Celery очередь.
        
        Args:
            folder_name: Имя папки в fileStorage для обработки
        """
        try:
            print(f"[Clinical API] Отправляем задачу для папки: {folder_name}")
            
            # Отправляем задачу в Celery очередь БЕЗ проверки папки
            # Проверка будет в math_app при получении задачи
            task = celery_client.send_task(
                "tasks.calculate",
                args=[folder_name],  
                queue="math_tasks"
            )
            
            return {
                "status": "queued",
                "task_id": task.id,
                "folder_name": folder_name,
                "message": f"Задача отправлена в очередь для обработки папки '{folder_name}'"
            }
            
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Ошибка при отправке задачи: {str(e)}"
            )
    
    
    ## 
    
    from celery import Celery
    import traceback
    
    app = FastAPI(
        title="Clinical API",
        description="Внешний API для отправки задач в систему сегментации",
        version="1.0.0"
    )
```

# Настройки Celery для отправки задач = они есть в env, надо передавать через переменую окружения 
```text
BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
```

```python
# Создаем Celery клиент
celery_client = Celery(
    "clinical_api",
    broker=BROKER_URL,
    backend=BACKEND_URL
)

# Настройки клиента
celery_client.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "tasks.calculate": {"queue": "math_tasks"},
        "tasks.parse_and_store": {"queue": "db_tasks"},
    }
)

# Все остальное в main не нужно
```