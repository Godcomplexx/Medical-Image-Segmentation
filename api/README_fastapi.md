# 🏥 Внешний клинический API

Имитирует работу клинической системы для тестирования интеграции с системой сегментации.

## 🚀 Запуск

### 1. Установить зависимости:
```bash
cd api
pip install -r requirements.txt
```

### 2. Запустить API:
```bash
python run.py
```

API будет доступен по адресу: http://localhost:8080

## 📋 Эндпоинты

### `GET /` - Главная страница
Информация об API и доступных эндпоинтах.

### `GET /folders` - Список папок
Показывает все папки в `fileStorage` с информацией о количестве файлов и размере.

**Пример ответа:**
```json
{
  "folders": [
    {
      "name": "test_data",
      "path": "math_app/fileStorage/test_data",
      "file_count": 5,
      "size_mb": 12.5
    }
  ],
  "total_folders": 1
}
```

### `POST /process` - Отправить на обработку
Отправляет папку в очередь Celery для обработки.

**Параметры:**
- `folder_name` (str) - имя папки в fileStorage
- `user_id` (int, optional) - ID пользователя (по умолчанию 123)

**Пример запроса:**
```bash
curl -X POST "http://localhost:8080/process" \
     -F "folder_name=test_data" \
     -F "user_id=123"
```

**Пример ответа:**
```json
{
  "status": "queued",
  "task_id": "abc123-def456-ghi789",
  "folder_name": "test_data",
  "user_id": 123,
  "file_count": 5,
  "message": "Задача отправлена в очередь для обработки папки 'test_data'"
}
```

### `GET /task/{task_id}` - Статус задачи
Проверяет статус выполнения задачи в Celery.

**Пример ответа:**
```json
{
  "task_id": "abc123-def456-ghi789",
  "status": "SUCCESS",
  "ready": true,
  "successful": true,
  "failed": false,
  "result": {
    "user_id": 123,
    "folder_name": "test_data",
    "status": "completed",
    "result": {...}
  }
}
```

### `GET /health` - Проверка здоровья
Проверяет доступность fileStorage и RabbitMQ.

## 🔄 Полный цикл тестирования

1. **Запустить все сервисы:**
   ```bash
   # Основные сервисы
   docker compose up -d
   
   # MQ система
   cd mq && docker compose up -d
   
   # Внешний API
   cd api && python run.py
   ```

2. **Создать тестовые данные:**
   ```bash
   mkdir -p math_app/fileStorage/test_data
   echo "test file" > math_app/fileStorage/test_data/test.txt
   ```

3. **Проверить доступные папки:**
   ```bash
   curl http://localhost:8080/folders
   ```

4. **Отправить на обработку:**
   ```bash
   curl -X POST "http://localhost:8080/process" \
        -F "folder_name=test_data"
   ```

5. **Проверить статус:**
   ```bash
   curl http://localhost:8080/task/{task_id}
   ```

## 🎯 Что происходит

1. **Clinical API** → отправляет задачу `tasks.calculate` в очередь `math_tasks`
2. **Math Worker** → получает задачу и вызывает `math_app`
3. **Math App** → обрабатывает папку через `process_survey()`
4. **Math Worker** → отправляет результат в `db_worker`
5. **DB Worker** → сохраняет результат в JSON файл

## 🌐 Swagger документация

Доступна по адресу: http://localhost:8080/docs
