import os
import shutil
import traceback
import uuid
from typing import List, Literal

import httpx
from celery import Celery
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.db import get_db
from schemas.schemas import ResearchCreateSchema, Research, ResearchStatusUpdateByIdSchema, ResearchFailSchema, ClassesAreaCreateSchema, ClassesArea
from crud import crud_research, crud_user, crud_status, crud_classes_area
from models.models import User as UserModel, Research as ResearchModel
from security import get_current_user, get_principal, Principal
from constants import (
    STATUS_UPLOADED,
    STATUS_QUEUED,
    STATUS_PROCESSING,
    STATUS_READY,
    STATUS_FAILED,
    InvalidStatusTransition,
)

FILESTORAGE_SERVICE_BASE_URL = os.getenv("FILESTORAGE_SERVICE_BASE_URL", "http://localhost:8003")
BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

router = APIRouter(
    prefix="/researches",
    tags=["researches"]
)

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
        "tasks.save_json": {"queue": "db_tasks"},
    }
)



def _require_research_owner(research: ResearchModel, current_user: UserModel) -> None:
    """Запрещает доступ к чужим исследованиям."""
    if research.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this research.",
        )


def _enforce_owner_unless_internal(
    principal: Principal, research: ResearchModel
) -> None:
    """Для пользователя проверяет владельца; внутренний сервис пропускается."""
    if not principal.is_internal:
        _require_research_owner(research, principal.user)


@router.post("/upload_zip", response_model=Research, status_code=status.HTTP_201_CREATED)
async def upload_zip_and_create_research(
        patient_id: uuid.UUID = Form(...),
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
        user_email: str = Form(None),  # принимается для совместимости, но владелец берётся из токена
):
    """
    Главный эндпоинт для создания проекта.
    1. Принимает zip-файл и email пользователя.
    2. Проксирует файл во внешний сервис обработки (filestorage).
    3. В случае успеха от внешнего сервиса, создает запись о проекте в своей БД.

    Входные данные (multipart/form-data):
    - **user_email**: Email существующего пользователя.
    - **file**: Zip-архив с данными для исследования.

    Возвращает:
    - Созданный объект проекта со статусом 201 Created.
    """
    # 1. Проверка конфигурации и входных данных
    if not FILESTORAGE_SERVICE_BASE_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FILESTORAGE_SERVICE_BASE_URL is not configured."
        )
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .zip files are accepted."
        )

    # 2. Владелец исследования — текущий аутентифицированный пользователь
    user = current_user
    # 3. Генерация ID и имени исследования
    research_id = uuid.uuid4()
    research_short_name = str(research_id)[:5]
    renamed_folder_name = f"{research_short_name}.zip"
    files_for_upload = {'file': (renamed_folder_name, file.file, 'application/zip')}

    # 4. Отправка данных во внешний сервис с помощью httpx
    upload_url = f"{FILESTORAGE_SERVICE_BASE_URL}/upload_zip"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(upload_url, files=files_for_upload, timeout=30.0)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"File storage service failed with status {response.status_code}: {response.text}"
                )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not connect to file storage service: {e}"
            )

    # 5. Если внешний сервис ответил успешно, создаем проект в нашей БД
    research_data = ResearchCreateSchema(
        user_email=current_user.email,
        patient_id=patient_id,
        name=f"Исследование {research_short_name}",
    )
    created_research = await crud_research.create_research(db, research_data=research_data, user=user, research_id=research_id)
    if created_research is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create research in DB after successful file upload."
        )

    return created_research


@router.post("/{research_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def process_folder(
        research_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Отправить исследование на обработку через Celery очередь.

    Args:
        research_id: UUID исследования из пути URL.
    """
    research = await crud_research.get_research(db, research_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _require_research_owner(research, current_user)

    # Идемпотентность: нельзя ставить в очередь исследование, которое уже в работе.
    if not research.status or research.status.status_name != STATUS_UPLOADED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Research can only be queued from the uploaded state "
                f"(current: {research.status.status_name if research.status else 'unknown'})."
            ),
        )

    # Имя папки в fileStorage — детерминированный префикс UUID исследования.
    folder_name = str(research.id)[:5]
    print(f"[Clinical API] Отправляем задачу для папки: {folder_name}")

    try:
        # Обновляем статус исследования на "В очереди"
        status_obj = await crud_status.get_status_by_name(db, STATUS_QUEUED)
        if not status_obj:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Queued status is not configured.",
            )
        research = await crud_research.update_research_status(
            db, research=research, status=status_obj
        )

        # Отправляем задачу в Celery очередь, передаём короткое имя и полный research_id
        task = celery_client.send_task(
            "tasks.calculate",
            args=[folder_name, str(research.id)],
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
        failed_status = await crud_status.get_status_by_name(db, STATUS_FAILED)
        if failed_status and research.status:
            try:
                await crud_research.set_research_failed(
                    db,
                    research=research,
                    status=failed_status,
                    error_message=f"Failed to enqueue processing task: {e}",
                )
            except InvalidStatusTransition:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при отправке задачи: {str(e)}"
        )

@router.post("/{research_id}/classes_areas", response_model=List[ClassesArea], status_code=status.HTTP_201_CREATED)
async def add_classes_area_to_research(
        research_id: str,
        data: List[List[str]],
        db: AsyncSession = Depends(get_db),
        principal: Principal = Depends(get_principal),
):
    """
    Добавляет новую область классификации к существующему исследованию.
    Если локализация или классификация с указанным именем не существуют, они будут созданы.

    Вызывается DB-воркером (внутренний токен) после расчёта, либо владельцем.

    Входные данные:
    - **research_id**: UUID исследования из пути URL.
    - **classes_area_data**: Схема `ClassesAreaCreateSchema` из тела POST-запроса.

    Возвращает:
    - Созданный объект области классификации (ClassesArea).
    """
    # Поддержка короткого ID (префикс UUID)
    try:
        research_uuid = uuid.UUID(research_id)
        research = await crud_research.get_research(db, research_id=research_uuid)
    except ValueError:
        # Не UUID — пробуем как короткий ID
        research = await crud_research.get_research_by_short_id(db, short_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _enforce_owner_unless_internal(principal, research)

    # Валидация входных данных
    for item in data:
        if len(item) != 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid item format: {item}. Each item must contain 3 elements."
            )
        try:
            float(item[2])
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid area value: '{item[2]}'. It must be a valid number."
            )

    # Вызываем CRUD-функцию для выполнения всей логики
    created_areas = await crud_classes_area.add_classes_area_to_research(
        db=db, data=data, research_id=research.id
    )
    # После успешного сохранения площадей — обновляем статус исследования на "Готов"
    status_obj = await crud_status.get_status_by_name(db, STATUS_READY)
    if status_obj:
        await crud_research.update_research_status(db, research=research, status=status_obj)
    return created_areas


@router.post("/{research_id}/fail", response_model=Research)
async def mark_research_failed(
        research_id: str,
        payload: ResearchFailSchema,
        db: AsyncSession = Depends(get_db),
        principal: Principal = Depends(get_principal),
):
    """Переводит исследование в статус "Ошибка" и сохраняет текст ошибки.

    Вызывается math-воркером (внутренний токен) при неустранимом сбое обработки,
    либо владельцем исследования.
    """
    try:
        research_uuid = uuid.UUID(research_id)
        research = await crud_research.get_research(db, research_id=research_uuid)
    except ValueError:
        research = await crud_research.get_research_by_short_id(db, short_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _enforce_owner_unless_internal(principal, research)

    status_obj = await crud_status.get_status_by_name(db, STATUS_FAILED)
    if not status_obj:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed status is not configured.")
    try:
        return await crud_research.set_research_failed(
            db, research=research, status=status_obj, error_message=payload.error_message
        )
    except InvalidStatusTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

@router.get("/{research_id}/download/{file_type}")
async def download_processed_file(
        research_id: uuid.UUID,
        file_type: Literal["t1", "t2", "prediction", "parcellation"],
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Прокси-эндпоинт для скачивания обработанных файлов из внешнего сервиса.
    Клиент обращается к этому API, а API "за спиной" обращается к
    сервису обработки и передает (стримит) ответ клиенту.

    Входные данные:
    - **research_id**: UUID проекта из пути URL.
    - **file_type**: Тип запрашиваемого файла ('t1', 't2', и т.д.) из пути URL.

    Возвращает:
    - Потоковый ответ (`StreamingResponse`), содержащий запрошенный файл.
    """
    # 1. Проверки
    if not FILESTORAGE_SERVICE_BASE_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FILESTORAGE_SERVICE_BASE_URL is not configured."
        )
    research = await crud_research.get_research(db, research_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _require_research_owner(research, current_user)

    # 2. Формирование URL для запроса к внешнему сервису
    study_name = str(research_id)[:5]
    download_url = f"{FILESTORAGE_SERVICE_BASE_URL}/download/{file_type}/{study_name}"

    client = httpx.AsyncClient(timeout=300.0)

    try:
        req = client.build_request("GET", download_url)
        r = await client.send(req,stream=True)
        if r.status_code != 200:
            error_detail = await r.aread()
            raise HTTPException(status_code=r.status_code, detail=error_detail.decode())

        return StreamingResponse(
            r.aiter_bytes(),
            headers=r.headers
        )
    except httpx.RequestError as e:
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"File storage service is unavailable: {e}")


@router.get("/", response_model=List[Research])
async def read_researches(
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Получает список проектов текущего пользователя.

    Возвращает:
    - Список объектов проектов, принадлежащих пользователю.
    """
    return await crud_user.get_user_researches(db, user_id=current_user.id)


@router.get("/{research_id}", response_model=Research)
async def read_research(
        research_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Получает один проект по его ID. Доступно только владельцу.

    Входные данные:
    - **research_id**: UUID проекта из пути URL.

    Возвращает:
    - Объект проекта или ошибку 404.
    """
    db_research = await crud_research.get_research(db, research_id=research_id)
    if db_research is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _require_research_owner(db_research, current_user)
    return db_research


@router.put("/{research_id}/status", response_model=Research, tags=["researches"])
async def update_research_status(
        research_id: str,
        status_update: ResearchStatusUpdateByIdSchema,
        db: AsyncSession = Depends(get_db),
        principal: Principal = Depends(get_principal),
):
    # Вызывается math-воркером (внутренний токен) либо владельцем исследования.
    # Поддержка как полного UUID, так и короткого ID (префикса)
    try:
        research_uuid = uuid.UUID(research_id)
        research = await crud_research.get_research(db, research_id=research_uuid)
    except ValueError:
        research = await crud_research.get_research_by_short_id(db, short_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _enforce_owner_unless_internal(principal, research)

    status_obj = await crud_status.get_status_by_id(db, status_id=status_update.status_id)
    if not status_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Status with ID '{status_update.status_id}' not found.")

    try:
        return await crud_research.update_research_status(db, research=research, status=status_obj)
    except InvalidStatusTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.delete("/{research_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research(
        research_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserModel = Depends(get_current_user),
):
    """
    Удаляет проект по его ID. Доступно только владельцу.

    Входные данные:
    - **research_id**: UUID проекта из пути URL.

    Возвращает:
    - Ответ 204 No Content, тело ответа пустое.
    """
    research = await crud_research.get_research(db, research_id=research_id)
    if not research:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research not found")
    _require_research_owner(research, current_user)

    await crud_research.delete_research(db, research=research)

    try:
        delete_url = f"{FILESTORAGE_SERVICE_BASE_URL}/delete/{str(research_id)[:5]}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(delete_url, timeout=30.0)
            if response.status_code not in [200,204]:
                print(f"WARNING: Delete request to filestorage failed with status {response.status_code}")
    except httpx.RequestError as e:
        print(f"WARNING: Could not connect to filestorage to delete files: {e}")

    return None
