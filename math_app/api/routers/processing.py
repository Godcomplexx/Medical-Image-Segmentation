from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status, Request
from fastapi.responses import JSONResponse, FileResponse, Response
import traceback
from pathlib import Path
import zipfile
import shutil
from src.pipelines.volumes_predict import process_survey
from src.filetools.safe_paths import safe_name, resolve_within, safe_extract_zip
from config.settings import settings
from config.logs_config import logger_load_data
from typing import Literal
from starlette.concurrency import run_in_threadpool
import asyncio

router = APIRouter()

@router.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    # Берём только базовое имя файла и валидируем как безопасный компонент пути.
    zip_filename = safe_name(Path(file.filename or "").name)
    folder_name = safe_name(Path(zip_filename).stem)
    zip_path = resolve_within(settings.input_data_path, zip_filename)
    target_dir = resolve_within(settings.input_data_path, folder_name)

    if zip_path.exists() or target_dir.exists():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="File or directory already exists.")

    # сохраняем файл чанками (без загрузки всего в память)
    with open(zip_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            safe_extract_zip(zf, target_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Not a valid ZIP file")
    finally:
        zip_path.unlink(missing_ok=True)

    return {"status": "ok", "folder_name": folder_name}

@router.post("/run_process")
async def run_process(folder_name: str = Form(...)):
    try:
        folder_path = resolve_within(settings.input_data_path, safe_name(folder_name))

        if not folder_path.exists():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Folder '{folder_name}' not found.")

        try:
            result = await run_in_threadpool(process_survey, folder_path)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Failed to process survey (process_survey)", "details": str(e)}
            )

        return {"result": result}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to process survey (outer)", "details": str(e)}
        )


@router.api_route("/download/{file_type}/{study_name}", methods=['GET', 'HEAD'])
async def download_file(
        request: Request,
        file_type: Literal["t1", "t2", "prediction", "parcellation"],
        study_name: str
):
    """
    Скачать файл по типу и имени исследования.

    Параметры:
    - file_type: тип файла (t1, t2, prediction, parcellation)
    - study_name: имя исследования

    Примеры:
    - GET /download/t1/my_study
    - GET /download/parcellation/my_study
    """
    try:
        # Базовая папка исследования (валидируем имя и путь)
        study_path = resolve_within(settings.processed_data_path, safe_name(study_name))
        if not study_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found")

        # Определяем путь к файлу в зависимости от типа
        if file_type == "t1":
            file_path = study_path / "T1.nii"
        elif file_type == "t2":
            file_path = study_path / "T2.nii"
        elif file_type == "prediction":
            file_path = study_path / "Predict" / "segmentation.nii"
        elif file_type == "parcellation":
            file_path = study_path / "ParcelationMask" / "parcelation.nii"
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        print(f"DEBUG: Checking for file at path: {file_path}")

        # Проверяем существование файла
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_path.name} not found for study {study_name}, path {file_path}"
            )

        headers = {
            "Content-Disposition": f"attachment; filename={file_path.name}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }

        if request.method == "HEAD":
            return Response(headers=headers, media_type="application/octet-stream")

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/octet-stream",
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to download file", "details": str(e)}
        )

@router.delete("/delete/{study_name}")
async def delete_survey(study_name: str):

    try:
        safe_study = safe_name(study_name)
        survey_path: Path = resolve_within(settings.processed_data_path, safe_study)
        survey_morf: Path = resolve_within(settings.morphometry_path, safe_study)
        input_files: Path = resolve_within(settings.input_data_path, safe_study)

        shutil.rmtree(survey_path,ignore_errors=True)
        shutil.rmtree(survey_morf,ignore_errors=True)
        shutil.rmtree(input_files, ignore_errors=True)


        return {'status': 'ok', 'deleted': safe_study}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to delete file", "details": str(e)}
        )

