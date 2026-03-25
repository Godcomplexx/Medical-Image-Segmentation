from pathlib import Path
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import logging
from starlette.concurrency import run_in_threadpool

app = APIRouter()
logging.basicConfig(level=logging.INFO)

@app.post("/run_fastsurfer/")
async def run_fastsurfer(
    t1_path: str = Form(...),
    subject_id: str = Form(...),
    license_path: str = Form(...),
    output_dir: str = Form(...)
):
    # Удаляем /app из начала пути, если есть
    t1_path = t1_path.replace('/math_app', '', 1)

    # Преобразуем строки в Path-объекты
    t1_file = Path(t1_path)
    license_file = Path(license_path)
    output_folder = Path(output_dir)

    # Проверка существования путей
    missing = []
    if not t1_file.exists():
        missing.append(f"T1 file does not exist: {t1_file}")
    if not license_file.exists():
        missing.append(f"License file does not exist: {license_file}")
    if not output_folder.exists():
        missing.append(f"Output directory does not exist: {output_folder}")

    if missing:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "missing": missing}
        )

    # Команда запуска
    cmd = [
        "/fastsurfer/run_fastsurfer.sh",
        "--t1", str(t1_file),
        "--sid", subject_id,
        "--sd", str(output_folder),
        "--fs_license", str(license_file),
        "--device", "cuda",
        "--parallel",
        "--allow_root"
    ]

    try:
        logging.info(f"Running FastSurfer with command: {' '.join(cmd)}")
        result = await run_in_threadpool(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"FastSurfer stdout:\n{result.stdout}")
        logging.info(f"FastSurfer stderr:\n{result.stderr}")
        return {
            "status": "ok",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.CalledProcessError as e:
        logging.error(f"FastSurfer failed with exit code {e.returncode}")
        logging.error(f"stdout:\n{e.stdout}")
        logging.error(f"stderr:\n{e.stderr}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "exit_code": e.returncode
            }
        )