@echo off
REM ============================================================
REM  Графический интерфейс: разметка очагов РС для проверки в 3D Slicer
REM  Можно копировать/переносить куда угодно (напр. на рабочий стол).
REM  Если проект переедет — поменяйте одну строку REPO ниже.
REM ============================================================
set "REPO=D:\Medical-Image-Segmentation"
set "PYW=%REPO%\.venv-train\Scripts\pythonw.exe"
set "PY=%REPO%\.venv-train\Scripts\python.exe"
set "GUI=%REPO%\review_tool\gui.py"

if not exist "%REPO%" (
  echo [!] Не найдена папка проекта: %REPO%
  echo     Отредактируйте строку REPO в этом файле.
  pause
  exit /b 1
)
cd /d "%REPO%"

if exist "%PYW%" (
  start "" "%PYW%" "%GUI%"
) else (
  if exist "%PY%" (
    "%PY%" "%GUI%"
  ) else (
    echo [!] Не найдено окружение .venv-train в %REPO%
    pause
  )
)
