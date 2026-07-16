@echo off
REM ============================================================
REM  Генерация кандидат-разметки очагов РС для проверки в 3D Slicer
REM  Можно переносить куда угодно. Если проект переедет — правьте REPO.
REM
REM  A) Перетащите папку с исследованиями ПРЯМО на этот .bat, ИЛИ
REM  B) положите исследования в review_tool\input и дважды кликните .bat
REM  Вход: DICOM-подпапки (внутри DICOM\...) или .nii/.nii.gz
REM ============================================================
setlocal
set "REPO=D:\Medical-Image-Segmentation"
set "PY=%REPO%\.venv-train\Scripts\python.exe"
set "WORKER=%REPO%\review_tool\detect_for_review.py"
set "OUTDIR=%REPO%\review_tool\output"

if not "%~1"=="" ( set "INDIR=%~1" ) else ( set "INDIR=%REPO%\review_tool\input" )

if not exist "%PY%" (
  echo [!] Не найдено окружение: "%PY%"  (проверьте строку REPO)
  pause & exit /b 1
)
cd /d "%REPO%"
if not exist "%INDIR%" mkdir "%INDIR%"

for /f %%N in ('dir /b /a "%INDIR%" 2^>nul ^| find /c /v ""') do set "CNT=%%N"
if "%CNT%"=="0" (
  echo.
  echo [!] Папка пуста: %INDIR%
  echo     Положите DICOM-исследования или .nii/.nii.gz, либо перетащите папку на .bat.
  echo.
  pause & exit /b 0
)

echo.
echo === Запуск детектора очагов (nnU-Net + 2.5D + авто-класс) ===
echo   Вход:  %INDIR%
echo   Выход: %OUTDIR%
echo.
"%PY%" "%WORKER%" --input "%INDIR%" --output "%OUTDIR%" --min-mm3 15
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo === Готово. Откройте результаты в 3D Slicer: ===
  echo   %OUTDIR%
) else (
  echo [!] Завершилось с ошибкой (код %RC%). Смотрите сообщения выше.
)
echo.
pause
endlocal
