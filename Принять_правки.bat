@echo off
REM ============================================================
REM  «Принять правки доктора»: проверенные в 3D Slicer сегментации
REM  переводятся обратно в формат обучающей разметки.
REM  Можно переносить куда угодно. Если проект переедет — правьте REPO.
REM
REM  - по умолчанию берёт проверенные файлы из review_tool\output
REM  - или перетащите на этот .bat папку с проверенными результатами
REM  Итог: review_tool\accepted\{images,labels} + manifest.csv
REM ============================================================
setlocal
set "REPO=D:\Medical-Image-Segmentation"
set "PY=%REPO%\.venv-train\Scripts\python.exe"
set "WORKER=%REPO%\review_tool\accept_edits.py"
if not "%~1"=="" ( set "INDIR=%~1" ) else ( set "INDIR=%REPO%\review_tool\output" )

if not exist "%PY%" ( echo [!] Нет окружения (проверьте REPO): "%PY%" & pause & exit /b 1 )
cd /d "%REPO%"

echo === Приём правок из: %INDIR% ===
"%PY%" "%WORKER%" --input "%INDIR%" --output "%REPO%\review_tool\accepted"
echo.
pause
endlocal
