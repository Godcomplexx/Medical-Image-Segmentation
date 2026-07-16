# Запуск обучения U-Net сегментатора очагов — полностью отсоединённо.
# Процесс переживёт закрытие терминала и разрыв интернета/сессии.
# Безопасно запускать повторно: если обучение уже идёт, второй раз не стартует.

$repo = "D:\Medical-Image-Segmentation"
$py   = "$repo\.venv-train\Scripts\python.exe"
$out  = "$repo\finetune\fileStorage\two_stage_full\seg_runs\unet_2p5d"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*train_seg*' }

if ($running) {
    Write-Host "Обучение УЖЕ идёт (PID: $($running.ProcessId -join ', ')). Повторно не запускаю." -ForegroundColor Yellow
}
else {
    $p = Start-Process -FilePath $py `
        -ArgumentList '-m','finetune.two_stage_pipeline.segmentation.train_seg',
                      '--epochs','40','--batch','6','--imgsz','640','--workers','2',
                      '--background-ratio','0.4','--lr','0.00005' `
        -WorkingDirectory $repo `
        -RedirectStandardOutput "$out\train.out" `
        -RedirectStandardError  "$out\train.err" `
        -WindowStyle Hidden -PassThru
    Write-Host "ЗАПУЩЕНО обучение, PID: $($p.Id)." -ForegroundColor Green
    Write-Host "Оно продолжит работать, даже если закрыть терминал или пропадёт интернет."
}

Write-Host ""
Write-Host "Смотреть прогресс:" -ForegroundColor Cyan
Write-Host "  Get-Content `"$out\results.csv`" -Wait -Tail 10"
Write-Host "Остановить (если нужно):" -ForegroundColor Cyan
Write-Host "  Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { `$_.CommandLine -like '*train_seg*' } | % { taskkill /PID `$_.ProcessId /T /F }"
