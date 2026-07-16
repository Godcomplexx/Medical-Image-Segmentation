# «Лёгкий» запуск обучения — оставляет видеопамять свободной, чтобы можно было
# работать за компьютером, пока идёт обучение.
#   batch 3 (вместо 6), лимит VRAM 55% (~4.4 ГБ из 8), 1 воркер.
# Отсоединён: переживёт закрытие терминала и обрыв интернета.
# Медленнее полного режима, но результат тот же.

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
                      '--epochs','40','--batch','3','--imgsz','640','--workers','1',
                      '--background-ratio','0.4','--lr','0.00005','--gpu-mem-fraction','0.55' `
        -WorkingDirectory $repo `
        -RedirectStandardOutput "$out\train.out" `
        -RedirectStandardError  "$out\train.err" `
        -WindowStyle Hidden -PassThru
    Write-Host "ЗАПУЩЕНО (лёгкий режим), PID: $($p.Id)." -ForegroundColor Green
    Write-Host "VRAM ограничена ~4.4 ГБ — остальное свободно для вашей работы."
}

Write-Host ""
Write-Host "Смотреть прогресс:" -ForegroundColor Cyan
Write-Host "  Get-Content `"$out\results.csv`" -Wait -Tail 10"
