$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$started = Get-Date
while ($true) {
    $proc = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*run_g1_worldmodel_hybrid.py*" }
    $files = @(Get-ChildItem "results/final" -Filter "g1-worldmodel-hybrid-seed*.csv" -ErrorAction SilentlyContinue)
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null)
    Clear-Host
    Write-Host "WORLD-MODEL HYBRID PROGRESS" -ForegroundColor Cyan
    Write-Host "================================"
    if ($proc) {
        $cmd = ($proc | Select-Object -First 1).CommandLine
        $seed = if ($cmd -match '--seed (\d+)') { $Matches[1] } else { '?' }
        Write-Host "Status: RUNNING"
        Write-Host "Seed: $seed"
    } else {
        Write-Host "Status: NO ACTIVE PROCESS" -ForegroundColor Yellow
    }
    Write-Host "Elapsed: $([int]((Get-Date)-$started).TotalMinutes) min"
    Write-Host "GPU: $gpu"
    Write-Host "Completed files: $($files.Count)/3"
    foreach ($file in $files) { Write-Host "  $($file.Name)" -ForegroundColor Green }
    Write-Host ""
    Write-Host "Recent seed logs:"
    Get-ChildItem "runs/g1_worldmodel_seed*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1 |
        ForEach-Object { Get-Content $_.FullName -Tail 5 }
    if (-not $proc -and $files.Count -ge 3) { break }
    Start-Sleep -Seconds 3
}
Write-Host "All world-model hybrid seeds completed." -ForegroundColor Green
Read-Host "Press Enter to close"
