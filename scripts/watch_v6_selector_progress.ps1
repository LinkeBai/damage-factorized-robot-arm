Set-Location (Split-Path -Parent $PSScriptRoot)
$started = Get-Date
while ($true) {
    $proc = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*run_v6_option_selector.py*" }
    $files = @(Get-ChildItem "results/final" -Filter "v6-option-selector-seed*.csv" -ErrorAction SilentlyContinue)
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
    Clear-Host
    Write-Host "V6 OPTION SELECTOR PROGRESS" -ForegroundColor Cyan
    Write-Host "==========================="
    if ($proc) {
        $cmd = ($proc | Select-Object -First 1).CommandLine
        $seed = if ($cmd -match '--seed (\d+)') { $Matches[1] } else { '?' }
        Write-Host "Status: RUNNING"
        Write-Host "Current seed: $seed"
    } else { Write-Host "Status: NO ACTIVE PROCESS" -ForegroundColor Yellow }
    Write-Host "Elapsed: $([int]((Get-Date)-$started).TotalMinutes) min"
    Write-Host "GPU: $gpu"
    Write-Host "Completed: $($files.Count)/3"
    foreach ($file in $files) { Write-Host "  $($file.Name)" -ForegroundColor Green }
    Write-Host ""
    Write-Host "Latest log:"
    Get-ChildItem "runs/v6_selector_seed*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1 |
        ForEach-Object { Get-Content $_.FullName -Tail 5 }
    if (-not $proc -and $files.Count -ge 3) { break }
    Start-Sleep -Seconds 3
}
Write-Host "All V6 selector seeds completed." -ForegroundColor Green
Read-Host "Press Enter to close"
