$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

$python = Join-Path $repo '.venv-cuda\Scripts\python.exe'
$collector = Join-Path $repo 'scripts\collect_real_push_automated.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $collector)) {
    throw "Collector not found: $collector"
}

# Take ownership only from the project's known preview server. Never terminate
# an unrelated process merely because it happens to use the same port.
$listeners = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($owner.CommandLine -notmatch 'serve_dual_camera_preview\.py') {
        throw "Port 8765 is occupied by an unrelated process (PID $($listener.OwningProcess))."
    }
    Write-Host "Stopping the old preview server (PID $($listener.OwningProcess))..."
    Stop-Process -Id $listener.OwningProcess -Force
}
Start-Sleep -Milliseconds 800

Write-Host ''
Write-Host 'Real Push Collector - standalone mode' -ForegroundColor Cyan
Write-Host 'Schedule: D2 then D3, one trial each.'
Write-Host 'The robot starts only after the cube stays in RESET START for 4 seconds.'
Write-Host 'Keep hands/persons outside the work area. Press Ctrl+C to stop.' -ForegroundColor Yellow
Write-Host ''

$runTag = 'final45-day2-standalone-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $python $collector `
    --conditions D2 D3 `
    --repeats 1 `
    --trial-prefix $runTag `
    --execute `
    --open-browser `
    --acknowledge-risk I_HAVE_CLEARED_WORKSPACE_SUPPORTED_ARM_AND_TESTED_ESTOP

$exitCode = $LASTEXITCODE
Write-Host ''
if ($exitCode -eq 0) {
    Write-Host 'Collection completed. The dual-camera preview remains running.' -ForegroundColor Green
} else {
    Write-Host "Collector stopped with exit code $exitCode. Inspect automated_collection logs." -ForegroundColor Red
}
Write-Host 'Press any key to close this window.'
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
exit $exitCode
