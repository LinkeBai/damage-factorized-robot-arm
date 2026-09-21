$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo ".venv-cuda\Scripts\python.exe"
$Setup = Join-Path $Repo "data\real_robot\session_20260901\setup"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ServoLog = Join-Path $Setup "final_day_servo_readonly_$Stamp.txt"
$PreviewLog = Join-Path $Setup "final_day_preview_$Stamp.log"

New-Item -ItemType Directory -Force -Path $Setup | Out-Null
Push-Location $Repo
try {
    Write-Host "[1/4] Running frozen software preflight..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot "preflight_final_real_ipwm_day.ps1") -SkipTests
    if ($LASTEXITCODE -ne 0) { throw "Software preflight failed" }

    Write-Host "[2/4] Reading COM3 servo status (no actuator command)..."
    & $Python scripts/read_sts_status_raw.py --port COM3 --baudrate 1000000 |
        Tee-Object -FilePath $ServoLog
    if ($LASTEXITCODE -ne 0) { throw "COM3 read-only status failed" }
    $ServoText = Get-Content -LiteralPath $ServoLog -Raw
    if (($ServoText | Select-String -AllMatches "valid=True").Matches.Count -ne 15) {
        throw "Expected 15 valid position/voltage/temperature register replies"
    }

    Write-Host "[3/4] Starting persistent dual-camera preview in camera-only mode..."
    $Existing = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
    if (-not $Existing) {
        $PreviousCameraOnly = $env:ROBOTARM_PREVIEW_CAMERA_ONLY
        $env:ROBOTARM_PREVIEW_CAMERA_ONLY = "1"
        try {
            Start-Process -FilePath $Python `
                -ArgumentList "scripts/serve_dual_camera_preview.py" `
                -WorkingDirectory $Repo -WindowStyle Hidden `
                -RedirectStandardOutput $PreviewLog -RedirectStandardError ($PreviewLog + ".err")
        }
        finally {
            $env:ROBOTARM_PREVIEW_CAMERA_ONLY = $PreviousCameraOnly
        }
        $Deadline = (Get-Date).AddSeconds(20)
        do {
            Start-Sleep -Milliseconds 500
            $Listening = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
        } while (-not $Listening -and (Get-Date) -lt $Deadline)
        if (-not $Listening) { throw "Camera preview did not listen on port 8765" }
    }

    Write-Host "[4/4] Opening the live page..."
    Start-Process "http://127.0.0.1:8765/?layout=final-push-20260906-live"
    Write-Host "READY: software, COM3 read-only replies, and preview service passed."
    Write-Host "Servo evidence: $ServoLog"
    Write-Host "Next formal trial: P20-I-01. Do not execute until the cube start gate is green."
}
finally {
    Pop-Location
}
