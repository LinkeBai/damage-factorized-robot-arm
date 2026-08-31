param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$FrozenSchedule,
    [Parameter(Mandatory = $true)][string]$CompletedLog,
    [string]$Python = ".\.venv-cuda\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Repository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repository

& $Python scripts\audit_real_robot_preflight.py $Manifest `
    --mode level_a `
    --schedule $FrozenSchedule `
    --output results\real_robot\preflight-audit.json
if ($LASTEXITCODE -ne 0) { throw "Level-A preflight failed" }

& $Python scripts\audit_real_robot_schedule_completion.py `
    $FrozenSchedule $CompletedLog `
    --output results\real_robot\schedule-completion-audit.json
if ($LASTEXITCODE -ne 0) { throw "Frozen/completed schedule identity audit failed" }

& $Python scripts\analyze_real_robot_push.py $CompletedLog `
    --require-files `
    --output results\real_robot\push-summary.json
if ($LASTEXITCODE -ne 0) { throw "Real-robot validity/statistics gate failed" }

& $Python scripts\build_real_robot_feasibility_assets.py `
    results\real_robot\push-summary.json `
    --figure paper\generated\real-robot-feasibility.pdf `
    --table paper\generated\real-robot-feasibility-table.tex
if ($LASTEXITCODE -ne 0) { throw "Real-robot paper-asset generation failed" }

Write-Output "Level-A chain PASS: preflight, schedule identity, raw files, statistics, and paper assets."
Write-Output "Run scripts\audit_goal_completion.py after integrating the generated assets and final independent score."
