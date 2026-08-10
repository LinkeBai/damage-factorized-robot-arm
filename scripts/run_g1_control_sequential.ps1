$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = ".\.venv\Scripts\python.exe"

function Run-Seed([int]$Seed) {
    $log = "runs/g1_control_seed${Seed}.log"
    $err = "runs/g1_control_seed${Seed}.err.log"
    & $python -u scripts/run_g1_control_gate.py --seeds $Seed --candidates 32 --max-steps 300 1>> $log 2>> $err
    if ($LASTEXITCODE -ne 0) { throw "control seed $Seed failed: $LASTEXITCODE" }
}

Run-Seed 7
Run-Seed 17
Run-Seed 27
