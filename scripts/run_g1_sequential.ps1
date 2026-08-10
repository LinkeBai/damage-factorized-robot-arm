param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputRoot = "runs/g1_split"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Wait-ForSeed7 {
    while ($true) {
        $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object {
                $_.CommandLine -like "*run_g1_benchmark.py*" -and
                $_.CommandLine -like "*--seeds 7*"
            }
        if (-not $running) {
            return
        }
        Start-Sleep -Seconds 15
    }
}

function Run-Seed([int]$Seed) {
    $log = "runs/g1_seed${Seed}.log"
    $err = "runs/g1_seed${Seed}.err.log"
    & $Python scripts/run_g1_benchmark.py `
        --out $OutputRoot `
        --seeds $Seed `
        --epochs 60 `
        --train-trajectories 2 `
        --calibration-trajectories 5 `
        --evaluation-trajectories 3 `
        --trajectory-steps 100 `
        --latent-steps 50 `
        --data-policy controller `
        --device cuda 1>> $log 2>> $err
    if ($LASTEXITCODE -ne 0) {
        throw "G1 seed $Seed failed with exit code $LASTEXITCODE"
    }
}

Wait-ForSeed7
Run-Seed 17
Run-Seed 27
