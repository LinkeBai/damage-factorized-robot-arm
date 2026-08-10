param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Missing .venv. Create it and install the project before running this script."
}

function Run-Step([string]$Name, [scriptblock]$Command) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] START $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] DONE  $Name" -ForegroundColor Green
}

Run-Step "tests" { & $python -m pytest -q }
Run-Step "hybrid baseline" { & $python -u scripts/run_g1_hybrid_baseline.py }
Run-Step "residual feedback" { & $python -u scripts/run_g1_residual_feedback.py }

if ($Mode -eq "full") {
    Run-Step "four-method prediction benchmark" {
        & $python -u scripts/run_g1_benchmark.py --out runs/g1_reproduction --seeds 7,17,27 --epochs 60 --train-trajectories 2 --calibration-trajectories 5 --evaluation-trajectories 3 --trajectory-steps 100 --latent-steps 50 --data-policy controller --device auto
    }
    Run-Step "world-model hybrid seeds" { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_g1_worldmodel_hybrid_sequential.ps1 }
    Run-Step "V6 option selector seeds" { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_v6_option_selector_sequential.ps1 }
}

Run-Step "V6 gate summary" { & $python scripts/evaluate_v6_gate.py }
Run-Step "run manifests" { & $python scripts/build_g1_manifests.py }
Write-Host "G1 reproduction complete. See reports/ and results/final/." -ForegroundColor Green
