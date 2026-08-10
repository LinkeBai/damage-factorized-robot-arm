$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = ".\.venv\Scripts\python.exe"
foreach($seed in 7,17,27){
  & $python -u scripts/run_g1_worldmodel_hybrid.py --seed $seed 1>> "runs/g1_worldmodel_seed$seed.log" 2>> "runs/g1_worldmodel_seed$seed.err.log"
  if($LASTEXITCODE -ne 0){ throw "worldmodel hybrid seed $seed failed" }
}
