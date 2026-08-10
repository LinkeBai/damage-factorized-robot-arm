$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = ".\.venv\Scripts\python.exe"
foreach($seed in 7,17,27){
  & $python -u scripts/run_v6_option_selector.py --seed $seed 1>> "runs/v6_selector_seed$seed.log" 2>> "runs/v6_selector_seed$seed.err.log"
  if($LASTEXITCODE -ne 0){ throw "V6 selector seed $seed failed" }
}
