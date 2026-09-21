param([Parameter(Mandatory = $true)][int]$ScaleRunnerPid)
$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\asus\Desktop\damage-factorized-robot-arm'
$scaleProcess = Get-Process -Id $ScaleRunnerPid -ErrorAction SilentlyContinue
if ($null -ne $scaleProcess) {
    $scaleProcess.WaitForExit()
}
$scaleStatus = Get-Content -LiteralPath "$repo\runs\ipwm_scale_goal_20260911\loop-status.json" -Raw | ConvertFrom-Json
if ($scaleStatus.status -ne 'complete' -or $scaleStatus.goal_complete -ne $true) {
    throw "The preceding scale study did not pass its completion audit (status=$($scaleStatus.status)); positive replication was not started."
}
& "$repo\.venv-cuda\Scripts\python.exe" "$repo\scripts\run_ipwm_positive_scale_loop.py"
