param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo ".venv-cuda\Scripts\python.exe"
$Protocol = Join-Path $Repo "config\experiment\real_ipwm_final_day_20260906.yaml"
$Schedule = Join-Path $Repo "data\real_robot\final_day_schedule_20260906.csv"
$Checkpoint = Join-Path $Repo "runs\icra_confirmation_d3_query_selective_w10\seed27\model.pt"
$ModelConfig = Join-Path $Repo "config\experiment\icra_primary_d2d4_eval_strict_3seed_v1.yaml"
$Axis = Join-Path $Repo "results\real_robot\push_axis_current_epoch_20260903.json"
$Safety = Join-Path $Repo "hardware\safety_limits.yaml"
$Output = Join-Path $Repo "results\real_robot\final_day_software_preflight_20260906.json"

$Required = @($Python, $Protocol, $Schedule, $Checkpoint, $ModelConfig, $Axis, $Safety)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file missing: $Path"
    }
}

Push-Location $Repo
try {
    if (-not $SkipTests) {
        & $Python -m pytest -q `
            tests/test_ipwm_receding_horizon.py `
            tests/test_real_ipwm_batched_scoring.py `
            tests/test_audit_real_ipwm_closed_loop_trial.py `
            tests/test_real_robot_trial_packet.py `
            tests/test_dual_camera_sync_audit.py `
            tests/test_audit_final_real_ipwm_day.py
        if ($LASTEXITCODE -ne 0) { throw "Targeted test suite failed" }
    }

    $Probe = & $Python -c @"
import csv, hashlib, json, pathlib, torch, yaml
paths = {
    'protocol': pathlib.Path(r'$Protocol'),
    'schedule': pathlib.Path(r'$Schedule'),
    'checkpoint': pathlib.Path(r'$Checkpoint'),
    'model_config': pathlib.Path(r'$ModelConfig'),
    'axis_calibration': pathlib.Path(r'$Axis'),
    'safety': pathlib.Path(r'$Safety'),
}
protocol = yaml.safe_load(paths['protocol'].read_text(encoding='utf-8-sig'))
rows = list(csv.DictReader(paths['schedule'].open(encoding='utf-8-sig', newline='')))
assert protocol['push']['goal_gate']['tolerance_px'] == 5
assert protocol['push']['matrices']['core_quantitative']['distance_px'] == 20
assert len(rows) == 27 and len({r['trial_id'] for r in rows}) == 27
assert all(r['task'] == 'push' for r in rows)
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
payload = {
    'status': 'PASS_SOFTWARE_ONLY_NO_HARDWARE_ACCESSED',
    'protocol_id': protocol['protocol_id'],
    'schedule_rows': len(rows),
    'core_rows': sum(r['priority'] == 'core' for r in rows),
    'conditional_rows': sum(r['priority'] == 'conditional' for r in rows),
    'cuda_available': torch.cuda.is_available(),
    'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'torch_version': torch.__version__,
    'sha256': {name: digest(path) for name, path in paths.items()},
}
assert payload['cuda_available'], 'NVIDIA CUDA is required for formal online IPWM execution'
print(json.dumps(payload))
"@
    if ($LASTEXITCODE -ne 0) { throw "CUDA/protocol/hash probe failed" }
    $Payload = $Probe | ConvertFrom-Json
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
    Write-Host "PASS: software-only final-day preflight"
    Write-Host "CUDA: $($Payload.cuda_device)"
    Write-Host "Schedule: $($Payload.schedule_rows) rows ($($Payload.core_rows) core, $($Payload.conditional_rows) conditional)"
    Write-Host "Saved: $Output"
    Write-Host "Next: start dual cameras, then run the COM3 read-only hardware preflight."
}
finally {
    Pop-Location
}

