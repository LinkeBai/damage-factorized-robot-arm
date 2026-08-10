$ErrorActionPreference = "SilentlyContinue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$seeds = @(7, 17, 27)
$started = Get-Date

while ($true) {
    $completed = @()
    Get-ChildItem "runs/g1_split" -Recurse -Filter summary.json | ForEach-Object {
        $summary = Get-Content $_.FullName -Raw | ConvertFrom-Json
        foreach ($seed in $summary.seeds) {
            if ($seeds -contains [int]$seed) {
                $completed += [int]$seed
            }
        }
    }
    $completed = @($completed | Sort-Object -Unique)

    $active = $null
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*run_g1_benchmark.py*" }
    foreach ($seed in $seeds) {
        if ($processes.CommandLine -like "*--seeds $seed*") {
            $active = $seed
            break
        }
    }

    $percent = [math]::Round(100 * $completed.Count / $seeds.Count)
    $elapsed = (Get-Date) - $started
    $status = if ($active) {
        "Running seed $active | completed $($completed.Count)/3 | monitored $([int]$elapsed.TotalMinutes) min"
    } elseif ($completed.Count -eq $seeds.Count) {
        "All seeds complete"
    } else {
        "Waiting for next seed | completed $($completed.Count)/3"
    }

    Clear-Host
    Write-Host "G1 FORMAL EXPERIMENT PROGRESS" -ForegroundColor Cyan
    Write-Host ""
    Write-Progress -Activity "G1: D2/D3, 4 methods, K=0/1/2/5" -Status $status -PercentComplete $percent
    Write-Host $status
    Write-Host "Completed seeds: $($completed -join ', ')"
    Write-Host "Results: $repo\runs\g1_split"
    Write-Host ""
    Write-Host "Auto-refreshing every 3 seconds. Keep this window open." -ForegroundColor Yellow

    if ($completed.Count -eq $seeds.Count) {
        Write-Progress -Activity "G1" -Completed
        Write-Host "All three G1 seeds completed." -ForegroundColor Green
        Start-Sleep -Seconds 30
        break
    }
    Start-Sleep -Seconds 3
}
