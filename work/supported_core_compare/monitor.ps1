$ErrorActionPreference = 'Stop'
$taskRoot = 'C:/Users/asus/Desktop/damage-factorized-robot-arm'
$taskOut = Join-Path $taskRoot 'runs/lockpusher_supported_core_20260913'
$taskState = Get-Content -LiteralPath (Join-Path $taskOut 'driver-state.json') -Raw | ConvertFrom-Json
$taskProcesses = @(Get-CimInstance Win32_Process)
$taskDriver = @($taskProcesses | Where-Object { $_.ProcessId -eq $taskState.pid -and $_.CommandLine -match 'supported_core_compare[\\/]driver\.py' })
$taskIds = [System.Collections.Generic.HashSet[int]]::new()
foreach ($taskItem in $taskDriver) { [void]$taskIds.Add([int]$taskItem.ProcessId) }
do {
    $taskPreviousCount = $taskIds.Count
    foreach ($taskItem in $taskProcesses) {
        if ($taskIds.Contains([int]$taskItem.ParentProcessId)) { [void]$taskIds.Add([int]$taskItem.ProcessId) }
    }
} while ($taskIds.Count -gt $taskPreviousCount)
$taskLive = @($taskProcesses | Where-Object { $taskIds.Contains([int]$_.ProcessId) } | ForEach-Object {
    $taskRuntime = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    [ordered]@{pid=$_.ProcessId; parent=$_.ParentProcessId; cpu_seconds=$taskRuntime.CPU; working_set_bytes=$taskRuntime.WorkingSet64}
})
$taskCounts = [ordered]@{}
foreach ($taskSplit in @('pool','validation','test')) {
    $taskSplitPath = Join-Path $taskOut "data/$taskSplit"
    $taskCount = 0
    $taskShardCount = 0
    if (Test-Path -LiteralPath $taskSplitPath) {
        foreach ($taskManifest in Get-ChildItem -LiteralPath $taskSplitPath -Filter manifest.json -Recurse -File) {
            if ($taskManifest.Directory.Name -match '^shard-\d+-\d+$') {
                $taskRecord = Get-Content -LiteralPath $taskManifest.FullName -Raw | ConvertFrom-Json
                if ($taskRecord.hard_gate_passed) { $taskCount += [int]$taskRecord.count; $taskShardCount += 1 }
            }
        }
    }
    $taskCounts[$taskSplit] = @{committed_trajectories=$taskCount; committed_shards=$taskShardCount}
}
$taskFits = @()
foreach ($taskSeed in @(7,17,27)) {
    foreach ($taskMethod in @('ipwm','carrier','global')) {
        $taskFitPath = Join-Path $taskOut "training/$taskMethod/seed$taskSeed"
        $taskComplete = Join-Path $taskFitPath 'complete.json'
        $taskProgress = Join-Path $taskFitPath 'progress.json'
        if (Test-Path -LiteralPath $taskComplete) {
            $taskFit = Get-Content -LiteralPath $taskComplete -Raw | ConvertFrom-Json
            $taskFits += @{method=$taskMethod; seed=$taskSeed; status='complete_file_present'; updates=$taskFit.total_updates}
        } elseif (Test-Path -LiteralPath $taskProgress) {
            $taskFit = Get-Content -LiteralPath $taskProgress -Raw | ConvertFrom-Json
            $taskFits += @{method=$taskMethod; seed=$taskSeed; status='progress_file_present'; progress=$taskFit}
        }
    }
}
$taskLog = Join-Path $taskOut "logs/$($taskState.current_job).log"
$taskLogTail = if (Test-Path -LiteralPath $taskLog) { @(Get-Content -LiteralPath $taskLog -Tail 4) } else { @() }
$taskSystem = Get-CimInstance Win32_OperatingSystem
[ordered]@{
    checked_utc=[DateTime]::UtcNow.ToString('o')
    scope='Read-only progress snapshot; file presence/count is not final scientific verification'
    driver_status=$taskState.status
    current_job=$taskState.current_job
    driver_pid=$taskState.pid
    driver_live_with_matching_command=($taskDriver.Count -eq 1)
    process_tree=$taskLive
    data=$taskCounts
    fits=$taskFits
    log_tail=$taskLogTail
    disk_free_bytes=(Get-PSDrive C).Free
    ram_available_bytes=([long]$taskSystem.FreePhysicalMemory * 1024)
} | ConvertTo-Json -Depth 8 -Compress
