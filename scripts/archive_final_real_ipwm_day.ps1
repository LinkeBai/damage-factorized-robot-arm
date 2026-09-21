param(
    [Parameter(Mandatory = $true)][string]$BackupA,
    [Parameter(Mandatory = $true)][string]$BackupB,
    [Parameter(Mandatory = $true)][string]$ReportDirectory
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo ".venv-cuda\Scripts\python.exe"
$Evidence = Join-Path $Repo "data\real_robot\session_20260901\final_day_trials"
$Audit = Join-Path $Repo "results\real_robot\final_day_completion_audit.json"
$Schedule = Join-Path $Repo "data\real_robot\final_day_schedule_20260906.csv"
$Protocol = Join-Path $Repo "config\experiment\real_ipwm_final_day_20260906.yaml"

Push-Location $Repo
try {
    & $Python scripts/audit_final_real_ipwm_day.py
    if ($LASTEXITCODE -ne 0) { throw "Completion auditor failed" }
    $Result = Get-Content -LiteralPath $Audit -Raw | ConvertFrom-Json
    if ($Result.evidence_ready_for_archive -ne $true) {
        throw "Evidence is incomplete; refusing to create a final archive or authorize dismantling"
    }
    if (-not (Test-Path -LiteralPath $Evidence -PathType Container)) {
        throw "Evidence root is missing: $Evidence"
    }

    $Reports = (Resolve-Path -LiteralPath $ReportDirectory).Path
    $SummaryPath = Join-Path $Reports "video_index_and_summary.json"
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if ($Summary.status -ne "READY_FOR_ARCHIVE") {
        throw "Submission report is preliminary; final report membership must be verified first"
    }
    foreach ($Name in @("results_table.md", "all_attempts.csv", "video_index_and_summary.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Reports $Name) -PathType Leaf)) {
            throw "Required submission artifact is missing: $Name"
        }
    }
    $RepoPrefix = $Repo.TrimEnd('\') + '\'
    if (-not $Reports.StartsWith($RepoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Report directory must be within the repository"
    }
    $A = [System.IO.Path]::GetFullPath($BackupA).TrimEnd('\')
    $B = [System.IO.Path]::GetFullPath($BackupB).TrimEnd('\')
    $AllRoots = @($Repo, $A, $B)
    for ($Left = 0; $Left -lt $AllRoots.Count; $Left++) {
        for ($Right = $Left + 1; $Right -lt $AllRoots.Count; $Right++) {
            $X = $AllRoots[$Left].TrimEnd('\') + '\'
            $Y = $AllRoots[$Right].TrimEnd('\') + '\'
            if ($X.StartsWith($Y, [StringComparison]::OrdinalIgnoreCase) -or
                $Y.StartsWith($X, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Repository and backup roots must be distinct and non-overlapping"
            }
        }
    }
    # Freeze the source manifest BEFORE either copy; matching copies alone do
    # not prove that all source bytes arrived intact.
    $SourceRoots = @($Evidence, $Reports,
        (Join-Path $Repo "data\real_robot\session_20260901\ipwm_preparations"),
        (Join-Path $Repo "src"), (Join-Path $Repo "scripts"),
        (Join-Path $Repo "config"), (Join-Path $Repo "hardware"))
    $SourceFiles = @($Audit, $Schedule, $Protocol,
        (Join-Path $Repo "runs\icra_confirmation_d3_query_selective_w10\seed27\model.pt"),
        (Join-Path $Repo "results\real_robot\push_axis_current_epoch_20260903.json"))
    foreach ($SourceRoot in $SourceRoots) {
        if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
            throw "Required archive source missing: $SourceRoot"
        }
        $SourceFiles += Get-ChildItem -LiteralPath $SourceRoot -File -Recurse |
            Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' } |
            Select-Object -ExpandProperty FullName
    }
    $SourceManifest = @($SourceFiles | Sort-Object -Unique | ForEach-Object {
        $Source = (Resolve-Path -LiteralPath $_).Path
        if (-not $Source.StartsWith($RepoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Archive source escaped repository: $Source"
        }
        $Item = Get-Item -LiteralPath $Source
        [pscustomobject]@{
            relative_path = $Source.Substring($RepoPrefix.Length)
            bytes = $Item.Length
            sha256 = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    foreach ($Destination in @($A, $B)) {
        if (Test-Path -LiteralPath $Destination) {
            throw "Refusing to overwrite existing backup destination: $Destination"
        }
        New-Item -ItemType Directory -Path $Destination | Out-Null
        foreach ($Entry in $SourceManifest) {
            $Source = Join-Path $Repo $Entry.relative_path
            $Copy = Join-Path $Destination $Entry.relative_path
            New-Item -ItemType Directory -Path (Split-Path -Parent $Copy) -Force | Out-Null
            Copy-Item -LiteralPath $Source -Destination $Copy
            if ((Get-Item -LiteralPath $Copy).Length -ne $Entry.bytes -or
                (Get-FileHash -LiteralPath $Copy -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Entry.sha256) {
                throw "Copy differs from frozen source: $Copy"
            }
        }
        $SourceManifest | Sort-Object relative_path | Export-Csv `
            -LiteralPath (Join-Path $Destination "SHA256SUMS.csv") -NoTypeInformation -Encoding utf8
    }

    $ManifestA = Import-Csv -LiteralPath (Join-Path $A "SHA256SUMS.csv")
    $ManifestB = Import-Csv -LiteralPath (Join-Path $B "SHA256SUMS.csv")
    $JsonA = $ManifestA | ConvertTo-Json -Compress
    $JsonB = $ManifestB | ConvertTo-Json -Compress
    if ($JsonA -ne $JsonB) { throw "The two backup manifests differ" }
    foreach ($Entry in $SourceManifest) {
        $Source = Join-Path $Repo $Entry.relative_path
        if ((Get-Item -LiteralPath $Source).Length -ne $Entry.bytes -or
            (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Entry.sha256) {
            throw "Source changed while archiving: $Source"
        }
    }
    $Receipt = [pscustomobject]@{
        status = "TWO_COPIES_VERIFIED_AGAINST_FROZEN_SOURCE"
        verified_utc = (Get-Date).ToUniversalTime().ToString("o")
        backup_a = $A
        backup_b = $B
        source_file_count = $SourceManifest.Count
        manifest_a_sha256 = (Get-FileHash -LiteralPath (Join-Path $A "SHA256SUMS.csv") -Algorithm SHA256).Hash
        manifest_b_sha256 = (Get-FileHash -LiteralPath (Join-Path $B "SHA256SUMS.csv") -Algorithm SHA256).Hash
        physical_shutdown_verified = $false
        ready_to_dismantle = $false
    }
    $ReceiptPath = Join-Path $Repo ("results\real_robot\archive_receipt_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")
    $Receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    Write-Host "PASS: both copies were checked against frozen source hashes; source bytes remained unchanged."
    Write-Host "Backup A: $A"
    Write-Host "Backup B: $B"
    Write-Host "The evidence archive gate is satisfied; perform physical home/torque-off separately."
}
finally {
    Pop-Location
}
