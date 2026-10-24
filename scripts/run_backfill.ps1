<#
.SYNOPSIS
    Run the ANJANEYA index-history backfill (PowerShell wrapper).

.DESCRIPTION
    Fetches the long NIFTY index history plus the recent top-up and upserts it
    into DuckDB. Extra arguments are forwarded, e.g.:

        .\scripts\run_backfill.ps1 --dry-run
        .\scripts\run_backfill.ps1 --force-refresh
        .\scripts\run_backfill.ps1 --years 2        # ~2 years, overriding the config

    To bootstrap candles for the option contracts Kite currently lists, run the
    capture app directly with `--mode backfill-options` (see scripts/runbook.md).
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BackfillArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    $exe = $venvPython
    $pre = @()
}
elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    $exe = 'uv'
    $pre = @('run', 'python')
}
else {
    $exe = 'python'
    $pre = @()
}

Write-Host '==> ANJANEYA-CAPTURE index backfill (mode=backfill)' -ForegroundColor Cyan
$command = $pre + @('-m', 'apps.anjaneeya_capture.run', '--mode', 'backfill') + $BackfillArgs
& $exe @command
exit $LASTEXITCODE
