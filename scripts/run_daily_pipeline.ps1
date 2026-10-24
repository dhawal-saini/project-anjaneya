<#
.SYNOPSIS
    Run the ANJANEYA daily post-close pipeline (PowerShell wrapper).

.DESCRIPTION
    One command for every trading day after the 15:30 close: candles -> greeks ->
    features -> labels -> validate -> dataset -> train -> backtest. Extra arguments
    are forwarded to the pipeline script, e.g.:

        .\scripts\run_daily_pipeline.ps1
        .\scripts\run_daily_pipeline.ps1 --dry-run
        .\scripts\run_daily_pipeline.ps1 --date 2026-10-01 --force
        .\scripts\run_daily_pipeline.ps1 --skip validate,backtest

    Exit codes: 0 ok, 1 a step failed, 2 configuration, 4 below the training gate.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
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

Write-Host '==> ANJANEYA daily pipeline (candles -> backtest)' -ForegroundColor Cyan
$command = $pre + @('scripts\run_daily_pipeline.py') + $PipelineArgs
& $exe @command
exit $LASTEXITCODE
