<#
.SYNOPSIS
    Run the ANJANEYA weekly cadence pipeline (PowerShell wrapper).

.DESCRIPTION
    The daily chain plus the cadence steps that read the artefacts it leaves behind:
    candles -> greeks -> features -> labels -> validate -> dataset -> train -> backtest
    -> report -> explain -> importance -> charts -> accumulation. Extra arguments are
    forwarded to the pipeline script, e.g.:

        .\scripts\run_weekly.ps1
        .\scripts\run_weekly.ps1 --dry-run
        .\scripts\run_weekly.ps1 --date 2026-10-05 --force
        .\scripts\run_weekly.ps1 --skip charts,explain

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

Write-Host '==> ANJANEYA weekly pipeline (candles -> backtest -> report -> accumulation)' -ForegroundColor Cyan
$command = $pre + @('scripts\run_daily_pipeline.py', '--weekly') + $PipelineArgs
& $exe @command
exit $LASTEXITCODE
