<#
.SYNOPSIS
    Calendar-gated entry point for the ANJANEYA Windows scheduled tasks.

.DESCRIPTION
    A scheduled task that fires at a fixed time every weekday cannot know whether
    the exchange opens that day. This wrapper asks `scripts/check_calendar.py`
    first (`--require-trading-day --within 0`) and exits 0 without running anything
    when the answer is no - so a task firing on a holiday, a weekend or an
    unconfirmed holiday is a clean no-op rather than a failed run.

    Stages:
        morning    -> scripts/run_morning.py       (calendar -> universe -> login -> capture)
        postclose  -> scripts/run_daily_pipeline.py (candles -> ... -> backtest)

    Extra arguments are forwarded to the stage, e.g.:

        .\scripts\run_scheduled.ps1 -Stage postclose -- --skip backtest

    Exit codes: 0 ok (including a clean skip on a non-trading day), otherwise the
    stage's own exit code (1 failure, 2 configuration, 4 gate).
#>
[CmdletBinding()]
param(
    [ValidateSet('morning', 'postclose')]
    [string]$Stage = 'morning',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
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

# `--within 0` limits the risk window to today, so the check answers exactly one
# question: does today trade and is it confirmed? A non-zero exit is a skip.
$gateArgs = $pre + @('scripts/check_calendar.py', '--require-trading-day', '--within', '0')
& $exe @gateArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ANJANEYA: today is not a confirmed trading day - $Stage skipped" -ForegroundColor Yellow
    exit 0
}

if ($Stage -eq 'morning') {
    $stageArgs = $pre + @('scripts/run_morning.py') + $ExtraArgs
}
else {
    $stageArgs = $pre + @('scripts/run_daily_pipeline.py') + $ExtraArgs
}

Write-Host "ANJANEYA: running $Stage" -ForegroundColor Cyan
& $exe @stageArgs
exit $LASTEXITCODE
