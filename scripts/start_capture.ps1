<#
.SYNOPSIS
    Start an ANJANEYA-CAPTURE live session (PowerShell wrapper).

.DESCRIPTION
    Runs `scripts/daily_login.py` (unless -SkipLogin) and then the capture app.
    Anything after the recognised switches is forwarded to
    `python -m apps.anjaneeya_capture.run`, so every CLI flag keeps working:

        .\scripts\start_capture.ps1 --smoke
        .\scripts\start_capture.ps1 --until 15:30
        .\scripts\start_capture.ps1 -SkipLogin --dry-run

    For a one-command morning (browser login *and* the session, no copy/paste of
    the request_token) use `python scripts/login_and_start.py` instead; this
    wrapper stays useful when the token is already cached or when the login has
    to happen separately.

.EXAMPLE
    .\scripts\start_capture.ps1 -SkipLogin --dry-run
#>
[CmdletBinding()]
param(
    [switch]$SkipLogin,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CaptureArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Prefer the project environment, then `uv run`, then a bare `python`.
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

if (-not $SkipLogin) {
    Write-Host '==> ANJANEYA daily login' -ForegroundColor Cyan
    $loginArgs = $pre + @('scripts/daily_login.py')
    & $exe @loginArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "login exited with $LASTEXITCODE; capture will report any auth failure"
    }
}

Write-Host '==> ANJANEYA-CAPTURE session (mode=capture)' -ForegroundColor Cyan
$captureCommand = $pre + @('-m', 'apps.anjaneeya_capture.run', '--mode', 'capture') + $CaptureArgs
& $exe @captureCommand
exit $LASTEXITCODE

