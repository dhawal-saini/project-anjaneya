<#
.SYNOPSIS
    Run the ANJANEYA morning routine (PowerShell wrapper).

.DESCRIPTION
    The one command for the start of a trading day: guard the calendar, verify the
    option universe resolves, mint a fresh Kite token and start the capture
    session. Extra arguments are forwarded to the Python entry point, e.g.:

        .\scripts\run_morning.ps1
        .\scripts\run_morning.ps1 --refresh-instruments
        .\scripts\run_morning.ps1 --dry-run
        .\scripts\run_morning.ps1 -- --smoke --until 15:30

    Exit codes: 0 ok (a clean NO_SESSION on a holiday is a 0), 1 login or capture
    failed, 2 configuration, 4 the universe resolved to zero subscriptions.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MorningArgs
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

Write-Host '==> ANJANEYA morning routine (calendar -> universe -> login -> capture)' -ForegroundColor Cyan
$command = $pre + @('scripts\run_morning.py') + $MorningArgs
& $exe @command
exit $LASTEXITCODE
