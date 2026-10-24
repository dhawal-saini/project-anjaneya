<#
.SYNOPSIS
    Install, report on, or remove the ANJANEYA Windows scheduled tasks.

.DESCRIPTION
    Registers two Task Scheduler jobs under the \Anjaneya folder - a weekday
    morning capture and a weekday post-close pipeline - each running through
    `scripts/run_scheduled.ps1`, which gates on `check_calendar.py` so a holiday
    is a clean no-op rather than a failed run.

    Both tasks are registered to:
        * run whether the user is logged on or not (S4U, no stored password),
        * run with the highest privileges,
        * ignore a new instance while one is still running (MultipleInstances=IgnoreNew),
        * stop after 8 hours (well past the longest session + pipeline),
        * start as soon as possible if the start time was missed (StartWhenAvailable).

    Usage:
        .\scripts\schedule_tasks.ps1 install
        .\scripts\schedule_tasks.ps1 install -MorningTime 09:10 -PostCloseTime 15:45
        .\scripts\schedule_tasks.ps1 status
        .\scripts\schedule_tasks.ps1 remove

    `install` and `remove` need an elevated PowerShell session.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('install', 'status', 'remove')]
    [string]$Action = 'status',
    [string]$MorningTime = '09:05',
    [string]$PostCloseTime = '15:45',
    [string]$TaskFolder = '\Anjaneya'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repoRoot 'scripts\run_scheduled.ps1'
$folderName = $TaskFolder.Trim('\')

$tasks = @(
    [pscustomobject]@{ Name = 'Anjaneya-Morning'; Stage = 'morning'; At = $MorningTime
        Description = 'ANJANEYA morning: calendar guard, universe check, Kite login, capture session.' },
    [pscustomobject]@{ Name = 'Anjaneya-PostClose'; Stage = 'postclose'; At = $PostCloseTime
        Description = 'ANJANEYA post-close: candles -> greeks -> features -> labels -> validate -> dataset -> train -> backtest.' }
)

function Get-AnjaneyaTasks {
    Get-ScheduledTask -TaskPath "\$folderName\*" -ErrorAction SilentlyContinue
}

function Install-AnjaneyaTasks {
    if (-not (Test-Path $runner)) { throw "runner not found: $runner" }
    $identity = "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
        -StartWhenAvailable `
        -DontStopOnIdleEnd `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries

    foreach ($task in $tasks) {
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Stage $($task.Stage)" `
            -WorkingDirectory $repoRoot
        $trigger = New-ScheduledTaskTrigger -Weekly `
            -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $task.At
        Register-ScheduledTask -TaskName $task.Name -TaskPath "\$folderName" `
            -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
            -Description $task.Description -Force | Out-Null
        Write-Host "installed : \$folderName\$($task.Name) ($($task.Stage) at $($task.At), Mon-Fri)" -ForegroundColor Green
    }
    Write-Host "principal : $identity (S4U, highest privileges, run whether logged on or not)" -ForegroundColor Green
}

function Show-AnjaneyaTasks {
    $found = Get-AnjaneyaTasks
    if (-not $found) {
        Write-Host "no ANJANEYA tasks under \$folderName - run '.\scripts\schedule_tasks.ps1 install'" -ForegroundColor Yellow
        return
    }
    foreach ($task in $found) {
        $info = Get-ScheduledTaskInfo -TaskName $task.TaskName -TaskPath $task.TaskPath
        Write-Host "task      : $($task.TaskPath)$($task.TaskName)"
        Write-Host "  state   : $($task.State)"
        Write-Host "  last run: $($info.LastRunTime) (result 0x$('{0:X}' -f $info.LastTaskResult))"
        Write-Host "  next run: $($info.NextRunTime)"
    }
}

function Remove-AnjaneyaTasks {
    $found = Get-AnjaneyaTasks
    if (-not $found) {
        Write-Host "no ANJANEYA tasks under \$folderName - nothing to remove" -ForegroundColor Yellow
        return
    }
    foreach ($task in $found) {
        Unregister-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath -Confirm:$false
        Write-Host "removed   : $($task.TaskPath)$($task.TaskName)" -ForegroundColor Green
    }
}

switch ($Action) {
    'install' { Install-AnjaneyaTasks }
    'status' { Show-AnjaneyaTasks }
    'remove' { Remove-AnjaneyaTasks }
}
