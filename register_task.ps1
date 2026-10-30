# Windows 작업 스케줄러에 매일 07:30 실행 작업을 등록한다. (PowerShell에서 한 번 실행)
# 해제: Unregister-ScheduledTask -TaskName "NaerincheonSS_daily" -Confirm:$false
$bat = Join-Path $PSScriptRoot "run_daily.bat"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At 07:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "NaerincheonSS_daily" -Action $action -Trigger $trigger -Settings $settings -Description "내린천2 SS 대시보드 매일 갱신" -Force
