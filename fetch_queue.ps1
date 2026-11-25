# Start fetch_queue.cmd hidden and detached (survives closing the terminal / Claude app).
#   .\fetch_queue.ps1
#   Get-Content C:\Users\seddi\darija_data\logs\fetch_queue.log -Tail 20 -Wait
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force "C:\Users\seddi\darija_data\logs" | Out-Null
$p = Start-Process cmd.exe -ArgumentList "/c", "`"$here\fetch_queue.cmd`"" -WindowStyle Hidden -PassThru
Write-Output "queue started (cmd PID $($p.Id)); log: C:\Users\seddi\darija_data\logs\fetch_queue.log"
