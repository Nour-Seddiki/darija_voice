# Start (or resume) the Omni-7B Darija fine-tune as a detached background process.
# Survives closing the terminal / Claude app. Safe to re-run: it resumes from <run>\last.
#   .\run_train.ps1                    # default run name
#   .\run_train.ps1 -Run omni7b_asr_v2 -Extra "--lr 5e-5"
param(
    [string]$Run = "omni7b_asr_v1",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "..\.venv\Scripts\python.exe"
$logDir = "C:\Users\seddi\darija_data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $logDir "train_${Run}_$stamp.log"
$err = Join-Path $logDir "train_${Run}_$stamp.err.log"
$argList = "-u -m omni.train_omni --run $Run --resume $Extra"
$p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
    -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
Write-Output "started PID $($p.Id)"
Write-Output "log:    $out"
Write-Output "watch:  Get-Content '$out' -Tail 20 -Wait"
Write-Output "stop:   Stop-Process -Id $($p.Id)"
