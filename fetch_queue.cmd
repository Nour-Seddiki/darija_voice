@echo off
rem Long download queue (resumable). Start it detached with fetch_queue.ps1.
rem Order = what unblocks the most work first. Re-running skips finished files/shards.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "PY=%~dp0..\.venv\Scripts\python.exe"
set "LOG=C:\Users\seddi\darija_data\logs\fetch_queue.log"

call :step -u -m common.robust_download model Qwen/Qwen2.5-Omni-7B
call :step -u -m asr.prepare_data --sources kahwa rawi yasmine
call :step -u -m asr.prepare_data --sources siwis_fr cv_ar mls_fr
call :step -u -m asr.prepare_data --sources loubna
call :step -u -m asr.prepare_data --sources clartts_ar
echo === %date% %time% QUEUE DONE>> "%LOG%"
exit /b

:step
echo === %date% %time% START %*>> "%LOG%"
"%PY%" %* >> "%LOG%" 2>&1
echo === %date% %time% END exit %errorlevel% : %*>> "%LOG%"
exit /b
