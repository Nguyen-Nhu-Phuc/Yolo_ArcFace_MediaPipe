@echo off
rem Chay may chu web; trinh duyet tu mo khi trang san sang. Them tham so neu can, vd: chay.bat --port 5001
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua cai dat. Hay nhap dup cai_dat.bat truoc.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
set PORT=5000
:args
if "%~1"=="" goto :run
if "%~1"=="--port" set PORT=%~2
shift
goto :args

:run
rem Doi may chu san sang (lan dau co the mat vai phut de tai mo hinh) roi mo trinh duyet
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 300; $i++) { try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%PORT% -TimeoutSec 2 | Out-Null; Start-Process http://127.0.0.1:%PORT%; break } catch { Start-Sleep 2 } }"

echo Dang khoi dong... Dong cua so nay (hoac bam Ctrl+C) de dung chuong trinh.
".venv\Scripts\python.exe" app.py %*
pause
