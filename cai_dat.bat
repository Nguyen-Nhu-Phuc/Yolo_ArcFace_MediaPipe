@echo off
rem Cai thu vien lan dau: tao moi truong ao .venv trong thu muc du an roi cai theo requirements.txt
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [LOI] Chua cai Python. Tai Python 3.12 64-bit tai https://www.python.org/downloads/
    echo       Khi cai, nho tich chon "Add python.exe to PATH".
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 (
    echo [LOI] Can Python 3.12. Phien ban dang co:
    python --version
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Dang tao moi truong ao .venv ...
    python -m venv .venv || goto :fail
)
echo Dang cai thu vien (10-20 phut, can Internet) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail

echo.
echo Cai dat xong. Nhap dup chay.bat de chay chuong trinh.
pause
exit /b 0

:fail
echo.
echo [LOI] Cai dat that bai. Neu loi khi cai insightface: cai "Build Tools for Visual Studio" (Desktop development with C++) roi chay lai.
pause
exit /b 1
