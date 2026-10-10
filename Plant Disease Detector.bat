@echo off
cd /d "%~dp0"
chcp 65001 >nul
title Plant Disease Detector (By Robin John)

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: تشغيل التطبيق تلقائياً عند فتح الملف لأول مرة
call :START_APP

:MENU
cls
echo ===================================================
echo     Plant Disease Detector (By Robin John)
echo ===================================================
echo.
echo  [1] Restart   (Restart the application)
echo  [2] Pause     (Pause the application)
echo  [3] Show Log  (View logs)
echo  [4] Exit      (Exit)
echo.
echo ===================================================
set /p choice="Select an option (1-4): "

if "%choice%"=="1" goto RESTART
if "%choice%"=="2" goto PAUSE_APP
if "%choice%"=="3" goto SHOW_LOG
if "%choice%"=="4" exit
goto MENU

:START_APP
echo [*] Cleaning cache and stopping any previous process...
if exist "app\__pycache__" rd /s /q "app\__pycache__"
taskkill /f /im python.exe >nul 2>&1

echo [*] Checking Python environment...
if not exist "python_embed\python.exe" (
    echo [ERROR] python_embed folder not found!
    pause
    exit
)

echo [*] Launching application in the background...
start /b "" "python_embed\python.exe" -m streamlit run app\planet_app.py > app_log.txt 2>&1
echo [+] Application started successfully!
timeout /t 2 >nul
goto :eof

:RESTART
echo.
call :START_APP
goto MENU

:PAUSE_APP
echo.
echo [*] Pausing the application...
taskkill /f /im python.exe >nul 2>&1
echo [+] Application paused successfully.
timeout /t 2 >nul
goto MENU

:SHOW_LOG
echo.
if exist app_log.txt (
    start notepad app_log.txt
) else (
    echo [!] No log file is currently available.
    timeout /t 2 >nul
)
goto MENU