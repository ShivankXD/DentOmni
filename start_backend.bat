@echo off
title DentOmni — Backend (FastAPI)
color 0B

echo.
echo  ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗███╗   ██╗██╗
echo  ██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║████╗  ██║██║
echo  ██║  ██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██╔██╗ ██║██║
echo  ██║  ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║╚██╗██║██║
echo  ██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║
echo  ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝
echo.
echo  [AI Dental Diagnostics — FastAPI Backend]
echo  ==========================================
echo.

:: Move to project root (where the .pth files live)
cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
  echo  [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)

:: Kill any existing process on port 8000
echo  [*] Stopping any existing backend on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING" 2^>nul') do (
  taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: Check if uvicorn is installed
python -m uvicorn --version >nul 2>&1
if errorlevel 1 (
  echo  [INFO] uvicorn not found. Installing requirements...
  pip install -r backend\requirements.txt
  echo.
)

:: Check model weights
if not exist "FINAL_dental_model.pth" (
  echo  [WARNING] FINAL_dental_model.pth not found in project root.
  echo            The Faster R-CNN model will fail until the weights are placed here.
  echo.
)
if not exist "FINAL_resnet50.pth" (
  echo  [WARNING] FINAL_resnet50.pth not found in project root.
  echo            The ResNet-50 model will fail until the weights are placed here.
  echo.
)

echo  [*] Starting DentOmni API on http://localhost:8000
echo  [*] API docs available at http://localhost:8000/docs
echo  [*] Press Ctrl+C to stop the server.
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
