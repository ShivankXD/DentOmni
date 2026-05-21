@echo off
title DentOmni — Frontend (Local Server)
color 0B

echo.
echo  ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗███╗   ██╗██╗
echo  ██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║████╗  ██║██║
echo  ██║  ██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██╔██╗ ██║██║
echo  ██║  ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║╚██╗██║██║
echo  ██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║
echo  ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝
echo.
echo  [AI Dental Diagnostics — Frontend Server]
echo  ==========================================
echo.

cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
  echo  [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)

:: Check if the main HTML file exists
if not exist "hero-animation.html" (
  echo  [ERROR] hero-animation.html not found in project root.
  pause
  exit /b 1
)

echo  [*] Serving DentOmni frontend on http://localhost:3000
echo  [*] Opening browser automatically...
echo  [*] Press Ctrl+C to stop the server.
echo.

:: Open in browser after 1.5s (start is non-blocking, timeout gives server time to bind)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:3000/hero-animation.html"

:: Serve the current directory
python -m http.server 3000 --bind 127.0.0.1

pause
