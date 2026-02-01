@echo off
setlocal enabledelayedexpansion

echo ====================================
echo Starting Speech-to-Text Application
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if HF_TOKEN is set
if defined HF_TOKEN (
    echo.
    echo [OK] HF_TOKEN is set - speaker diarization will work
    echo.
) else (
    echo.
    echo ============================================================
    echo WARNING: HF_TOKEN environment variable is NOT set
    echo ============================================================
    echo.
    echo Speaker diarization will NOT work without it.
    echo.
    echo To set it PERMANENTLY:
    echo   1. Press Win+R, type: sysdm.cpl
    echo   2. Advanced tab -^> Environment Variables
    echo   3. User variables -^> New
    echo   4. Name: HF_TOKEN
    echo   5. Value: your_token_from_huggingface.co
    echo   6. OK, then restart Command Prompt
    echo.
    echo Get token: https://huggingface.co/settings/tokens
    echo Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
    echo.
    echo ============================================================
    echo.
    echo Press any key to continue WITHOUT speaker diarization...
    echo Or close this window to exit and set the token first.
    echo.
    pause >nul
)

REM Start the application
echo.
echo ====================================
echo Starting server...
echo ====================================
echo.
echo Server will be available at: http://localhost:8000
echo Open your browser: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ====================================
echo.

python app.py
set APP_EXIT_CODE=%errorlevel%

REM If we get here, the app has stopped
echo.
echo ====================================
echo Server stopped
echo ====================================
echo.

if %APP_EXIT_CODE% NEQ 0 (
    echo.
    echo ERROR: Application exited with error code: %APP_EXIT_CODE%
    echo.
    echo Common issues:
    echo - Port 8000 already in use: Try closing other programs
    echo - Missing dependencies: Run fix_dependencies.bat
    echo - CUDA issues: Run python verify_installation.py
    echo.
    echo For detailed troubleshooting, see TROUBLESHOOTING.md
    echo.
) else (
    echo Application closed normally.
)

echo.
pause
