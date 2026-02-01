@echo off
echo ====================================
echo Speech-to-Text - Quick Start
echo ====================================
echo.
echo This script will help you set your HF_TOKEN and run the app.
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo Please enter your HuggingFace token.
echo.
echo If you don't have one:
echo   1. Go to https://huggingface.co/settings/tokens
echo   2. Create a new token (read access is enough)
echo   3. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
echo.
echo Or press Ctrl+C to exit and run without speaker diarization.
echo.

set /p HF_TOKEN="Enter your HF_TOKEN: "

if "%HF_TOKEN%"=="" (
    echo.
    echo No token entered. Running without speaker diarization...
    echo.
) else (
    echo.
    echo Token set for this session!
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the application
echo.
echo ====================================
echo Starting server...
echo ====================================
echo.
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ====================================
echo.

python app.py

REM If we get here, the app has stopped
echo.
echo ====================================
echo Server stopped
echo ====================================
echo.

if errorlevel 1 (
    echo.
    echo ERROR: The application exited with an error!
    echo See above for error details.
    echo.
)

pause
