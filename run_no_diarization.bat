@echo off
echo ====================================
echo Starting App (NO Speaker Diarization)
echo ====================================
echo.
echo This script runs the app WITHOUT speaker diarization.
echo All transcriptions will show "Unknown" for the speaker.
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

REM Start the application
echo.
echo ====================================
echo Starting server...
echo ====================================
echo.
echo Server will be available at: http://localhost:8000
echo Open your browser and navigate to http://localhost:8000
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
    echo.
    echo Try running: python verify_installation.py
    echo.
)

pause
