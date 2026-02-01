@echo off
echo ====================================
echo Fixing Missing Dependencies
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

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing missing dependencies...
echo.

pip install requests==2.31.0

echo.
echo Verifying all dependencies are installed...
echo.

pip install -r requirements.txt

echo.
echo ====================================
echo Dependencies fixed!
echo ====================================
echo.
echo You can now run the application with:
echo   run_no_diarization.bat
echo.

pause
