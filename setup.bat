@echo off
echo ====================================
echo Speech-to-Text Setup Script
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

echo Python found!
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Installing other dependencies...
echo Installing av (PyAV) separately to ensure pre-built wheel is used...
pip install av==13.1.0
if errorlevel 1 (
    echo ERROR: Failed to install av package
    echo.
    echo This might be due to missing pre-built wheels.
    echo Please install Microsoft Visual C++ Build Tools from:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo Or try installing an older version of Python (3.9 or 3.10)
    pause
    exit /b 1
)

echo.
echo Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ====================================
echo Setup complete!
echo ====================================
echo.
echo Running verification script...
python verify_installation.py
echo.
echo.
echo IMPORTANT: Before running the app, you need to:
echo 1. Install FFmpeg and add it to your PATH
echo 2. Set HF_TOKEN environment variable for speaker diarization
echo.
echo To set HF_TOKEN:
echo   set HF_TOKEN=your_huggingface_token
echo.
echo To run the application:
echo   run.bat
echo.
pause
