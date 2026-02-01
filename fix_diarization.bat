@echo off
echo ====================================
echo Fixing Speaker Diarization
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
echo This will fix the speaker diarization token authentication issue.
echo.

echo Step 1: Ensuring correct huggingface_hub version...
pip install "huggingface-hub>=0.23.0,<0.26.0"

echo.
echo Step 2: Verifying installation...
python -c "from huggingface_hub import login; print('HuggingFace Hub: OK')"

echo.
echo ====================================
echo Fix applied!
echo ====================================
echo.
echo Now you can run the app with:
echo   run.bat  (if HF_TOKEN is set as environment variable)
echo   run_with_token.bat  (to enter token interactively)
echo.
echo To verify HF_TOKEN is set:
echo   echo %%HF_TOKEN%%
echo.

pause
