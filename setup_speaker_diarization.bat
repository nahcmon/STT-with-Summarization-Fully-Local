@echo off
echo ====================================
echo Speaker Diarization Setup
echo ====================================
echo.
echo This will help you set up speaker diarization (multi-speaker detection).
echo.
echo IMPORTANT: You need to accept terms for multiple HuggingFace models.
echo.
echo ====================================
echo Step 1: Get HuggingFace Token
echo ====================================
echo.
echo 1. Go to: https://huggingface.co/settings/tokens
echo 2. Click "New token"
echo 3. Name it "STT-App" (or anything you like)
echo 4. Select "Read" permission
echo 5. Click "Create token"
echo 6. COPY the token (starts with hf_)
echo.
echo Press any key when you have your token copied...
pause >nul

echo.
echo ====================================
echo Step 2: Accept Model Terms
echo ====================================
echo.
echo You need to visit these URLs and click "Agree and access repository" on each:
echo.
echo Opening browser windows...
echo.

start https://huggingface.co/pyannote/speaker-diarization-3.1
timeout /t 2 >nul
start https://huggingface.co/pyannote/segmentation-3.0
timeout /t 2 >nul
start https://huggingface.co/pyannote/segmentation
timeout /t 2 >nul

echo.
echo Browser windows opened. For EACH page:
echo   1. Log in if needed
echo   2. Click "Agree and access repository"
echo   3. Wait for confirmation
echo.
echo Press any key when you've accepted all terms...
pause >nul

echo.
echo ====================================
echo Step 3: Set Environment Variable
echo ====================================
echo.
echo Now we'll set your HF_TOKEN as a permanent environment variable.
echo.
echo Please enter your HuggingFace token (paste with right-click):
set /p USER_TOKEN="Token: "

if "%USER_TOKEN%"=="" (
    echo.
    echo ERROR: No token entered!
    pause
    exit /b 1
)

echo.
echo Setting HF_TOKEN environment variable...
setx HF_TOKEN "%USER_TOKEN%"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to set environment variable!
    echo.
    echo Please set it manually:
    echo   1. Press Win+R
    echo   2. Type: sysdm.cpl
    echo   3. Advanced -^> Environment Variables
    echo   4. User variables -^> New
    echo   5. Name: HF_TOKEN
    echo   6. Value: %USER_TOKEN%
    echo.
    pause
    exit /b 1
)

echo.
echo ====================================
echo Step 4: Test the Setup
echo ====================================
echo.
echo IMPORTANT: Close this window and open a NEW Command Prompt.
echo Environment variables only work in NEW windows!
echo.
echo In the new Command Prompt, run:
echo   echo %%HF_TOKEN%%
echo.
echo Should show your token starting with hf_
echo.
echo Then run the app:
echo   run.bat
echo.
echo Look for: "Successfully loaded speaker diarization pipeline"
echo.
echo ====================================
echo Setup complete!
echo ====================================
echo.
echo Remember: CLOSE this window and open a NEW Command Prompt!
echo.
pause
