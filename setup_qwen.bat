@echo off
echo ========================================
echo  Qwen3-ASR Setup
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate

echo Installing Qwen3-ASR package...
pip install qwen-asr

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install qwen-asr
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Flash Attention Setup (Optional)
echo ========================================
echo.
echo Flash Attention significantly improves Qwen3-ASR performance
echo but requires MSVC C++ build tools...
echo.

REM Check for MSVC build tools
set "MSVC_FOUND=0"

REM Method 1: Check for vswhere (Visual Studio installer tool)
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >nul 2>&1
    if not errorlevel 1 (
        set "MSVC_FOUND=1"
        echo [OK] MSVC build tools detected
    )
)

REM Method 2: Try to find cl.exe in common locations
if "%MSVC_FOUND%"=="0" (
    where cl.exe >nul 2>&1
    if not errorlevel 1 (
        set "MSVC_FOUND=1"
        echo [OK] MSVC compiler (cl.exe) found in PATH
    )
)

if "%MSVC_FOUND%"=="0" (
    echo [!] MSVC build tools not detected
    echo.
    echo Attempting to install via winget...
    echo.

    REM Check if winget is available
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: winget not available
        echo.
        echo Please install Visual Studio Build Tools manually:
        echo 1. Download from: https://aka.ms/vs/17/release/vs_BuildTools.exe
        echo 2. Run the installer
        echo 3. Select "Desktop development with C++"
        echo 4. Install and restart this script
        echo.
        pause
        exit /b 1
    )

    echo Installing Visual Studio Build Tools (C++)...
    echo This may take 10-20 minutes and requires ~6GB download...
    echo.

    winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

    if errorlevel 1 (
        echo.
        echo ERROR: Automated installation failed
        echo.
        echo Please install manually:
        echo 1. Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
        echo 2. Select "Desktop development with C++"
        echo 3. Restart this script after installation
        echo.
        pause
        exit /b 1
    )

    echo.
    echo [OK] Build tools installed successfully
    echo Restarting script to use new tools...
    echo.
    pause

    REM Restart the script to pick up new environment
    call "%~f0"
    exit /b
)

echo.
echo Installing flash-attention...
echo This compilation may take 5-10 minutes...
echo.

REM Set up MSVC environment if needed
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            echo Setting up MSVC environment...
            call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
        )
    )
)

REM Install flash-attn with proper flags
pip install flash-attn --no-build-isolation

if errorlevel 1 (
    echo.
    echo ========================================
    echo  WARNING: Flash Attention Failed
    echo ========================================
    echo.
    echo Qwen3-ASR will still work, but may be slower.
    echo.
    echo Troubleshooting:
    echo 1. Ensure you have NVIDIA GPU with Compute Capability 7.0+
    echo 2. Verify CUDA is installed (nvcc --version)
    echo 3. Try running this script as Administrator
    echo.
    echo You can continue without flash-attention.
    echo.
) else (
    echo.
    echo [OK] Flash Attention installed successfully!
    echo Qwen3-ASR will run with optimal performance.
    echo.
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Qwen3-ASR models will download automatically on first use:
echo   - Qwen3-ASR-0.6B: ~1.2GB
echo   - Qwen3-ASR-1.7B: ~3.4GB
echo   - ForcedAligner-0.6B: ~1.2GB
echo.
echo Total download on first use: ~5.8GB
echo.
echo Models are cached in: %USERPROFILE%\.cache\huggingface\hub
echo.
pause
