"""
Verification script to check if all dependencies are properly installed
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"[OK] {package_name} is installed")
        return True
    except ImportError:
        print(f"[ERROR] {package_name} is NOT installed")
        return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available (Device: {torch.cuda.get_device_name(0)})")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
            return True
        else:
            print("[ERROR] CUDA is NOT available")
            print("  PyTorch is installed but CUDA support is not detected")
            return False
    except ImportError:
        print("[ERROR] PyTorch is NOT installed")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"[OK] FFmpeg is installed: {version_line}")
            return True
        else:
            print("[ERROR] FFmpeg is NOT working properly")
            return False
    except FileNotFoundError:
        print("[ERROR] FFmpeg is NOT installed or not in PATH")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking FFmpeg: {e}")
        return False

def check_hf_token():
    """Check if HuggingFace token is set"""
    import os
    token = os.getenv("HF_TOKEN")
    if token:
        print("[OK] HF_TOKEN environment variable is set")
        return True
    else:
        print("[WARN] HF_TOKEN environment variable is NOT set")
        print("  Speaker diarization will not work without it")
        return False

def main():
    print("=" * 60)
    print("Speech-to-Text Installation Verification")
    print("=" * 60)
    print()

    all_ok = True

    print("Checking Python packages...")
    print("-" * 60)

    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("jinja2", "Jinja2"),
        ("aiohttp", "aiohttp"),
        ("faster_whisper", "faster-whisper"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("numpy", "NumPy"),
    ]

    for module, name in packages:
        if not check_import(module, name):
            all_ok = False

    print()
    print("Checking PyTorch and CUDA...")
    print("-" * 60)
    if not check_cuda():
        all_ok = False

    print()
    print("Checking pyannote.audio...")
    print("-" * 60)
    if not check_import("pyannote.audio", "pyannote.audio"):
        print("  Speaker diarization will be disabled")

    print()
    print("Checking Qwen3-ASR (optional)...")
    print("-" * 60)
    if check_import("qwen_asr", "qwen-asr"):
        print("  Qwen3-ASR models available:")
        print("    - Qwen3-ASR-0.6B (2GB VRAM, fast)")
        print("    - Qwen3-ASR-1.7B (5GB VRAM, accurate)")
        try:
            import flash_attn
            print("  [OK] flash-attention installed (optimized performance)")
        except ImportError:
            print("  [INFO] flash-attention not installed (optional)")
            print("    Install with: pip install flash-attn --no-build-isolation")
    else:
        print("  Qwen3-ASR not installed (optional)")
        print("  Install with: pip install qwen-asr")
        print("  Or run: setup_qwen.bat")

    print()
    print("Checking FFmpeg...")
    print("-" * 60)
    if not check_ffmpeg():
        all_ok = False

    print()
    print("Checking HuggingFace token...")
    print("-" * 60)
    check_hf_token()

    print()
    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All critical dependencies are installed!")
        print("  You can run the application with: python app.py")
    else:
        print("[FAILED] Some critical dependencies are missing")
        print("  Please install missing dependencies before running the app")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
