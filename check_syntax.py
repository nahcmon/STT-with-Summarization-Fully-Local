"""
Script to check for syntax errors and basic lint issues
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        ast.parse(code)
        print(f"[OK] {file_path.name}: No syntax errors")
        return True
    except SyntaxError as e:
        print(f"[ERROR] {file_path.name}: Syntax error at line {e.lineno}")
        print(f"  {e.msg}")
        return False
    except Exception as e:
        print(f"[ERROR] {file_path.name}: Error - {e}")
        return False

def main():
    print("=" * 60)
    print("Checking Python files for syntax errors...")
    print("=" * 60)
    print()

    python_files = [
        Path("app.py"),
        Path("whisper_handler.py"),
        Path("llm_handler.py"),
        Path("verify_installation.py"),
    ]

    all_ok = True
    for file_path in python_files:
        if file_path.exists():
            if not check_syntax(file_path):
                all_ok = False
        else:
            print(f"[WARN] {file_path.name}: File not found")
            all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All files passed syntax check!")
    else:
        print("[FAILED] Some files have errors")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
