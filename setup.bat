@echo off
REM ============================================================
REM CSIPBLLM — Combined RAG + RL Setup (Windows)
REM Run once on first install:
REM   Double-click setup.bat  OR  run in CMD
REM ============================================================

echo.
echo ==========================================
echo    CSIPBLLM Combined RAG + RL — Setup
echo ==========================================
echo.

REM ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

REM ── Create virtual environment ────────────────────────────────
if not exist "venv\" (
    echo.
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment 'venv\' created.
) else (
    echo [OK] Virtual environment already exists, skipping.
)

REM ── Activate and install packages ─────────────────────────────
echo.
echo [INFO] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] All dependencies installed.

REM ── Create .env from .env.example ─────────────────────────────
if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo [INFO] '.env' created from '.env.example'
    echo        REQUIRED: Open '.env' and fill in your OPENAI_API_KEY!
) else (
    echo [OK] '.env' already exists.
)

REM ── Create required directories ───────────────────────────────
if not exist "history_logs\" mkdir history_logs
if not exist "rl_logs\"      mkdir rl_logs
if not exist "rl_plots\"     mkdir rl_plots
if not exist "plots\"        mkdir plots
if not exist "logs\"         mkdir logs
if not exist "logs\eval_results\" mkdir logs\eval_results
echo [OK] Directories ready.

echo.
echo ==========================================
echo    Setup complete! Next steps:
echo.
echo    1. Edit .env
echo       Fill in OPENAI_API_KEY
echo.
echo    2. Run start.bat
echo.
echo    3. Open browser: http://localhost:8000
echo    4. API Docs:     http://localhost:8000/docs
echo ==========================================
echo.
pause
