@echo off
REM ============================================================
REM CSIPBLLM — Combined RAG + RL Start Server (Windows)
REM Run every time you want to start the server:
REM   Double-click start.bat  OR  run in CMD
REM ============================================================

echo.
echo ==========================================
echo    CSIPBLLM Combined RAG + RL — Server
echo ==========================================
echo.

REM ── Check venv ───────────────────────────────────────────────
if not exist "venv\" (
    echo [ERROR] Virtual environment not found.
    echo         Run setup.bat first.
    pause
    exit /b 1
)

REM ── Activate venv ────────────────────────────────────────────
call venv\Scripts\activate.bat

REM ── Load .env ────────────────────────────────────────────────
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set line=%%a
        if not "%%a"=="" (
            if not "%%a:~0,1%"=="#" set "%%a=%%b"
        )
    )
    echo [OK] Environment variables loaded from .env
) else (
    echo [WARN] .env not found.
    echo        Create from .env.example and fill in API key.
)

REM ── Create directories ────────────────────────────────────────
if not exist "history_logs\" mkdir history_logs
if not exist "rl_logs\"      mkdir rl_logs
if not exist "rl_plots\"     mkdir rl_plots
if not exist "plots\"        mkdir plots
if not exist "logs\"         mkdir logs
if not exist "logs\eval_results\" mkdir logs\eval_results

REM ── Start server ─────────────────────────────────────────────
echo.
echo [START] Server     →  http://localhost:8000
echo [START] Swagger    →  http://localhost:8000/docs
echo [INFO]  Auto-reload active. Press Ctrl+C to stop.
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
