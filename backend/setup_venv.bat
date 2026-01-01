@echo off
echo =====================================
echo Setting up Python Virtual Environment
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv

if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/3] Installing Python dependencies...
echo This may take several minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo =====================================
echo Setup Complete!
echo =====================================
echo.
echo To start the FastAPI server:
echo   1. Activate the virtual environment: venv\Scripts\activate
echo   2. Run the server: python -m uvicorn server:app --reload --port 8000
echo.
echo The server will be available at: http://localhost:8000
echo API documentation will be at: http://localhost:8000/docs
echo.
pause
