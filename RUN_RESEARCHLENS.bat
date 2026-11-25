@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo First run setup with START_RESEARCHLENS.bat
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
