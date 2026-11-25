@echo off
cd /d "%~dp0"
echo ==============================================
echo             ResearchLens Setup
echo ==============================================
if not exist .venv (
    echo Creating Python environment...
    py -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing / updating packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Launching ResearchLens...
streamlit run app.py
pause
