@echo off
cd /d "%~dp0backend"
call ..\\.venv\\Scripts\\activate.bat 2>nul || call ..\\venv\\Scripts\\activate.bat 2>nul
pip install -r requirements.txt --quiet
uvicorn main:app --reload --port 8000
pause
