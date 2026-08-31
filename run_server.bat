@echo off
REM Double-click this file to run the OurionSpectra FastAPI Backend Server.
REM It always runs from the folder this .bat file is in.

cd /d "%~dp0"

echo Installing/checking dependencies...
python -m pip install -r requirements.txt --quiet

echo Starting OurionSpectra Backend Server...
echo API documentation will be available at http://127.0.0.1:8000/docs
python server.py --host 127.0.0.1 --port 8000

pause
