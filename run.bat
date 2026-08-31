@echo off
REM Double-click this file to run OurionSpectra on Windows.
REM It always runs from the folder this .bat file is in, so path
REM problems from PowerShell "cd" mistakes can't happen.

cd /d "%~dp0"

echo Installing/checking dependencies...
python -m pip install -r requirements.txt --quiet

echo Starting OurionSpectra...
python main.py

pause
