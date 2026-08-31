@echo off
REM Builds a standalone OurionSpectra.exe that runs without Python installed.
REM Output goes to dist\OurionSpectra\OurionSpectra.exe

cd /d "%~dp0"

echo Installing build dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install pyinstaller --quiet

echo Building OurionSpectra.exe (this can take a minute)...
python -m PyInstaller ourionspectra.spec --noconfirm

echo.
echo Done. Find the app at: dist\OurionSpectra\OurionSpectra.exe
echo You can copy the whole "dist\OurionSpectra" folder anywhere and run it
echo without needing Python installed on that machine.
pause
