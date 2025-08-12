@echo off
REM Set root folder for all build output
set ROOT=C:\Users\jnoaturn\OneDrive - University of Waterloo\Temp\SeaIceMap

REM Run PyInstaller
pyinstaller visualizer.py ^
  --onedir ^
  --windowed ^
  --name SeaIceMap ^
  --distpath "%ROOT%\dist" ^
  --workpath "%ROOT%\build" ^
  --specpath "%ROOT%\spec" ^
  --clean ^
  --noconfirm

echo.
echo Build complete. Output is in "%ROOT%\dist"
pause
