@echo off
REM Build executable

set ROOT="C:\Users\jnoaturn\OneDrive - University of Waterloo\Temp"

pyinstaller visualizer.py ^
  --onedir ^
  --windowed ^
  --name SeaIceMap ^
  --distpath %ROOT% ^
  --add-data "icons;icons" ^
  --noconfirm ^
  --clean

echo.
echo Build complete. Output is in "%ROOT%\SeaIceMap"
pause