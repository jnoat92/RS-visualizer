@echo off
REM ============================================
REM Build executable for SeaIceMap (PyInstaller)
REM ============================================

REM ---- Your chosen output root (OneDrive path) ----
set ROOT="C:\Users\kev80\OneDrive\Documents\Coop-w26\SeaIceMap-executable"

echo Cleaning previous build...

REM ---- Delete old dist folder ----
if exist %ROOT%\SeaIceMap (
    echo Removing old dist folder: %ROOT%\SeaIceMap
    rmdir /S /Q %ROOT%\SeaIceMap
)

REM ---- Delete old build folder ----
if exist build\SeaIceMap (
    echo Removing old build folder: .\build\SeaIceMap
    rmdir /S /Q build\SeaIceMap
)

echo.
echo Starting PyInstaller build...

pyinstaller visualizer.py ^
  --onedir ^
  --windowed ^
  --name SeaIceMap ^
  --distpath %ROOT% ^
  --add-data "icons;icons" ^
  --noconfirm ^
  --clean

echo.
echo ============================================
echo Build complete. Output is in: %ROOT%\SeaIceMap
echo ============================================
echo.
pause
