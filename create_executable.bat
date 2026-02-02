@echo off
REM ============================================
REM Build executable for SeaIceMap (PyInstaller)
REM ============================================

REM ---- Your chosen output root (OneDrive path) ----
set ROOT="Your_path\SeaIceMap-executable"

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

pyinstaller main.py ^
  --onedir ^
  --windowed ^
  --name SeaIceMap ^
  --distpath %ROOT% ^
  --add-data "icons;icons" ^
  --add-data "model;model" ^
  --add-data "landmask;landmask" ^
  --noconfirm ^
  --clean ^
  --hidden-import rasterio.serde ^
  --collect-submodules rasterio ^
  --collect-data rasterio ^
  --hidden-import fiona ^
  --collect-submodules fiona ^
  --collect-data fiona ^
  --collect-binaries fiona

echo.
echo ============================================
echo Build complete. Output is in: %ROOT%\SeaIceMap
echo ============================================
echo.
pause
