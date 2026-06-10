@echo off
setlocal

set VS_VCVARS="C:\Programme\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_EXE="C:\Programme\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set NINJA_PATH=C:\Programme\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja
set PATH=%NINJA_PATH%;%PATH%
set OPENCV_DLL=H:\Programmieren\opencv\build\x64\vc16\bin

if not exist %VS_VCVARS% (
    echo FEHLER: Visual Studio vcvars64.bat nicht gefunden.
    pause & exit /b 1
)

call %VS_VCVARS%

if not exist build mkdir build
cd build

%CMAKE_EXE% .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo FEHLER: CMake-Konfiguration fehlgeschlagen.
    pause & exit /b 1
)

%CMAKE_EXE% --build . --config Release
if errorlevel 1 (
    echo FEHLER: Build fehlgeschlagen.
    pause & exit /b 1
)

echo.
echo Build erfolgreich. Kopiere OpenCV-DLL...
copy "%OPENCV_DLL%\opencv_world411.dll" . >nul 2>&1
copy "%OPENCV_DLL%\opencv_world411d.dll" . >nul 2>&1

echo.
echo Fertig. Aufruf:
echo   build\ball_detector_demo.exe              (Webcam)
echo   build\ball_detector_demo.exe video.mp4    (Videodatei)
echo.
pause
