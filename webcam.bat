@echo off
REM ===========================================================================
REM  Vantage - live webcam detection + tracking
REM
REM  Double-click this file, or run it from a terminal.  Anything you type
REM  after it is appended to the vantage command, so it overrides the defaults
REM  below:
REM
REM      webcam.bat --model yolox-tiny --detect-interval 1
REM      webcam.bat --classes person --no-hud
REM      webcam.bat --source webcam:1
REM
REM  The defaults can also be set as environment variables (VANTAGE_MODEL,
REM  VANTAGE_DEVICE, VANTAGE_SOURCE, VANTAGE_INTERVAL) so you can change them
REM  permanently without editing this file.
REM
REM  Press q or Esc in the video window to stop.
REM ===========================================================================

setlocal

if not defined VANTAGE_SOURCE   set "VANTAGE_SOURCE=webcam:0"
if not defined VANTAGE_MODEL    set "VANTAGE_MODEL=dfine-s-obj365"
if not defined VANTAGE_DEVICE   set "VANTAGE_DEVICE=gpu"
if not defined VANTAGE_INTERVAL set "VANTAGE_INTERVAL=2"

REM Run from the folder this script lives in, so the relative "models" cache is
REM found however the script was launched.  No absolute path is baked in: the
REM script works from any clone of the repository.
pushd "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo.
    echo   No virtual environment found at:
    echo       %PYTHON%
    echo.
    echo   Create one first:
    echo       py -3 -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -e .
    echo.
    set "RESULT=1"
    goto :finish
)

REM Echo the real command rather than the defaults: arguments passed to this
REM script are appended, and a later --model or --device silently wins, so a
REM banner built from the variables above would report the wrong model.
set "ARGS=run --source %VANTAGE_SOURCE% --track --model %VANTAGE_MODEL% --device %VANTAGE_DEVICE% --detect-interval %VANTAGE_INTERVAL% %*"
echo vantage %ARGS%
echo Press q or Esc in the video window to stop.
echo.

"%PYTHON%" -m vantage %ARGS%
set "RESULT=%ERRORLEVEL%"

:finish
popd

REM Hold the window open on failure only.  Explorer closes the console the
REM instant the script ends, which would take the error message with it; a
REM successful run has nothing left to read.  Sniffing %cmdcmdline% to detect a
REM double-click was tried and rejected - it also fires when PowerShell shells
REM out to cmd, so every ordinary terminal run ended with "press any key".
if not "%RESULT%"=="0" (
    echo.
    echo   Exited with code %RESULT%.
    pause
)

exit /b %RESULT%
