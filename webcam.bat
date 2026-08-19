@echo off
REM ===========================================================================
REM  Vantage - live webcam analysis
REM
REM  Double-click this file, or run it from a terminal. An optional first word
REM  picks what to run:
REM
REM      webcam.bat            pose    (default) people, skeletons, motion state
REM      webcam.bat pose       same, said explicitly
REM      webcam.bat objects    365-class detection + tracking, no pose
REM      webcam.bat plain      detection only, no tracking and no pose
REM
REM  The two modes use different detectors on purpose, and the reason is
REM  measured rather than assumed. Pose costs about 5 ms per person on the
REM  iGPU, but only when the detector leaves the GPU room: paired with
REM  yolox-tiny (10.8 ms) the whole pipeline holds 30 fps, while the 365-class
REM  dfine-s-obj365 (84 ms) needs the frame budget for itself.
REM
REM  Anything else you type is appended to the vantage command, so it
REM  overrides these defaults:
REM
REM      webcam.bat pose --pose-max-persons 2
REM      webcam.bat objects --classes person,laptop
REM      webcam.bat --source webcam:1 --no-hud
REM
REM  VANTAGE_MODEL, VANTAGE_DEVICE, VANTAGE_SOURCE and VANTAGE_INTERVAL change
REM  the defaults permanently without editing this file.
REM
REM  Press q or Esc in the video window to stop, h to toggle the HUD,
REM  s to save a snapshot.
REM ===========================================================================

setlocal

set "MODE=pose"
set "ARGS="

REM Parse a leading mode word, then collect everything else verbatim. %1 rather
REM than %~1 when appending, so a quoted path with spaces survives intact.
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="pose" (
    set "MODE=pose"
    shift
    goto parse
)
if /I "%~1"=="objects" (
    set "MODE=objects"
    shift
    goto parse
)
if /I "%~1"=="plain" (
    set "MODE=plain"
    shift
    goto parse
)
set "ARGS=%ARGS% %1"
shift
goto parse
:parsed

REM Per-mode defaults. An environment variable, if set, still wins over these.
if /I "%MODE%"=="pose" (
    set "MODE_MODEL=yolox-tiny"
    set "MODE_INTERVAL=1"
    set "MODE_FLAGS=--track --pose"
)
if /I "%MODE%"=="objects" (
    set "MODE_MODEL=dfine-s-obj365"
    set "MODE_INTERVAL=2"
    set "MODE_FLAGS=--track"
)
if /I "%MODE%"=="plain" (
    set "MODE_MODEL=yolox-tiny"
    set "MODE_INTERVAL=1"
    set "MODE_FLAGS=--detect"
)

if not defined VANTAGE_SOURCE   set "VANTAGE_SOURCE=webcam:0"
if not defined VANTAGE_DEVICE   set "VANTAGE_DEVICE=gpu"
if not defined VANTAGE_MODEL    set "VANTAGE_MODEL=%MODE_MODEL%"
if not defined VANTAGE_INTERVAL set "VANTAGE_INTERVAL=%MODE_INTERVAL%"

REM Run from the folder this script lives in, so the relative "models" cache is
REM found however the script was launched. No absolute path is baked in: the
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
    echo       .venv\Scripts\python.exe -m pip install -e ".[dev,detect]"
    echo.
    set "RESULT=1"
    goto finish
)

REM Echo the real command rather than the defaults: appended arguments are
REM allowed to override, and a banner built from the variables above would
REM report the wrong model whenever they did.
set "VARGS=run --source %VANTAGE_SOURCE% %MODE_FLAGS% --model %VANTAGE_MODEL% --device %VANTAGE_DEVICE% --detect-interval %VANTAGE_INTERVAL%%ARGS%"
echo [%MODE%] vantage %VARGS%
echo Press q or Esc in the video window to stop, h for the HUD, s for a snapshot.
echo.

"%PYTHON%" -m vantage %VARGS%
set "RESULT=%ERRORLEVEL%"

:finish
popd

REM Hold the window open on failure only. Explorer closes the console the
REM instant the script ends, which would take the error message with it; a
REM successful run has nothing left to read. Sniffing %cmdcmdline% to detect a
REM double-click was tried and rejected - it also fires when PowerShell shells
REM out to cmd, so every ordinary terminal run ended with "press any key".
if not "%RESULT%"=="0" (
    echo.
    echo   Exited with code %RESULT%.
    pause
)

exit /b %RESULT%
