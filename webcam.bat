@echo off
REM ===========================================================================
REM  Vantage - live webcam analysis
REM
REM  Double-click this file, or run it from a terminal. An optional first word
REM  picks what to run:
REM
REM      webcam.bat            pose     (default) people, skeletons, motion state
REM      webcam.bat pose       same, said explicitly
REM      webcam.bat activity   pose plus activity recognition, tuned to demo
REM      webcam.bat objects    365-class detection + tracking, no pose
REM      webcam.bat plain      detection only, no tracking and no pose
REM      webcam.bat identity   tracking plus face identification (opt-in)
REM      webcam.bat enroll     add a person to the identity gallery
REM      webcam.bat who        list who is enrolled
REM      webcam.bat checks     no camera: score tracker, activity and spatial
REM
REM  The detection modes use different detectors on purpose, and the reason is
REM  measured rather than assumed. Pose costs about 5 ms per person on the
REM  iGPU, but only when the detector leaves the GPU room: paired with
REM  yolox-tiny (10.8 ms) the whole pipeline holds 30 fps, while the 365-class
REM  dfine-s-obj365 (84 ms) needs the frame budget for itself.
REM
REM  Anything else you type is appended to the vantage command, so it
REM  overrides these defaults:
REM
REM      webcam.bat activity --set activity.loiter_s=20
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
if /I "%~1"=="activity" (
    set "MODE=activity"
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
if /I "%~1"=="identity" (
    set "MODE=identity"
    shift
    goto parse
)
if /I "%~1"=="enroll" (
    set "MODE=enroll"
    shift
    goto parse
)
if /I "%~1"=="who" (
    set "MODE=who"
    shift
    goto parse
)
if /I "%~1"=="checks" (
    set "MODE=checks"
    shift
    goto parse
)
set "ARGS=%ARGS% %1"
shift
goto parse
:parsed

REM Per-mode defaults. An environment variable, if set, still wins over these.
set "MODE_EXTRA="
if /I "%MODE%"=="pose" (
    set "MODE_MODEL=yolox-tiny"
    set "MODE_INTERVAL=1"
    set "MODE_FLAGS=--track --pose"
)
if /I "%MODE%"=="activity" (
    set "MODE_MODEL=yolox-tiny"
    set "MODE_INTERVAL=1"
    set "MODE_FLAGS=--track --pose"
    REM Loitering ships at 20 seconds, which is a long time to stand still in
    REM front of your own webcam to see whether a feature works. Five makes it
    REM demonstrable; it is a demo value, not a recommendation, and the banner
    REM below says so rather than letting you infer the shipped default is 5.
    set "MODE_EXTRA=--set activity.loiter_s=5"
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
if /I "%MODE%"=="identity" (
    set "MODE_MODEL=yolox-tiny"
    set "MODE_INTERVAL=1"
    set "MODE_FLAGS=--track --identify"
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

if /I "%MODE%"=="checks" goto checks
if /I "%MODE%"=="enroll" goto enroll
if /I "%MODE%"=="who" goto who

REM Echo the real command rather than the defaults: appended arguments are
REM allowed to override, and a banner built from the variables above would
REM report the wrong model whenever they did.
set "VARGS=run --source %VANTAGE_SOURCE% %MODE_FLAGS% --model %VANTAGE_MODEL% --device %VANTAGE_DEVICE% --detect-interval %VANTAGE_INTERVAL% %MODE_EXTRA%%ARGS%"
echo [%MODE%] vantage %VARGS%
if /I "%MODE%"=="activity" (
    echo.
    echo   Watch the HUD "activity" line. To see each activity:
    echo     walking / running  - walk across the frame, then jog
    echo     loitering          - stand still for 5s ^(shipped default is 20s^)
    echo     arm_raised         - hold a hand above your shoulder
    echo     sitting_down       - sit, with your KNEES in frame
    echo   Posture reads "unknown" unless your legs are visible; that is the
    echo   correct answer, not a fault, and the HUD prints the reason.
)
if /I "%MODE%"=="identity" (
    echo.
    echo   Names appear over people the system recognises. With nobody
    echo   enrolled, everyone is "unknown" - which is the truth, not a fault.
    echo   Enrol first:  webcam.bat enroll --name alice --consent
    echo.
    echo   A name is shown only after several agreeing looks, so expect a
    echo   second or two of "identifying" before it settles.
)
echo.
echo Press q or Esc in the video window to stop, h for the HUD, s for a snapshot.
echo.

"%PYTHON%" -m vantage %VARGS%
set "RESULT=%ERRORLEVEL%"
goto finish

:enroll
REM Deliberately does NOT supply --consent. Everything else here is a
REM convenience - the source, the model paths - but the consent flag is the
REM one thing that makes this an enrolment rather than a capture, and a
REM launcher that passed it silently would turn a statement about a person
REM into a property of the shortcut you happened to double-click.
echo %ARGS% | findstr /I /C:"--consent" >nul
if errorlevel 1 (
    echo.
    echo   Enrolment needs --consent, and this script will not add it for you.
    echo.
    echo   The flag asserts that the person being enrolled knows about it and
    echo   agreed to it. That is a statement about a person, so it has to come
    echo   from you rather than from a shortcut.
    echo.
    echo       webcam.bat enroll --name alice --consent
    echo       webcam.bat enroll --name bob   --consent --samples 12
    echo.
    echo   Stored: a 128-number face template. No photograph is written to
    echo   disk. Remove one later with:
    echo       .venv\Scripts\python.exe -m vantage identity forget --name alice
    echo.
    set "RESULT=2"
    goto finish
)
echo [enroll] vantage identity enroll --source %VANTAGE_SOURCE%%ARGS%
echo.
echo   Look at the camera, straight on and well lit. Move your head a little
echo   between captures - eight copies of one pose generalise no better than
echo   one capture does.
echo.
"%PYTHON%" -m vantage identity enroll --source %VANTAGE_SOURCE%%ARGS%
set "RESULT=%ERRORLEVEL%"
goto finish

:who
"%PYTHON%" -m vantage identity list%ARGS%
set "RESULT=%ERRORLEVEL%"
goto finish

:checks
REM No camera, no weights, no inference runtime: both harnesses score their
REM subsystem against scripted ground truth, and both exit non-zero on failure.
REM
REM Arguments are refused rather than forwarded. The two harnesses have
REM different scenario names, so "checks --scenarios walk" would score the
REM activity rules and then fail on the tracker - a half-success that reads as
REM a bug in the tool. Per-harness flags belong on the harness.
if not "%ARGS%"=="" (
    echo.
    echo   'checks' takes no arguments, because it runs three harnesses whose
    echo   scenario names differ. Run one directly instead:
    echo.
    echo       .venv\Scripts\python.exe -m vantage activity eval%ARGS%
    echo       .venv\Scripts\python.exe -m vantage spatial eval%ARGS%
    echo       .venv\Scripts\python.exe -m vantage track eval%ARGS%
    echo.
    set "RESULT=2"
    goto finish
)
echo [checks] scoring the tracker, activity rules and spatial geometry
echo.
"%PYTHON%" -m vantage activity eval
set "RESULT=%ERRORLEVEL%"
echo.
"%PYTHON%" -m vantage spatial eval
if not "%ERRORLEVEL%"=="0" set "RESULT=%ERRORLEVEL%"
echo.
"%PYTHON%" -m vantage track eval
if not "%ERRORLEVEL%"=="0" set "RESULT=%ERRORLEVEL%"

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
