@echo off
REM Launch the soundboard and open the UI.
REM
REM The PATH lines are load-bearing: faster-whisper dies with
REM "Library cublas64_12.dll is not found" unless the bundled NVIDIA DLL
REM directories are on PATH *before the process starts*. Calling
REM os.add_dll_directory() from inside Python is NOT sufficient, which is why
REM this is a .cmd wrapper and not a few lines at the top of server.py.
setlocal
set HERE=%~dp0
set SP=%HERE%.venv\Lib\site-packages\nvidia
set PATH=%SP%\cublas\bin;%SP%\cudnn\bin;%PATH%

if not exist "%HERE%.venv\Scripts\python.exe" (
  echo No virtualenv found. Run install.ps1 first:
  echo     powershell -ExecutionPolicy Bypass -File install.ps1
  pause
  exit /b 1
)

REM Open the browser once the server has had a moment to bind. Backgrounded so
REM it does not block startup, and pointed at localhost rather than the file --
REM opening index.html directly gives a page that cannot reach the API.
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8770"

"%HERE%.venv\Scripts\python.exe" "%HERE%server.py" %*
