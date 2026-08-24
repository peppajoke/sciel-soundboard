@echo off
REM Launch the soundboard.
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
"%HERE%.venv\Scripts\python.exe" "%HERE%server.py" %*
