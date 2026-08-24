# Rebuild the soundboard on a fresh machine after cloning.
#
# The repo carries the clips, library.json and config.json, so a clone is a
# working soundboard once this finishes -- it only rebuilds what cannot be
# committed: the 2.3 GB virtualenv.
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Write-Host "Creating virtualenv..."
python -m venv .venv

Write-Host "Installing dependencies..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip -q
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Checking ffmpeg..." -NoNewline
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host " found."
} else {
    Write-Host " MISSING."
    Write-Host "  Install it:  winget install Gyan.FFmpeg"
    Write-Host "  Imports transcode with ffmpeg, so nothing can be added without it."
}

Write-Host "Checking VoiceMeeter..." -NoNewline
$vm = Get-CimInstance Win32_SoundDevice -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "Voicemeeter" }
if ($vm) {
    Write-Host " found."
} else {
    Write-Host " MISSING."
    Write-Host "  Install it:  winget install VB-Audio.Voicemeeter.Banana"
    Write-Host "  Only needed to route clips into Discord; Streamlabs can take"
    Write-Host "  any output device directly."
}

Write-Host ""
Write-Host "Done. Start it with:  .\run.cmd    then open http://localhost:8770"
Write-Host "Check Settings -> Outputs first: device names differ between machines,"
Write-Host "and config.json carries this machine's choices."
