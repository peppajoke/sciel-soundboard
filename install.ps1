# Soundboard installer (Windows).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Installs missing prerequisites with winget, builds the virtualenv, and puts
# a shortcut on the Desktop and in the Start Menu. Safe to re-run: every step
# checks before acting, so this doubles as a repair/upgrade command.
#
# Flags:
#   -WithVoicemeeter   also install VoiceMeeter Banana (needed only to route
#                      clips into Discord; Streamlabs can take any device)
#   -NoShortcuts       skip Desktop/Start Menu shortcuts
#   -Quiet             no prompts, assume yes

[CmdletBinding()]
param(
    [switch]$WithVoicemeeter,
    [switch]$NoShortcuts,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Say($msg, $colour = "White") { Write-Host $msg -ForegroundColor $colour }
function Ok($msg)   { Say "  [ok]   $msg" "Green" }
function Warn($msg) { Say "  [warn] $msg" "Yellow" }
function Fail($msg) { Say "  [FAIL] $msg" "Red" }

# winget writes to the machine PATH, but this session was started with the old
# one. Re-reading it means a tool installed a moment ago is usable immediately
# instead of needing a new terminal.
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Install-With-Winget($id, $label) {
    if (-not (Have "winget")) {
        Fail "$label is missing and winget is not available to install it."
        Say  "       Install $label manually, then re-run this script."
        return $false
    }
    Say "  installing $label via winget..."
    winget install --id $id --accept-package-agreements --accept-source-agreements `
                   --silent --disable-interactivity | Out-Null
    Refresh-Path
    return $true
}

Say ""
Say "Soundboard installer" "Cyan"
Say "====================" "Cyan"
Say ""

# ---------------------------------------------------------------- Python
Say "Python"
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    if (Have $candidate) {
        try {
            $v = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            # 3.9 and older lack the typing syntax used throughout this project.
            if ($v -and [version]$v -ge [version]"3.10") { $python = $candidate; break }
        } catch { }
    }
}
if (-not $python) {
    Warn "no Python 3.10+ found"
    if (Install-With-Winget "Python.Python.3.13" "Python 3.13") {
        foreach ($candidate in @("python", "python3", "py")) {
            if (Have $candidate) { $python = $candidate; break }
        }
    }
    if (-not $python) {
        Fail "Python still not on PATH. Open a new terminal and re-run this script."
        exit 1
    }
}
$pyver = & $python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
Ok "Python $pyver ($python)"

# ---------------------------------------------------------------- ffmpeg
Say ""
Say "ffmpeg  (transcodes and loudness-normalises every import)"
if (Have "ffmpeg") {
    Ok "ffmpeg found"
} else {
    Warn "ffmpeg missing"
    Install-With-Winget "Gyan.FFmpeg" "ffmpeg" | Out-Null
    if (Have "ffmpeg") { Ok "ffmpeg installed" }
    else { Warn "ffmpeg still not on PATH -- adding clips will fail until it is" }
}

# ---------------------------------------------------------------- VoiceMeeter
Say ""
Say "VoiceMeeter  (optional: routes clips into Discord)"
$vm = Get-CimInstance Win32_SoundDevice -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "Voicemeeter" }
if ($vm) {
    Ok "VoiceMeeter present"
} elseif ($WithVoicemeeter) {
    Install-With-Winget "VB-Audio.Voicemeeter.Banana" "VoiceMeeter Banana" | Out-Null
    Warn "VoiceMeeter makes itself the default playback device; check Windows sound settings"
} else {
    Warn "not installed - skipping (re-run with -WithVoicemeeter to add it)"
    Say  "       Streamlabs works without it; only Discord routing needs it."
}

# ---------------------------------------------------------------- venv
Say ""
Say "Python environment"
if (Test-Path ".venv\Scripts\python.exe") {
    Ok "virtualenv already exists"
} else {
    Say "  creating .venv (this pulls ~2 GB of CUDA libraries, give it a few minutes)..."
    & $python -m venv .venv
    Ok "virtualenv created"
}
$venvPy = Join-Path $here ".venv\Scripts\python.exe"

Say "  installing dependencies..."
& $venvPy -m pip install --upgrade pip -q
& $venvPy -m pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) { Fail "dependency install failed"; exit 1 }
Ok "dependencies installed"

# ---------------------------------------------------------------- self-test
Say ""
Say "Checking the install"
$check = & $venvPy -c @"
import sys
try:
    import flask, numpy, sounddevice, soundfile, soundcard, faster_whisper
    outs = [d for d in sounddevice.query_devices() if d['max_output_channels'] > 0]
    print('OK|%d' % len(outs))
except Exception as exc:
    print('ERR|%s' % exc)
"@
$parts = $check -split '\|', 2
if ($parts[0] -eq "OK") {
    Ok "all modules import; $($parts[1]) output device(s) visible"
} else {
    Fail "import check failed: $($parts[1])"
    exit 1
}

# GPU transcription is optional -- CPU works, just slower -- so this reports
# rather than fails.
$cuda = & $venvPy -c @"
import os, pathlib
sp = pathlib.Path(r'$here') / '.venv' / 'Lib' / 'site-packages' / 'nvidia'
print('yes' if (sp / 'cublas' / 'bin').exists() and (sp / 'cudnn' / 'bin').exists() else 'no')
"@
if ($cuda -eq "yes") { Ok "CUDA libraries present (run.cmd puts them on PATH)" }
else { Warn "no CUDA libraries - auto mode will transcribe on CPU" }

# ---------------------------------------------------------------- shortcuts
if (-not $NoShortcuts) {
    Say ""
    Say "Shortcuts"
    $target = Join-Path $here "run.cmd"
    $icon = "$env:SystemRoot\System32\SHELL32.dll,138"   # speaker icon
    $shell = New-Object -ComObject WScript.Shell
    foreach ($dir in @([Environment]::GetFolderPath("Desktop"),
                       (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"))) {
        try {
            $lnk = $shell.CreateShortcut((Join-Path $dir "Soundboard.lnk"))
            $lnk.TargetPath = $target
            $lnk.WorkingDirectory = $here
            $lnk.IconLocation = $icon
            $lnk.Description = "Stream soundboard"
            $lnk.Save()
            Ok "shortcut in $(Split-Path $dir -Leaf)"
        } catch {
            Warn "could not create shortcut in $dir : $_"
        }
    }
}

Say ""
Say "Done." "Cyan"
Say "  Start it:  .\run.cmd   (or the Soundboard shortcut)"
Say "  Then open: http://localhost:8770"
Say ""
Say "  First thing to check: Settings -> Outputs. Audio device names differ"
Say "  between machines, and the committed config.json holds the last one's."
Say ""
