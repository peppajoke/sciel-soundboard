"""
Transcript listener: audio -> faster-whisper -> text lines.

Captures from either the microphone (what Jack says), a WASAPI loopback of the
system output (what the game says), or both at once as independent streams.
Each finished chunk is handed to a callback as plain text; matching decisions
live in matcher.py, not here.

WHY `soundcard` AND NOT `sounddevice`: sounddevice 0.5.6 exposes no loopback
flag on WasapiSettings, so it cannot capture system output at all. soundcard's
`include_loopback=True` can. Playback stays on sounddevice, which has the
lower-latency callback model that a soundboard needs.

THE SILENT-INPUT TRAP: NVIDIA Broadcast emits true digital silence between
words, and a wrongly-resolved device emits digital silence forever. Those look
identical to a naive level check, and a deaf listener looks exactly like a
healthy one -- no error, no frames, just nothing ever firing. So we track
"seconds since we last saw ANY signal" and warn loudly past a threshold rather
than failing silently.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import soundcard as sc

# soundcard talks to WASAPI through COM, and COM is initialised PER THREAD.
# Every capture runs on its own thread, so without this each one fails with
# 0x800401F0 (CO_E_NOTINITIALIZED) the moment it touches a device. It only
# happened to work when a single capture inherited the main thread's state.
if sys.platform == "win32":
    import ctypes
    _ole32 = ctypes.windll.ole32
    COINIT_MULTITHREADED = 0x0

    def _com_init():
        # S_FALSE (1) means "already initialised on this thread", which is fine.
        hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        if hr not in (0, 1):
            log.warning("CoInitializeEx returned 0x%08X", hr & 0xFFFFFFFF)

    def _com_uninit():
        _ole32.CoUninitialize()
else:
    def _com_init():
        pass

    def _com_uninit():
        pass

log = logging.getLogger("soundboard.listener")

RATE = 16000          # whisper's native rate; the recorder resamples for us
SILENCE_RMS = 1e-5    # below this is indistinguishable from digital silence
SILENT_WARN_S = 45.0
# Falling this far behind realtime means the backlog is discarded rather than
# transcribed late.
MAX_LAG_S = 1.0


@dataclass
class Line:
    text: str
    source: str       # "voice" or "game"
    at: float


class _Capture(threading.Thread):
    """One capture stream -> transcript lines on `emit`."""

    daemon = True

    def __init__(self, source, device_name, loopback, model, chunk_s, emit,
                 gate=None, hop_s=0.75, prompt_fn=None, beam_size=5):
        super().__init__(name=f"listen-{source}")
        self.source = source
        self.device_name = device_name
        self.loopback = loopback
        self.model = model
        self.chunk_s = chunk_s
        self.hop_s = min(hop_s, chunk_s)
        self.emit = emit
        # gate() -> True means "drop this chunk". Stops the loopback listener
        # transcribing the soundboard's own output and re-triggering itself,
        # which otherwise loops until the cooldown saves you.
        self.gate = gate
        # Returns the trigger phrases as a decoding prompt. Biasing whisper
        # toward the words we actually care about measured +0.04 on a 12-clip
        # eval -- the same gain as jumping to medium.en, at no latency cost.
        self.prompt_fn = prompt_fn
        self.beam_size = beam_size
        self.stop_flag = threading.Event()
        self.last_signal = time.time()
        self.healthy = True
        self.error = None
        self.peak_db = -120.0      # level of the last chunk, for the UI meter
        self.lag = 0.0             # seconds behind realtime, if we fall behind
        self.dropped = 0           # hops discarded to catch up
        self.chunks = 0            # chunks with audio in them
        self.transcribed = 0       # chunks that produced text
        self.device_label = None
        self._last_quiet_note = 0.0

    def _mic(self):
        if self.device_name:
            # Match ONLY within the kind we asked for. Loopback entries are
            # named after the output they shadow ("Speakers (NVIDIA
            # Broadcast)"), so an unfiltered substring search for a mic can
            # silently return a speaker's loopback -- which then sits at
            # digital silence forever and looks like a broken microphone.
            for mic in sc.all_microphones(include_loopback=True):
                if bool(mic.isloopback) != self.loopback:
                    continue
                if self.device_name.lower() in mic.name.lower():
                    return mic
            kind = "loopback output" if self.loopback else "microphone"
            raise ValueError(f"no {kind} matching {self.device_name!r}")
        if self.loopback:
            # Loopback of whatever the system is actually playing out of.
            return sc.get_microphone(sc.default_speaker().name, include_loopback=True)
        return sc.default_microphone()

    def run(self):
        _com_init()
        try:
            self._run()
        finally:
            _com_uninit()

    def _run(self):
        try:
            mic = self._mic()
        except Exception as exc:
            self.error = str(exc)
            self.healthy = False
            log.error("[%s] cannot open input: %s", self.source, exc)
            return

        self.device_label = mic.name
        log.info("[%s] listening on %s (window %.1fs, hop %.2fs)",
                 self.source, mic.name, self.chunk_s, self.hop_s)
        hop_frames = int(RATE * self.hop_s)
        window_frames = int(RATE * self.chunk_s)
        started = time.time()
        frames_read = 0
        # Rolling buffer: record a short hop, then transcribe the trailing
        # window. Detection latency becomes the hop rather than the window,
        # and phrases still get full context instead of being cut in half.
        buffer = np.zeros(0, dtype=np.float32)
        try:
            with mic.recorder(samplerate=RATE, channels=1) as rec:
                while not self.stop_flag.is_set():
                    data = rec.record(numframes=hop_frames)
                    chunk = np.asarray(data, dtype=np.float32).flatten()
                    frames_read += len(chunk)

                    # How far behind realtime are we? If a scan takes longer
                    # than a hop the recorder queues audio, and every later
                    # transcript describes speech from further in the past --
                    # which is how a soundbite lands absurdly late.
                    now = time.time()
                    self.lag = (now - started) - (frames_read / RATE)
                    if self.lag > MAX_LAG_S:
                        # Throw the backlog away and resynchronise. A late
                        # soundbite is worse than a missed one.
                        skipped = 0
                        while self.lag > 0.3 and skipped < 200:
                            data = rec.record(numframes=hop_frames)
                            frames_read += len(np.asarray(data).flatten())
                            self.lag = (time.time() - started) - (frames_read / RATE)
                            skipped += 1
                        self.dropped += skipped
                        buffer = np.zeros(0, dtype=np.float32)
                        log.warning("[%s] fell %.1fs behind; dropped %d hop(s)",
                                    self.source, self.lag + skipped * self.hop_s, skipped)
                        continue

                    # Stamp with when this audio was CAPTURED, so downstream can
                    # refuse to fire on anything stale.
                    captured_at = now
                    buffer = np.concatenate([buffer, chunk])[-window_frames:]
                    audio = buffer

                    rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
                    peak = float(np.abs(chunk).max()) if chunk.size else 0.0
                    self.peak_db = (20 * np.log10(peak)) if peak > 1e-6 else -120.0
                    now = time.time()
                    if rms > SILENCE_RMS:
                        self.last_signal = now
                        self.healthy = True
                    elif now - self.last_signal > SILENT_WARN_S:
                        # Loud, repeated, and surfaced in /api/status. A
                        # silently deaf listener is the failure mode that
                        # wastes a whole stream.
                        self.healthy = False
                        log.warning(
                            "[%s] NO AUDIO for %.0fs on %r -- wrong device, "
                            "muted input, or nothing playing.",
                            self.source, now - self.last_signal, mic.name)
                        self.last_signal = now

                    if rms <= SILENCE_RMS:
                        continue
                    if self.gate and self.gate():
                        buffer = np.zeros(0, dtype=np.float32)
                        continue

                    try:
                        segments, _ = self.model.transcribe(
                            audio, language="en", vad_filter=True,
                            beam_size=self.beam_size,
                            initial_prompt=self.prompt_fn() if self.prompt_fn else None,
                        )
                        text = " ".join(s.text for s in segments).strip()
                    except Exception as exc:
                        log.exception("[%s] transcribe failed: %s", self.source, exc)
                        continue

                    self.chunks += 1
                    if text:
                        self.transcribed += 1
                        self.emit(Line(text=text, source=self.source, at=captured_at))
                    # Audio with no words is NOT emitted. It used to send a
                    # "(sound, no speech - peak N dB)" note, which then got
                    # stitched into the word stream and matched against --
                    # diagnostics have no business in the transcript. The
                    # level meters carry that information instead.
        except Exception as exc:
            self.error = str(exc)
            self.healthy = False
            log.exception("[%s] capture died: %s", self.source, exc)


def _label(device, loopback):
    base = device or ("system audio" if loopback else "default mic")
    # Trim the "(VB-Audio Voicemeeter VAIO)" style suffix; the feed is narrow
    # and the leading words are what identify a device.
    short = base.split("(")[0].strip() or base
    return short + (" [game]" if loopback else "")


class Listener:
    """Owns the whisper model and one or two capture threads."""

    def __init__(self, cfg, emit, gate=None, prompt_fn=None):
        self.cfg = cfg
        self.emit = emit
        self.gate = gate
        self.prompt_fn = prompt_fn
        self.model = None
        self.captures = []
        self.model_info = ""

    def _load_model(self):
        from faster_whisper import WhisperModel
        want_device = self.cfg.get("device", "cuda")
        compute = self.cfg.get("compute_type", "float16")
        name = self.cfg.get("model", "small.en")
        try:
            model = WhisperModel(name, device=want_device, compute_type=compute)
            self.model_info = f"{name} on {want_device}/{compute}"
        except Exception as exc:
            # CUDA failure here usually means the cublas/cudnn DLLs are not on
            # PATH (see run.cmd). CPU still works, just slower -- better a slow
            # soundboard than a dead one mid-stream.
            log.warning("CUDA model load failed (%s); falling back to CPU", exc)
            model = WhisperModel(name, device="cpu", compute_type="int8")
            self.model_info = f"{name} on cpu/int8 (CUDA unavailable: {exc})"
        return model

    def start(self):
        if self.captures:
            return
        self.model = self._load_model()
        log.info("whisper: %s", self.model_info)

        chunk = float(self.cfg.get("chunk_s", 3.0))
        for entry in self.cfg.get("inputs") or []:
            device = entry.get("device")
            loopback = bool(entry.get("loopback"))
            # Label is what shows in the transcript feed, so make it the thing
            # you would recognise: the device name, not "input 3".
            label = entry.get("label") or _label(device, loopback)
            cap = _Capture(label, device, loopback, self.model, chunk,
                           self.emit, self.gate,
                           hop_s=float(self.cfg.get("hop_s", 0.75)),
                           prompt_fn=self.prompt_fn,
                           beam_size=int(self.cfg.get("beam_size", 5)))
            cap.start()
            self.captures.append(cap)

    def stop(self):
        for cap in self.captures:
            cap.stop_flag.set()
        self.captures = []

    @property
    def running(self):
        return any(c.is_alive() for c in self.captures)

    def status(self):
        return {
            "running": self.running,
            "model": self.model_info,
            "captures": [
                {"source": c.source, "healthy": c.healthy, "error": c.error,
                 "silent_for": round(time.time() - c.last_signal, 1),
                 "lag": round(c.lag, 2), "dropped": c.dropped,
                 "peak_db": round(c.peak_db, 1), "chunks": c.chunks,
                 "transcribed": c.transcribed, "device": c.device_label}
                for c in self.captures
            ],
        }
