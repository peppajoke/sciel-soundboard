"""
Multi-device audio playback.

Every clip plays to *all* configured output devices at once -- typically the
VoiceMeeter input (which feeds both Streamlabs and Discord) plus Jack's
headphones, so he hears what the stream hears. They're separate streams
rather than one, because Windows won't let two apps share an endpoint at
different rates and monitoring must not depend on the routing being right.

DESIGN: each device gets ONE persistent OutputStream whose callback mixes
whatever voices are currently active. The alternatives are worse --
opening a stream per press adds 50-200 ms of device-open latency (fatal for
a soundboard), and one-voice-at-a-time means a second press cuts the first,
which is not how anyone expects a soundboard to behave.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

BLOCK = 512          # ~10 ms at 48k: low latency, still large enough to be safe
RATE = 48000
CHANNELS = 2


class _Voice:
    """One in-flight clip playback."""

    def __init__(self, data: np.ndarray, gain: float = 1.0):
        self.data = data
        self.pos = 0
        self.gain = gain

    def read(self, frames: int) -> np.ndarray:
        chunk = self.data[self.pos : self.pos + frames]
        self.pos += len(chunk)
        if len(chunk) < frames:
            pad = np.zeros((frames - len(chunk), CHANNELS), dtype=np.float32)
            chunk = np.vstack([chunk, pad])
        return chunk * self.gain

    @property
    def done(self) -> bool:
        return self.pos >= len(self.data)


class _Sink:
    """One output device, with its own independent voice list.

    Per-device voice lists are NOT redundant: a shared list would be drained
    once per device callback, so with headphones + VoiceMeeter open every clip
    would advance two positions per block and play at double speed. Each sink
    owning its own copy of a voice is what keeps them in sync.
    """

    def __init__(self, index: int, master_gain: float):
        self.master_gain = master_gain
        self.voices: list[_Voice] = []
        self.lock = threading.Lock()
        # latency="low" asks PortAudio for the device's minimum safe buffer.
        # Without it the default is "high" -- measured 192 ms on the MME
        # endpoint here, which is the single biggest source of click-to-sound
        # delay, dwarfing decode (3 ms) and HTTP (~1 ms) combined.
        self.stream = sd.OutputStream(
            device=index, samplerate=RATE, channels=CHANNELS,
            dtype="float32", blocksize=BLOCK, callback=self._callback,
            latency="low",
        )
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            if not self.voices:
                outdata.fill(0)
                return
            mix = np.zeros((frames, CHANNELS), dtype=np.float32)
            for voice in self.voices:
                mix += voice.read(frames)
            self.voices = [v for v in self.voices if not v.done]
        mix *= self.master_gain
        # Hard-clip rather than let float overflow wrap into noise. Stacked
        # clips can exceed unity; loudnorm at import keeps this rare.
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:] = mix

    def replace(self, data: np.ndarray, gain: float) -> None:
        with self.lock:
            self.voices = [_Voice(data, gain)]

    def clear(self) -> None:
        with self.lock:
            self.voices.clear()

    @property
    def busy(self) -> bool:
        with self.lock:
            return bool(self.voices)

    def close(self) -> None:
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


class Player:
    def __init__(self, device_names: list[str], master_gain: float = 1.0):
        self.master_gain = master_gain
        self._sinks: list[_Sink] = []
        self._cache: dict[Path, np.ndarray] = {}
        self.devices: list[str] = []
        self.errors: list[str] = []
        self._open(device_names)

    def _open(self, device_names: list[str]) -> None:
        for name in device_names:
            try:
                index = _resolve(name)
                self._sinks.append(_Sink(index, self.master_gain))
                self.devices.append(sd.query_devices(index)["name"])
            except Exception as exc:
                # A missing output device must not take the whole soundboard
                # down -- headphones get unplugged mid-stream. Record and go on.
                self.errors.append(f"{name}: {exc}")

    def _load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is None:
            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
            if data.shape[1] == 1:
                data = np.repeat(data, CHANNELS, axis=1)
            if rate != RATE:
                raise ValueError(f"{path.name} is {rate} Hz, expected {RATE}")
            cached = self._cache[path] = data
        return cached

    def play(self, path: Path, gain: float = 1.0,
             start: float = 0.0, end: float | None = None) -> None:
        """Play `path`, optionally only the [start, end) window of it.

        Slicing happens here rather than on disk so the stored file is never
        cut: trim bounds stay editable, and a mistake costs one drag rather
        than a re-upload. The slice is a numpy view, so it is free.
        """
        data = self._load(Path(path))
        if start or end is not None:
            a = max(0, int(start * RATE))
            b = len(data) if end is None else min(len(data), int(end * RATE))
            data = data[a:b]
            if len(data) == 0:
                raise ValueError("trim range is empty")
        if not self._sinks:
            raise RuntimeError("no output devices open: " + "; ".join(self.errors))
        for sink in self._sinks:
            # ONE SOUND AT A TIME, always. A new clip cuts whatever was
            # playing; there is deliberately no option to stack, because two
            # soundbites at once is noise rather than comedy. The swap happens
            # under the sink's lock so no audio block is ever mixed from both.
            sink.replace(data, gain)

    def preload(self, paths) -> int:
        """Decode clips into the sample cache ahead of time.

        Decoding on first press costs a few milliseconds -- small, but it lands
        exactly on the press you most care about, and it grows with clip
        length. The whole library is a few tens of MB decoded, so there is no
        reason not to hold it.
        """
        loaded = 0
        for path in paths:
            try:
                self._load(Path(path))
                loaded += 1
            except Exception:
                pass          # a broken clip must not stop the warm-up
        return loaded

    def stop_all(self) -> None:
        for sink in self._sinks:
            sink.clear()

    @property
    def playing(self) -> bool:
        return any(sink.busy for sink in self._sinks)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def _resolve(name: str) -> int:
    """Resolve an output device by name substring, preferring WASAPI.

    Matching by name not index: indices reshuffle whenever a USB headset is
    plugged in, so a pinned index silently starts pointing at the wrong device.

    Host API preference is a LATENCY decision, not cosmetics. The same endpoint
    is exposed under MME, DirectSound and WASAPI; MME comes first in the device
    list and carries ~190 ms of output latency here, while WASAPI is an order
    of magnitude lower. Taking the first name match meant every clip was
    silently a fifth of a second late.
    """
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        return int(name)
    needle = name.lower()
    hostapis = sd.query_hostapis()
    wasapi = next((i for i, h in enumerate(hostapis)
                   if "wasapi" in h["name"].lower()), None)

    matches = [(i, d) for i, d in enumerate(sd.query_devices())
               if d["max_output_channels"] > 0 and needle in d["name"].lower()]
    if not matches:
        raise ValueError(f"no output device matching {name!r}")
    for i, dev in matches:
        if dev["hostapi"] == wasapi:
            return i
    return matches[0][0]


def _devices(kind: str) -> list[dict]:
    """Devices of `kind`, preferring WASAPI.

    Windows exposes every endpoint under MME, DirectSound AND WASAPI, so an
    unfiltered list is three walls of near-duplicates. Worse, MME truncates
    names to 31 characters -- picking the MME entry stores a clipped name like
    "Voicemeeter Input (VB-Audio Voi", which still resolves by substring but
    reads like a bug. WASAPI gives full names and the lowest latency of the
    three, so that is what the UI offers.
    """
    channels = "max_output_channels" if kind == "output" else "max_input_channels"
    hostapis = sd.query_hostapis()
    wasapi = next((i for i, h in enumerate(hostapis)
                   if "wasapi" in h["name"].lower()), None)

    found = []
    seen = set()
    for i, dev in enumerate(sd.query_devices()):
        if dev[channels] <= 0:
            continue
        if wasapi is not None and dev["hostapi"] != wasapi:
            continue
        if dev["name"] in seen:
            continue
        seen.add(dev["name"])
        found.append({"index": i, "name": dev["name"],
                      "hostapi": hostapis[dev["hostapi"]]["name"]})
    return found


def list_outputs() -> list[dict]:
    return _devices("output")


def list_inputs() -> list[dict]:
    return _devices("input")
