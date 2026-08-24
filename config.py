"""Config load/save with defaults. Single JSON file, hot-editable from the UI."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    # Where clips play. Name substrings, matched against output devices.
    # "Voicemeeter Input" is the VAIO virtual input -- Streamlabs takes it as
    # an audio source and Discord takes Voicemeeter's B1 output as its mic, so
    # one entry covers both. Add your headphones here to monitor yourself.
    "outputs": ["Voicemeeter Input (VB-Audio Voicemeeter VAIO)"],
    "master_gain": 1.0,
    # False (default): a new clip cuts the one still playing. Set true to let
    # clips stack.
    "allow_overlap": False,

    "listen": {
        # Capture list, mirroring `outputs`: every entry gets its own stream and
        # is transcribed independently, so you can scan a mic and the game at
        # once, or several mics. Each entry is
        #   {"device": <name substring or null for the default>,
        #    "loopback": true to capture what a device PLAYS rather than what it
        #                hears -- this is how game audio is scanned}
        "inputs": [{"device": None, "loopback": True}],
        "model": "small.en",     # faster-whisper model
        "compute_type": "float16",
        "device": "cuda",        # falls back to CPU automatically if CUDA is absent
        "chunk_s": 3.0,          # audio window handed to whisper
        "threshold": 0.82,       # default fuzzy-match score to fire
        # Per-clip cooldown, applied to EVERY automatic path (trigger matches
        # and random drops alike). A soundbite lands once and then rests --
        # repetition is what makes a bit stale fastest. Manual presses are
        # exempt: when Jack presses a pad he means it.
        "cooldown_s": 180.0,
        "global_cooldown_s": 2.0,  # any clip: stops a pile-up
        # Transcripts shorter than this are held to `short_line_threshold`
        # instead of being thrown away. A one-word chunk is usually noise, but
        # "Kiss" against a "kiss" trigger is an exact hit and must still fire.
        "min_words": 2,
        "short_line_threshold": 0.95,

        # Rate budget: at most `budget_count` auto-fires per `budget_window_s`.
        # Without this, a chatty cutscene can dump a dozen soundbites in a
        # minute and the joke dies. Manual presses are NOT counted -- the
        # budget restrains the robot, not Jack.
        "budget_count": 10,
        "budget_window_s": 300,
    },

    # Auto mode starts OFF every launch -- deliberately. It should be an
    # explicit act each stream, not a thing that surprises you on boot.
    "auto_enabled": False,

    # Random dropper: fires a whitelisted clip every so often, unprompted.
    "random": {
        "enabled": False,
        # A range, not a fixed number: a clip landing exactly every 5:00 reads
        # as a machine, and the audience starts predicting it.
        "min_minutes": 4.0,
        "max_minutes": 8.0,
        # Random drops draw from the same budget as auto-fires, so the two
        # features cannot conspire to make the stream noisy.
        "use_budget": True,
    },

    # Forward slashes deliberately: Windows and pathlib both accept them for
    # UNC paths, and they survive shell quoting, JSON escaping and heredocs
    # without a doubling step. A single dropped backslash here silently turns
    # the share into a nonexistent relative path and the watcher just never
    # imports anything -- which is exactly what happened once already.
    "nas_inbox": "//192.168.1.169/media/soundboard/inbox",
    "nas_poll_s": 20,
    "port": 8770,
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            out[key] = _merge(base[key], value)
        else:
            out[key] = value
    return out


def _migrate(cfg: dict) -> dict:
    """Convert the old single-source listen config into the inputs list."""
    listen = cfg.get("listen", {})
    if "inputs" in listen and listen["inputs"] is not None:
        return cfg
    source = listen.pop("source", "game")
    inputs = []
    if source in ("voice", "both"):
        inputs.append({"device": listen.get("voice_device"), "loopback": False})
    if source in ("game", "both"):
        inputs.append({"device": listen.get("game_device"), "loopback": True})
    listen.pop("voice_device", None)
    listen.pop("game_device", None)
    listen["inputs"] = inputs or [{"device": None, "loopback": True}]
    cfg["listen"] = listen
    return cfg


def load() -> dict:
    if CONFIG_PATH.exists():
        stored = _migrate(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        merged = _merge(DEFAULTS, stored)
        # _merge would union the default inputs with the stored ones for a
        # dict, but inputs is a LIST and must be taken wholesale -- otherwise
        # unchecking every input silently falls back to the default.
        merged["listen"]["inputs"] = stored["listen"]["inputs"]
        return merged
    save(DEFAULTS)
    return dict(DEFAULTS)


def save(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)
