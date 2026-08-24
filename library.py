"""
The clip library: metadata index + import pipeline.

Clips live in clips/ as normalized 48kHz stereo WAV. Everything is transcoded
on import rather than played as-uploaded, for two reasons:

  1. LOUDNESS. A soundboard sourced from random YouTube rips has clips 20 dB
     apart. Untreated, half are inaudible on stream and half blow out the mix.
     ffmpeg loudnorm puts them all at the same perceived level, so Jack sets
     the soundboard fader once and never touches it again.
  2. SAMPLE RATE. WASAPI shared mode refuses streams that don't match the
     device rate, so a 44.1k clip would simply fail to play on a 48k device.
     Normalizing at import means playback never has to resample.

The original upload is kept in clips/_originals/ so a bad transcode is
recoverable without re-uploading.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / "clips"
ORIGINALS = CLIPS / "_originals"
INDEX = ROOT / "library.json"

# Anything ffmpeg can decode; the extension list only gates obvious junk
# (a .txt dropped in the NAS inbox) so we don't spawn ffmpeg on every file.
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".flac",
              ".wma", ".webm", ".mp4", ".mkv", ".mov"}

TARGET_RATE = 48000
TARGET_CHANNELS = 2
# -16 LUFS is the streaming-loudness convention; sitting soundbites at the
# same target as the stream itself means they cut through without clipping.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "clip"


# Seeded once, on the first run that has no sources key at all. Deleting them
# all afterwards sticks -- the key exists by then, so nothing is re-added.
SEED_SOURCES = ["Wendy Williams", "Tim Robinson", "Trump", "DJ Khaled",
                "Arrested Development"]


@dataclass
class Source:
    id: str
    name: str
    added_at: float = 0.0
    # Filename inside static/source_images/, or None for the generated
    # initial avatar. Stored as a name rather than a path so the folder can
    # move with the repo.
    image: str | None = None


@dataclass
class Clip:
    id: str
    name: str
    file: str                      # relative to clips/
    triggers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    threshold: float | None = None  # None -> use the global default
    duration: float = 0.0           # full length of the stored file
    # Non-destructive trim. The file on disk is never cut -- these are honored
    # at playback, so the bounds stay adjustable forever and a bad trim costs
    # nothing. `end = None` means "play to the end of the file".
    start: float = 0.0
    end: float | None = None
    # Opt-in whitelist for the random-interval dropper. Off by default: firing
    # an arbitrary clip unprompted is only funny for the handful you picked.
    random_ok: bool = False
    source: str | None = None       # id of the Source this was clipped from
    # Per-clip level, linear multiplier applied at playback. Import already
    # loudness-normalises everything, so this is for taste -- a clip that
    # should sit under the game, or one that needs to punch through.
    gain: float = 1.0
    # Kept for SORTING only -- deliberately not shown on the pads. Written
    # back to disk by a debounced flush, never on the press itself, so a
    # counter can never sit in the click-to-sound path.
    plays: int = 0
    last_played: float = 0.0
    # Whether this clip has been cut down to the bit you actually want. Every
    # upload starts unfinished: a raw drop is nearly always longer than the
    # soundbite inside it, so "not yet clipped" is the honest default.
    finished: bool = False
    added_at: float = 0.0

    @property
    def path(self) -> Path:
        return CLIPS / self.file

    @property
    def play_duration(self) -> float:
        """Length actually heard, after trim. Drives cooldowns and the UI."""
        end = self.duration if self.end is None else min(self.end, self.duration)
        return max(0.0, end - self.start)


class Library:
    def __init__(self) -> None:
        self.clips: dict[str, Clip] = {}
        self.sources: dict[str, Source] = {}
        # Saves come from several threads at once: the engine's play-count
        # flush timer, the NAS watcher, and any Flask handler (the server runs
        # threaded). They all wrote the SAME .tmp path, so one thread could
        # rename a file another was still writing -- which is precisely the
        # half-written index that write-then-rename exists to prevent.
        self._save_lock = threading.Lock()
        self.load()

    # ---------- persistence ----------

    def load(self) -> None:
        # Set before parsing: seeding happens mid-load, but the write has to
        # wait until clips are in memory or save() would truncate them.
        self._seeded = False
        if INDEX.exists():
            raw = json.loads(INDEX.read_text(encoding="utf-8-sig"))
            if "sources" in raw:
                known_src = set(Source.__dataclass_fields__)
                self.sources = {
                    s["id"]: Source(**{k: v for k, v in s.items() if k in known_src})
                    for s in raw["sources"]
                }
            else:
                self.sources = {}
                for name in SEED_SOURCES:
                    self.add_source(name, save=False)
                self._seeded = True
            known = set(Clip.__dataclass_fields__)
            self.clips = {
                c["id"]: Clip(**{k: v for k, v in c.items() if k in known})
                for c in raw.get("clips", [])
            }
        else:
            self.clips = {}
            self.sources = {}
            for name in SEED_SOURCES:
                self.add_source(name, save=False)
            self._seeded = True

        if self._seeded:
            self.save()

    def save(self) -> None:
        # Write-then-rename: this box hard power-offs, and a half-written
        # index would lose the whole library rather than one clip.
        with self._save_lock:
            # Snapshot both dicts first. asdict() runs Python per clip, so
            # iterating the live dicts here raises "dictionary changed size
            # during iteration" the moment an upload or a NAS import lands
            # mid-save -- and that save is then lost.
            clips = list(self.clips.values())
            sources = list(self.sources.values())
            tmp = INDEX.with_suffix(".json.tmp")
            payload = {"clips": [asdict(c) for c in clips],
                       "sources": [asdict(s) for s in sources]}
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(INDEX)

    # ---------- import ----------

    def import_file(self, src: Path, name: str | None = None,
                    triggers: list[str] | None = None,
                    tags: list[str] | None = None) -> Clip:
        """Transcode `src` into the library and index it."""
        if src.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"unsupported file type: {src.suffix}")

        CLIPS.mkdir(parents=True, exist_ok=True)
        ORIGINALS.mkdir(parents=True, exist_ok=True)

        display = name or src.stem
        clip_id = f"{_slug(display)}-{uuid.uuid4().hex[:6]}"
        out_rel = f"{clip_id}.wav"
        out = CLIPS / out_rel

        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src),
             "-af", LOUDNORM,
             "-ar", str(TARGET_RATE), "-ac", str(TARGET_CHANNELS),
             "-c:a", "pcm_s16le", str(out)],
            check=True,
        )

        shutil.copy2(src, ORIGINALS / f"{clip_id}{src.suffix.lower()}")

        clip = Clip(
            id=clip_id,
            name=display,
            file=out_rel,
            triggers=triggers or [],
            tags=tags or [],
            duration=_duration(out),
            added_at=time.time(),
        )
        self.clips[clip.id] = clip
        self.save()
        return clip

    # ---------- queries / edits ----------

    def search(self, query: str, sort: str = "name") -> list[Clip]:
        """Substring match over name, tags and triggers. Empty query = all.

        Alphabetical by default -- with dozens of clips it is the only order
        you can navigate by muscle memory, since it does not move around.
        """
        q = query.strip().lower()
        items = list(self.clips.values())
        if q:
            items = [c for c in items
                     if q in c.name.lower()
                     or any(q in t.lower() for t in c.tags)
                     or any(q in t.lower() for t in c.triggers)]

        if sort == "plays":
            # Name as the tiebreak, so the long tail of never-played clips
            # stays alphabetical instead of arbitrary.
            items.sort(key=lambda c: (-c.plays, c.name.lower()))
        elif sort == "newest":
            items.sort(key=lambda c: -c.added_at)
        elif sort == "oldest":
            items.sort(key=lambda c: c.added_at)
        elif sort == "longest":
            # play_duration, not file duration: the trimmed length is what you
            # actually hear, and sorting by the raw file would rank a 2s bite
            # cut from a 40s upload as "long".
            items.sort(key=lambda c: (-c.play_duration, c.name.lower()))
        elif sort == "shortest":
            items.sort(key=lambda c: (c.play_duration, c.name.lower()))
        else:
            items.sort(key=lambda c: c.name.lower())
        return items

    # ---------- sources ----------

    def add_source(self, name: str, save: bool = True) -> Source:
        name = name.strip()
        if not name:
            raise ValueError("source needs a name")
        for existing in self.sources.values():
            if existing.name.lower() == name.lower():
                return existing          # idempotent: adding a dupe is a no-op
        src = Source(id=f"{_slug(name)}-{uuid.uuid4().hex[:4]}", name=name,
                     added_at=time.time())
        self.sources[src.id] = src
        if save:
            self.save()
        return src

    def rename_source(self, source_id: str, name: str) -> Source:
        src = self.sources[source_id]
        src.name = name.strip() or src.name
        self.save()
        return src

    def set_source_image(self, source_id: str, filename: str | None) -> Source:
        src = self.sources[source_id]
        src.image = filename
        self.save()
        return src

    def delete_source(self, source_id: str) -> int:
        """Remove a source; clips that used it become unassigned.

        Returns how many clips were orphaned. Clips are never deleted along
        with a source -- losing soundbites because a label was tidied up would
        be a nasty surprise.
        """
        self.sources.pop(source_id, None)
        orphaned = 0
        for clip in self.clips.values():
            if clip.source == source_id:
                clip.source = None
                orphaned += 1
        self.save()
        return orphaned

    def source_counts(self) -> dict:
        counts = {sid: 0 for sid in self.sources}
        counts[""] = 0                    # unassigned
        for clip in self.clips.values():
            key = clip.source if clip.source in self.sources else ""
            counts[key] = counts.get(key, 0) + 1
        return counts

    def counts(self) -> dict:
        done = sum(1 for c in self.clips.values() if c.finished)
        return {"done": done, "todo": len(self.clips) - done, "all": len(self.clips)}

    def random_clips(self) -> list[Clip]:
        """Clips whitelisted for random dropping."""
        return [c for c in self.clips.values() if c.random_ok]

    def auto_clips(self) -> list[Clip]:
        """Clips eligible for auto-fire: those with at least one trigger."""
        return [c for c in self.clips.values() if c.triggers]

    def duplicate(self, clip_id: str) -> Clip:
        """Copy a clip's entry, pointing at the SAME audio file.

        This is how one upload yields several soundbites: duplicate, then set
        different trim bounds on each. Nothing is re-encoded and no disk is
        used, so the copies also share the decoded sample cache -- a duplicate
        is free to play.
        """
        src = self.clips[clip_id]
        base = re.sub(r" \(\d+\)$", "", src.name)
        existing = {c.name for c in self.clips.values()}
        n = 2
        while f"{base} ({n})" in existing:
            n += 1

        clip = Clip(
            id=f"{_slug(base)}-{uuid.uuid4().hex[:6]}",
            name=f"{base} ({n})",
            file=src.file,               # shared, deliberately
            triggers=list(src.triggers),
            tags=list(src.tags),
            threshold=src.threshold,
            random_ok=src.random_ok,
            # Two cuts of one upload are from the same place and want the same
            # level. Dropping these left the copy unassigned, so it vanished
            # out of whichever source tab you were duplicating from, and at
            # full volume, so a clip deliberately turned down came back loud.
            source=src.source,
            gain=src.gain,
            # A duplicate exists to become a DIFFERENT cut, so it starts
            # unfinished no matter what the source was marked.
            finished=False,
            duration=src.duration,
            start=src.start,
            end=src.end,
            added_at=time.time(),
        )
        self.clips[clip.id] = clip
        self.save()
        return clip

    def update(self, clip_id: str, **fields) -> Clip:
        clip = self.clips[clip_id]
        for key, value in fields.items():
            if hasattr(clip, key):
                setattr(clip, key, value)
        self.save()
        return clip

    def delete(self, clip_id: str) -> None:
        clip = self.clips.pop(clip_id, None)
        if not clip:
            return
        # Duplicates share one file, so only remove it once nothing points at
        # it any more. Unlinking unconditionally would silently break every
        # other cut made from the same upload.
        still_used = any(c.file == clip.file for c in self.clips.values())
        if not still_used:
            clip.path.unlink(missing_ok=True)
        self.save()


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return round(float(out.stdout.strip()), 3)
    except Exception:
        return 0.0
