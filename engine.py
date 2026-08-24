"""
The engine: wires library + player + listener + matcher together and owns all
the runtime state the web UI reads.

Everything the UI can do goes through here, so manual clicks and auto-fires
take exactly the same path and can't drift apart in behaviour.
"""

from __future__ import annotations

import logging
import random as _random
import threading
import time
from collections import deque
from pathlib import Path

import config
import matcher
from library import Library
from player import Player

log = logging.getLogger("soundboard.engine")

# How long after a clip finishes the loopback listener stays gated. Without
# this the soundboard hears itself: a clip plays out of the speakers, the
# loopback capture transcribes it, and it matches its own trigger. The clip
# duration is not enough on its own because whisper works on a trailing
# window, so the tail of the clip is still inside the next chunk.
SELF_HEAR_GUARD_S = 2.5


class Engine:
    def __init__(self):
        self.cfg = config.load()
        self.library = Library()
        self.player = Player(self.cfg["outputs"], self.cfg.get("master_gain", 1.0))
        self.listener = None

        self._last_play_ended = 0.0
        self._last_fire_at = 0.0                 # global cooldown
        self._clip_last_fired: dict[str, float] = {}
        self._auto_fires: deque[float] = deque()   # timestamps, for the budget
        self._random_stop = threading.Event()
        self._random_thread = None
        self._last_random = None
        self._dirty = False
        self._lock = threading.Lock()

        # Ring buffer of everything that happened, newest last. The UI polls
        # this to show a live transcript, which is the only practical way to
        # tune thresholds -- you need to see the near-misses, not just the hits.
        self.events = deque(maxlen=300)

        self.warm_cache()
        threading.Thread(target=self._flush_loop, daemon=True).start()

        if self.player.errors:
            for err in self.player.errors:
                log.error("output device: %s", err)
                self._event("error", f"output device: {err}")

    def _flush_loop(self):
        """Persist play counts periodically rather than per press.

        Losing a few counts to a hard power-off is fine -- they only drive a
        sort order. Keeping the write out of the press path is not optional.
        """
        while True:
            time.sleep(15)
            if self._dirty:
                self._dirty = False
                try:
                    self.library.save()
                except Exception:
                    log.exception("failed to flush play counts")

    def warm_cache(self):
        """Decode the library into memory in the background."""
        def run():
            n = self.player.preload(c.path for c in self.library.clips.values())
            self._event("info", f"{n} clip(s) ready in memory")
        threading.Thread(target=run, daemon=True).start()

    # ---------- events ----------

    def _event(self, kind: str, text: str, **extra):
        self.events.append({"at": time.time(), "kind": kind, "text": text, **extra})

    # ---------- playback ----------

    def play(self, clip_id: str, why: str = "manual"):
        clip = self.library.clips.get(clip_id)
        if not clip:
            raise KeyError(clip_id)
        if not clip.path.exists():
            raise FileNotFoundError(clip.path)

        self.player.play(clip.path, gain=clip.gain,
                         start=clip.start, end=clip.end,
                         exclusive=not self.cfg.get("allow_overlap", False))
        with self._lock:
            now = time.time()
            self._last_fire_at = now
            self._clip_last_fired[clip_id] = now
            # Trimmed length, not file length: gating the listener for the full
            # file after a 2 s trim of a 30 s upload would deafen auto mode.
            self._last_play_ended = now + clip.play_duration
        # In memory only. library.save() writes the whole index, and doing
        # that on every press cost a disk write inside the press path.
        clip.plays += 1
        clip.last_played = now
        self._dirty = True

        self._event("play", clip.name, clip_id=clip_id, why=why)
        return clip

    def stop(self):
        self.player.stop_all()
        with self._lock:
            self._last_play_ended = time.time()
        self._event("stop", "stopped all")

    # ---------- auto mode ----------

    def gate(self) -> bool:
        """True while the listener should ignore audio (we're making noise)."""
        if self.player.playing:
            return True
        with self._lock:
            return time.time() < self._last_play_ended + SELF_HEAR_GUARD_S

    def on_line(self, line):
        """A transcript chunk arrived."""
        listen = self.cfg["listen"]
        words = matcher.normalize(line.text)

        if not words:
            return

        # Short lines are not discarded -- they are held to a higher bar. A
        # loose match on one word is noise; an exact one is a real hit.
        short = len(words) < int(listen.get("min_words", 2))
        floor = float(listen.get("short_line_threshold", 0.95)) if short else 0.0

        if not self.cfg.get("auto_enabled"):
            self._event("heard", line.text, source=line.source, fired=None,
                        note="auto off")
            return

        now = time.time()
        with self._lock:
            if now - self._last_fire_at < float(listen.get("global_cooldown_s", 2.0)):
                self._event("heard", line.text, source=line.source, fired=None,
                            note="global cooldown")
                return

        if not self._budget_allows(now):
            limit = int(listen.get("budget_count", 10))
            window = float(listen.get("budget_window_s", 300))
            with self._lock:
                wait = int(window - (now - self._auto_fires[0])) if self._auto_fires else 0
            self._event("heard", line.text, source=line.source, fired=None,
                        note=f"budget spent ({limit}/{int(window/60)}min, "
                             f"resets in {wait}s)")
            return

        candidates = self.library.auto_clips()
        hit = matcher.find(line.text, candidates,
                           float(listen.get("threshold", 0.82)), floor=floor)

        if not hit:
            # Record the best near-miss so thresholds can be tuned from the UI
            # rather than guessed at.
            best = 0.0
            best_name = None
            for clip in candidates:
                for phrase in clip.triggers:
                    s = matcher.score(line.text, phrase)
                    if s > best:
                        best, best_name = s, clip.name
            self._event("heard", line.text, source=line.source, fired=None,
                        best=round(best, 3), best_clip=best_name,
                        note=f"short line, needs {floor:.2f}" if short else None)
            return

        cooldown = float(listen.get("cooldown_s", 10.0))
        with self._lock:
            last = self._clip_last_fired.get(hit.clip_id, 0.0)
            if now - last < cooldown:
                self._event("heard", line.text, source=line.source, fired=None,
                            note=f"clip cooldown ({hit.score:.2f})")
                return

        clip = self.library.clips[hit.clip_id]
        self._event("heard", line.text, source=line.source, fired=clip.name,
                    best=round(hit.score, 3), phrase=hit.phrase)
        try:
            self.play(hit.clip_id, why=f"auto:{hit.phrase} ({hit.score:.2f})")
            with self._lock:
                self._auto_fires.append(time.time())
        except Exception as exc:
            log.exception("auto play failed")
            self._event("error", f"auto play failed: {exc}")

    # ---------- random dropper ----------

    def _budget_allows(self, now: float) -> bool:
        """True if the shared auto/random fire budget has room left."""
        listen = self.cfg["listen"]
        window = float(listen.get("budget_window_s", 300))
        limit = int(listen.get("budget_count", 10))
        if limit <= 0:
            return True
        with self._lock:
            while self._auto_fires and now - self._auto_fires[0] > window:
                self._auto_fires.popleft()
            return len(self._auto_fires) < limit

    def _random_loop(self):
        while not self._random_stop.is_set():
            cfg = self.cfg.get("random", {})
            lo = float(cfg.get("min_minutes", 4)) * 60
            hi = max(lo, float(cfg.get("max_minutes", 8)) * 60)
            wait = _random.uniform(lo, hi)
            if self._random_stop.wait(wait):
                return
            if not self.cfg.get("random", {}).get("enabled"):
                continue

            pool = self.library.random_clips()
            if not pool:
                self._event("info", "random: nothing whitelisted, skipping")
                continue

            now = time.time()
            if self.player.playing:
                self._event("info", "random: skipped, something is playing")
                continue
            if self.cfg["random"].get("use_budget", True) and not self._budget_allows(now):
                self._event("info", "random: skipped, budget spent")
                continue

            # Avoid repeating the last pick when there is a real choice --
            # random with replacement produces doubles often enough to look
            # broken to anyone watching.
            cooldown = float(self.cfg["listen"].get("cooldown_s", 180))
            with self._lock:
                rested = [c for c in pool
                          if now - self._clip_last_fired.get(c.id, 0.0) >= cooldown]
            if not rested:
                self._event("info", "random: every eligible clip is on cooldown")
                continue

            # Avoid repeating the last pick when there is a real choice --
            # random with replacement produces doubles often enough to look
            # broken to anyone watching.
            choices = [c for c in rested if c.id != self._last_random] or rested
            clip = _random.choice(choices)
            self._last_random = clip.id
            try:
                self.play(clip.id, why="random")
                if self.cfg["random"].get("use_budget", True):
                    with self._lock:
                        self._auto_fires.append(time.time())
            except Exception as exc:
                log.exception("random play failed")
                self._event("error", f"random play failed: {exc}")

    def start_random(self):
        if self._random_thread and self._random_thread.is_alive():
            return
        self._random_stop.clear()
        self._random_thread = threading.Thread(target=self._random_loop, daemon=True)
        self._random_thread.start()

    def stop_random(self):
        self._random_stop.set()

    def start_listener(self):
        if self.listener and self.listener.running:
            return
        from listener import Listener
        self.listener = Listener(self.cfg["listen"], self.on_line, self.gate)
        self._event("info", "starting listener (loading whisper model)")
        threading.Thread(target=self._start_listener_bg, daemon=True).start()

    def _start_listener_bg(self):
        try:
            self.listener.start()
            self._event("info", f"listener up: {self.listener.model_info}")
        except Exception as exc:
            log.exception("listener failed to start")
            self._event("error", f"listener failed: {exc}")

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self._event("info", "listener stopped")

    # ---------- config ----------

    def update_config(self, patch: dict):
        """Apply a config patch. Device/listen changes restart what they touch."""
        outputs_changed = "outputs" in patch and patch["outputs"] != self.cfg["outputs"]
        gain_changed = "master_gain" in patch
        listen_changed = "listen" in patch

        self.cfg = config._merge(self.cfg, patch)
        config.save(self.cfg)

        if outputs_changed or gain_changed:
            self.player.close()
            self.player = Player(self.cfg["outputs"], self.cfg.get("master_gain", 1.0))
            self.warm_cache()
            self._event("info", f"outputs: {', '.join(self.player.devices) or 'none'}")
            for err in self.player.errors:
                self._event("error", f"output device: {err}")

        if listen_changed and self.listener and self.listener.running:
            self.stop_listener()
            self.start_listener()

        return self.cfg

    def _budget_state(self):
        listen = self.cfg["listen"]
        window = float(listen.get("budget_window_s", 300))
        limit = int(listen.get("budget_count", 10))
        now = time.time()
        with self._lock:
            used = sum(1 for t in self._auto_fires if now - t <= window)
        return {"used": used, "limit": limit, "window_s": window}

    def status(self):
        return {
            "auto_enabled": bool(self.cfg.get("auto_enabled")),
            "outputs": self.player.devices,
            "output_errors": self.player.errors,
            "playing": self.player.playing,
            "clips": len(self.library.clips),
            "auto_clips": len(self.library.auto_clips()),
            "budget": self._budget_state(),
            "random": {
                "enabled": bool(self.cfg.get("random", {}).get("enabled")),
                "eligible": len(self.library.random_clips()),
                "running": bool(self._random_thread and self._random_thread.is_alive()),
            },
            "listener": self.listener.status() if self.listener else {"running": False},
        }
