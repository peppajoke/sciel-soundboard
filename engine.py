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

# The single key under which all captures' words merge. See on_line().
STREAM_KEY = "mix"

# What every cooldown collapses to when "disable cooldowns" is ticked. Low
# enough to feel off, high enough that overlapping scan windows cannot fire
# the same clip repeatedly off a single phrase.
BYPASS_COOLDOWN_S = 5.0




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
        # Running transcript per capture source: (words, last_seen).
        self._stream: dict[str, tuple[list, float]] = {}
        # Clips already fired from the CURRENT utterance, per source. The words
        # that matched deliberately stay in the stream (see on_line), so the
        # per-clip cooldown is the only thing between one phrase and a stutter
        # -- and that cooldown is 5 s once "disable cooldowns" is ticked, while
        # the words linger for stream_gap_s (10 s). Reset when the stream
        # expires, i.e. when the utterance ends.
        self._fired_this_stream: dict[str, set[str]] = {}
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
                         start=clip.start, end=clip.end)
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

        # Stitch into the running transcript for this source. The listener
        # re-scans an overlapping window, so matching each emission on its own
        # missed any phrase that straddled a window boundary -- "fuck fuck
        # fuck" arriving as "fuck fuck" then "fuck fuck fuck" never matched.
        now = time.time()
        gap = float(listen.get("stream_gap_s", 10.0))
        keep = int(listen.get("stream_words", 20))

        # ONE stream regardless of how many inputs are configured. Keying by
        # line.source meant two mics hearing the same speech built two streams,
        # each displayed AND matched separately -- the same phrase fired the
        # same clip twice and the transcript showed it twice. The stitcher's
        # fuzzy overlap absorbs the second mic's near-identical re-decode.
        prev_words, last_seen = self._stream.get(STREAM_KEY, ([], 0.0))
        if now - last_seen > gap:
            prev_words = []
            # New utterance: whatever fired off the old one may fire again.
            self._fired_this_stream.pop(STREAM_KEY, None)
        stitched = matcher.stitch(prev_words, words, max_words=keep)
        self._stream[STREAM_KEY] = (stitched, now)

        # Matching runs on a vocabulary-snapped view: words whisper mangled
        # by a letter or two are pulled onto the trigger bank ("batter" ->
        # "battle"), so a one-word mishear cannot sink a whole phrase. The
        # DISPLAY stream stays raw -- what you said, as heard -- while events
        # record the snapped text, which is what the matcher actually saw.
        candidates = self.library.auto_clips()
        vocab = {w for c in candidates for t in c.triggers
                 for w in matcher.normalize(t)}
        text = " ".join(matcher.snap(stitched, vocab))

        # Un-mark fired clips whose trigger has scrolled OUT of the window.
        # The mark used to clear only on a 10s silence gap -- but on a stream
        # you never stop talking for 10s, so a clip that fired once was locked
        # for the rest of the monologue and read as "cooldown" with cooldowns
        # disabled. Once no trigger of a marked clip matches the current
        # window, the words that fired it are gone: saying the phrase AGAIN is
        # a new event and may fire. While the phrase is still visible in the
        # window the mark holds, which is the machine-gun protection.
        fired = self._fired_this_stream.get(STREAM_KEY)
        if fired:
            thr = float(listen.get("threshold", 0.82))
            for clip_id in list(fired):
                clip = self.library.clips.get(clip_id)
                if clip is None or not any(
                        matcher.score(text, t) >= min(thr,
                            clip.threshold or thr)
                        for t in clip.triggers):
                    fired.discard(clip_id)

        # The floor applies to the SEQUENCE, not one emission: a phrase that
        # has only produced one word so far is still genuinely short.
        short = len(stitched) < int(listen.get("min_words", 2))
        floor = float(listen.get("short_line_threshold", 0.95)) if short else 0.0

        if not self.cfg.get("auto_enabled"):
            self._event("heard", text, source=line.source, fired=None,
                        note="auto off")
            return

        with self._lock:
            if now - self._last_fire_at < self._cooldown("global_cooldown_s", 2.0):
                self._event("heard", text, source=line.source, fired=None,
                            note="global cooldown")
                return

        if not self._budget_allows(now):
            limit = int(listen.get("budget_count", 10))
            window = float(listen.get("budget_window_s", 300))
            with self._lock:
                wait = int(window - (now - self._auto_fires[0])) if self._auto_fires else 0
            self._event("heard", text, source=line.source, fired=None,
                        note=f"budget spent ({limit}/{int(window/60)}min, "
                             f"resets in {wait}s)")
            return

        # Freshness deadline, checked as late as possible so it accounts for
        # time spent in this function too.
        max_age = float(listen.get("max_fire_age_s", 2.0))
        age = time.time() - getattr(line, "at", now)
        if age > max_age:
            self._event("heard", text, source=line.source, fired=None,
                        note=f"too late to fire ({age:.1f}s old)")
            return

        hits = matcher.find_all(text, candidates,
                                float(listen.get("threshold", 0.82)), floor=floor)

        if not hits:
            # Record the best near-miss so thresholds can be tuned from the UI
            # rather than guessed at.
            best = 0.0
            best_name = None
            for clip in candidates:
                for phrase in clip.triggers:
                    s = matcher.score(text, phrase)
                    if s > best:
                        best, best_name = s, clip.name
            self._event("heard", text, source=line.source, fired=None,
                        best=round(best, 3), best_clip=best_name,
                        note=f"short line, needs {floor:.2f}" if short else None)
            return

        # Walk EVERY clearing match, best first, and fire the first one that
        # is not individually blocked. Considering only the single best match
        # let a stale phrase shadow a new one: "motherfucker" (already fired,
        # still scoring 1.0 while it sat in the window) was returned as the
        # best match and blocked, so "prepare for battle" -- spoken verbatim,
        # visible in the stream, clearly matching -- was never even evaluated.
        cooldown = self._cooldown("cooldown_s", 180.0)
        fired_set = self._fired_this_stream.get(STREAM_KEY, set())
        held = None
        hit = None
        for cand in hits:
            if cand.clip_id in fired_set:
                held = held or f"already fired this line ({cand.score:.2f})"
                continue
            with self._lock:
                last = self._clip_last_fired.get(cand.clip_id, 0.0)
            if now - last < cooldown:
                held = held or f"clip cooldown ({cand.score:.2f})"
                continue
            hit = cand
            break

        if hit is None:
            self._event("heard", text, source=line.source, fired=None,
                        note=held or "held")
            return

        age = time.time() - getattr(line, "at", now)
        if age > max_age:
            self._event("heard", text, source=line.source, fired=None,
                        note=f"too late to fire ({age:.1f}s old)")
            return

        clip = self.library.clips[hit.clip_id]
        # The sequence is deliberately NOT cleared here. Clearing it wiped the
        # live transcript every time a clip fired, so the display looked dead
        # exactly when things were working. Re-firing is already prevented by
        # the per-clip cooldown, whose floor (5 s even when "disabled") is
        # longer than the 3 s scan window that could re-deliver the phrase.
        # That last part only held before stitching: the STREAM re-delivers the
        # phrase for far longer than the scan window does, which is what
        # _fired_this_stream (checked above) covers.
        self._event("heard", text, source=line.source, fired=clip.name,
                    best=round(hit.score, 3), phrase=hit.phrase)
        try:
            self.play(hit.clip_id, why=f"auto:{hit.phrase} ({hit.score:.2f})")
            self._fired_this_stream.setdefault(STREAM_KEY, set()).add(hit.clip_id)
            with self._lock:
                self._auto_fires.append(time.time())
        except Exception as exc:
            log.exception("auto play failed")
            self._event("error", f"auto play failed: {exc}")

    def trigger_prompt(self) -> str:
        """Trigger phrases as a whisper decoding prompt.

        Capped at whisper's usable prompt budget. Measured on the known-clip
        eval with the live greedy settings: cap 250 scored 0.956, cap 1200
        scored 0.993, both ~120 ms/clip -- a bigger prompt is free accuracy,
        and the earlier tighter cap (added chasing decode time) was costing
        recognition for no speed gain. Longest triggers win the space.
        """
        phrases = sorted({t for c in self.library.auto_clips() for t in c.triggers},
                         key=len, reverse=True)
        out, total = [], 0
        for phrase in phrases:
            if total + len(phrase) > 1200:
                break
            out.append(phrase)
            total += len(phrase) + 2
        return ", ".join(out)

    # ---------- random dropper ----------

    def _cooldown(self, key: str, default: float) -> float:
        """Effective cooldown, honouring the 'disable cooldowns' switch."""
        value = float(self.cfg["listen"].get(key, default))
        if self.cfg["listen"].get("cooldowns_off"):
            return min(value, BYPASS_COOLDOWN_S)
        return value

    def _budget_allows(self, now: float) -> bool:
        """True if the shared auto/random fire budget has room left.

        "Disable cooldowns" bypasses the budget as well. The toggle means
        "stop restraining me"; during trigger testing the 10-per-5-min budget
        was exhausted within minutes and every subsequent match was silently
        held, which read as the app ignoring perfectly-parsed speech.
        """
        listen = self.cfg["listen"]
        if listen.get("cooldowns_off"):
            return True
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
            cooldown = self._cooldown("cooldown_s", 180.0)
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
        self.listener = Listener(self.cfg["listen"], self.on_line, self.gate,
                                 prompt_fn=self.trigger_prompt)
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
        # Same rule as outputs: rebuild on a real CHANGE, not on the key merely
        # being present. A patch carries whatever the panel last read back, so
        # keying off presence would close and reopen every output stream -- and
        # cut whatever was playing -- on an unrelated edit.
        gain_changed = ("master_gain" in patch
                        and patch["master_gain"] != self.cfg.get("master_gain"))
        # Restart only when the capture list itself changed. The settings
        # panel sends the whole listen block on every edit, so keying off
        # "listen" in patch tore the listener down for a threshold nudge --
        # which made the meters vanish and reappear, shoving the panel around
        # under the cursor mid-click.
        old_inputs = self.cfg.get("listen", {}).get("inputs")
        new_inputs = patch.get("listen", {}).get("inputs", old_inputs)
        listen_changed = new_inputs != old_inputs

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

    def live_streams(self) -> list:
        """The running transcripts, expiring any past the silence gap.

        This is the SAME state matching reads, so what the UI shows is
        exactly what a trigger would be compared against.
        """
        gap = float(self.cfg["listen"].get("stream_gap_s", 10.0))
        now = time.time()
        out = []
        for src, (words, seen) in list(self._stream.items()):
            age = now - seen
            if age > gap:
                # Expire it for real, so the next line starts a fresh
                # utterance and the display goes quiet.
                self._stream[src] = ([], seen)
                continue
            if words:
                out.append({"source": src, "text": " ".join(words),
                            "age": round(age, 1), "words": len(words)})
        return out

    def _cooling(self) -> dict:
        """Seconds of cooldown left per clip, for clips currently resting.

        Only auto-fire is blocked by a cooldown -- a manual press always
        works -- so this is shown as a countdown rather than a locked pad.
        """
        cooldown = self._cooldown("cooldown_s", 180.0)
        now = time.time()
        with self._lock:
            fired = dict(self._clip_last_fired)
        out = {}
        for clip_id, last in fired.items():
            left = cooldown - (now - last)
            if left > 0:
                out[clip_id] = round(left, 1)
        return out

    def _global_cooldown_left(self) -> float:
        left = self._cooldown("global_cooldown_s", 2.0) - (time.time() - self._last_fire_at)
        return round(max(0.0, left), 1)

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
            "cooldowns_off": bool(self.cfg["listen"].get("cooldowns_off")),
            "cooling": self._cooling(),
            "cooldown_s": self._cooldown("cooldown_s", 180.0),
            "global_cooldown_left": self._global_cooldown_left(),
            # The live word stream. Expired entries are dropped HERE, not
            # merely when the next line arrives -- otherwise the display kept
            # showing words long after they had aged out of matching, which
            # looks like the listener has stalled.
            "streams": self.live_streams(),
            "stream_gap_s": float(self.cfg["listen"].get("stream_gap_s", 10.0)),
            "random": {
                "enabled": bool(self.cfg.get("random", {}).get("enabled")),
                "eligible": len(self.library.random_clips()),
                "running": bool(self._random_thread and self._random_thread.is_alive()),
            },
            "listener": self.listener.status() if self.listener else {"running": False},
        }
