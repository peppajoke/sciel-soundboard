"""
Fuzzy trigger matching for the soundboard.

Auto mode feeds transcript lines in here; this decides whether any clip's
trigger phrases are "close enough" to fire. Kept free of audio/IO imports so
test_matcher.py can prove the behaviour with no microphone and no GPU.

WHY FUZZY: whisper mangles proper nouns constantly ("Sora" -> "Sore"/"Sora,"
/"so are"), and Jack asked to fire when speech is *close to* a phrase rather
than equal to it. Exact matching would almost never fire on game dialogue.

WHY SLIDING WINDOW: a trigger is usually a few words inside a longer line.
Comparing the whole line against a short trigger tanks the ratio -- "well
that's just great" vs trigger "just great" scores ~0.55 whole-line but 1.0
on the right window. So we score every word-window of comparable length and
keep the best.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Whisper likes to emit these as filler; they add length without meaning and
# drag ratios around. Stripped from both sides before comparison.
_FILLER = {"uh", "um", "erm", "ah", "eh", "hmm", "mm", "mhm", "like", "you know"}

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase, strip accents/punctuation, drop filler. Returns word list."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _FILLER]


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score(line: str, phrase: str) -> float:
    """Best similarity between `phrase` and any comparable window of `line`.

    Returns 0.0-1.0. Compares on the joined-word string rather than word lists
    so that a single mis-heard character costs a little, not a whole word.
    """
    hay = normalize(line)
    needle = normalize(phrase)
    if not hay or not needle:
        return 0.0

    target = " ".join(needle)
    n = len(needle)

    # Window sizes either side of the trigger length, because whisper both
    # drops words ("the") and splits them ("gonna" -> "going to").
    best = 0.0
    for size in range(max(1, n - 1), n + 2):
        if size > len(hay):
            # Trigger is longer than the whole line: still worth scoring the
            # line as-is, otherwise a truncated final chunk can never match.
            best = max(best, _ratio(" ".join(hay), target))
            break
        for i in range(len(hay) - size + 1):
            best = max(best, _ratio(" ".join(hay[i : i + size]), target))
            if best == 1.0:
                return 1.0
    return best


def stitch(tail: list[str], new: list[str], max_words: int = 60) -> list[str]:
    """Append `new` to the running transcript `tail`, removing the overlap.

    The listener re-scans a rolling window several times a second, so
    consecutive transcripts repeat most of their text:

        window 1: "fuck fuck"
        window 2: "fuck fuck fuck"

    Matching each emission separately is what made "fuck fuck fuck" fail --
    neither batch ever contained the whole phrase, and a trigger that spans a
    window boundary is invisible. Stitching produces one continuous sequence
    to match against instead.

    Overlap is found by the longest suffix of `tail` that is also a prefix of
    `new`. Longest-first matters: with repeated words, a short match would
    leave duplicates behind ("fuck fuck fuck fuck").
    """
    if not tail:
        return new[-max_words:]
    if not new:
        return tail[-max_words:]

    limit = min(len(tail), len(new))
    for k in range(limit, 0, -1):
        if tail[-k:] == new[:k]:
            return (tail + new[k:])[-max_words:]
    # No overlap at all: a gap in speech, or whisper changed its mind entirely.
    return (tail + new)[-max_words:]


@dataclass(frozen=True)
class Match:
    clip_id: str
    phrase: str
    score: float


def find(line: str, clips, default_threshold: float = 0.82,
         floor: float = 0.0) -> Match | None:
    """Best-scoring clip whose trigger clears its threshold, or None.

    `clips` is any iterable of objects with `.id`, `.triggers` (list[str]) and
    optional `.threshold`. Per-clip thresholds exist because short triggers
    need to be stricter -- a 2-word trigger hits 0.8 against unrelated speech
    far more often than a 6-word one does.

    `floor` raises the bar for every clip regardless of its own setting. It is
    used for very short transcripts, where a loose match is mostly noise but an
    exact one is still a legitimate hit.
    """
    best: Match | None = None
    for clip in clips:
        threshold = max(getattr(clip, "threshold", None) or default_threshold,
                        floor)
        for phrase in getattr(clip, "triggers", ()):
            s = score(line, phrase)
            if s >= threshold and (best is None or s > best.score):
                best = Match(clip.id, phrase, s)
    return best
