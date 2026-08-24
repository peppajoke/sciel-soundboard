"""
Matcher tests. No microphone, no GPU, no whisper -- pure text in, decision out.

These exist because auto-mode tuning is otherwise guesswork: the only way to
know whether raising the threshold to 0.9 kills real hits is to have a set of
realistic transcripts and check. Run with:  .venv\\Scripts\\python.exe test_matcher.py
"""

import matcher


class FakeClip:
    def __init__(self, cid, triggers, threshold=None):
        self.id = cid
        self.triggers = triggers
        self.threshold = threshold


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


def main():
    results = []

    # --- scoring ---
    results.append(check(
        "exact phrase scores 1.0",
        matcher.score("you shall not pass", "you shall not pass") == 1.0))

    results.append(check(
        "phrase inside a longer line still scores high",
        matcher.score("well I guess that is just great isn't it", "just great") > 0.9))

    results.append(check(
        "punctuation and case are ignored",
        matcher.score("Just GREAT!!", "just great") == 1.0))

    results.append(check(
        "filler words do not block a match",
        matcher.score("uh you know that is just great", "just great") > 0.9))

    results.append(check(
        "unrelated speech scores low",
        matcher.score("I need to go find more potions", "just great") < 0.7))

    results.append(check(
        "whisper mishearing one character still matches",
        matcher.score("you shall not pas", "you shall not pass") > 0.9))

    # --- firing decisions ---
    clips = [
        FakeClip("gandalf", ["you shall not pass"]),
        FakeClip("sad", ["that is just great", "just great"]),
        FakeClip("strict", ["no"], threshold=0.99),
    ]

    hit = matcher.find("wait, you shall not pass!", clips)
    results.append(check("fires on a real trigger", hit and hit.clip_id == "gandalf"))

    results.append(check(
        "does not fire on unrelated speech",
        matcher.find("where did I leave the map", clips) is None))

    results.append(check(
        "picks the best-scoring clip when two could match",
        (h := matcher.find("that is just great", clips)) and h.clip_id == "sad"))

    # A short trigger with a high per-clip threshold should stay quiet against
    # speech that merely contains a similar sound. This is the case that makes
    # per-clip thresholds worth having.
    results.append(check(
        "per-clip threshold suppresses a loose short trigger",
        matcher.find("I know what you mean", [clips[2]]) is None))

    results.append(check(
        "empty transcript never fires",
        matcher.find("", clips) is None))

    results.append(check(
        "raising the global threshold suppresses a marginal hit",
        matcher.find("you shall not pas", clips, 0.995) is None))

    # Short transcripts: whisper emits a lot of one-word chunks, most of them
    # noise. They are held to a floor rather than discarded, so an exact hit
    # still fires while a near-miss does not.
    short = [FakeClip("kiss", ["kiss"])]
    results.append(check(
        "an exact one-word match still fires under the short-line floor",
        (h := matcher.find("Kiss", short, 0.82, floor=0.95)) and h.score == 1.0))

    results.append(check(
        "a loose one-word match is suppressed by the floor",
        matcher.find("kissing", short, 0.82, floor=0.95) is None))

    results.append(check(
        "the floor overrides a permissive per-clip threshold",
        matcher.find("kissing", [FakeClip("kiss", ["kiss"], threshold=0.5)],
                     0.82, floor=0.95) is None))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
