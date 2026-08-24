"""Fetch a soundbite from a URL straight into the library.

    .venv\\Scripts\\python.exe fetch.py --url URL --start 12.5 --end 15.2 \\
        --name "Bears. Beets." --source "The Office" \\
        --trigger "bears beets battlestar galactica" --trigger "bears beets"

Or a batch, which is what you want for more than one or two:

    .venv\\Scripts\\python.exe fetch.py --batch clips.json

where clips.json is a list of objects with the same keys as the flags.

Only the requested span is downloaded (yt-dlp --download-sections), so a
three-second bite does not pull a forty-minute video. The result goes through
the same import as a hand-dropped file: loudness-normalised, 48 kHz stereo,
original kept in clips/_originals/.

A clip arrives marked FINISHED when a span was given, since that span was
chosen deliberately -- unlike a raw drop, which starts in "needs clipping".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from library import Library

# The running server holds the library in memory and rewrites library.json on
# its own schedule. A second process writing that file gets silently clobbered
# -- a fetched clip simply vanishes, which is exactly what happened in testing.
# Reloading the server afterwards does not fix it either, because the race is
# unbounded: it can flush stale state at any moment. So when the server is up
# everything goes THROUGH it and there is exactly one writer. Direct library
# access is only the offline fallback.
BASE = "http://localhost:8770"

YTDLP = [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "yt_dlp"]
CRLF = "\r\n"


# ---------------------------------------------------------------- server path

def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/api/status", timeout=2).read()
        return True
    except Exception:
        return False


def _post_file(path: Path) -> dict:
    boundary = "----sbfetch"
    head = (
        "--" + boundary + CRLF
        + 'Content-Disposition: form-data; name="files"; filename="'
        + path.name + '"' + CRLF
        + "Content-Type: application/octet-stream" + CRLF + CRLF
    ).encode()
    tail = (CRLF + "--" + boundary + "--" + CRLF).encode()
    req = urllib.request.Request(
        f"{BASE}/api/upload", data=head + path.read_bytes() + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _patch(clip_id: str, fields: dict) -> None:
    req = urllib.request.Request(
        f"{BASE}/api/clips/{clip_id}", data=json.dumps(fields).encode(),
        headers={"Content-Type": "application/json"}, method="PATCH")
    urllib.request.urlopen(req, timeout=30).read()


def _source_id(name: str) -> str:
    with urllib.request.urlopen(f"{BASE}/api/sources", timeout=15) as r:
        for s in json.loads(r.read())["sources"]:
            if s["name"].lower() == name.lower():
                return s["id"]
    req = urllib.request.Request(
        f"{BASE}/api/sources", data=json.dumps({"name": name}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["id"]


def _import_via_server(spec: dict, downloaded: Path, trimmed: bool) -> str:
    result = _post_file(downloaded)
    if not result.get("added"):
        raise RuntimeError(str(result.get("failed") or "upload rejected"))
    clip = result["added"][0]

    fields = {"triggers": spec.get("triggers") or [], "finished": trimmed}
    if spec.get("name"):
        fields["name"] = spec["name"]
    if spec.get("source"):
        fields["source"] = _source_id(spec["source"])
    _patch(clip["id"], fields)
    return fields.get("name") or clip["name"]


# ---------------------------------------------------------------- offline path

def _import_direct(lib: Library, spec: dict, downloaded: Path, trimmed: bool) -> str:
    clip = lib.import_file(downloaded,
                           name=spec.get("name") or downloaded.stem,
                           triggers=spec.get("triggers") or [])
    if spec.get("source"):
        src = next((s for s in lib.sources.values()
                    if s.name.lower() == spec["source"].lower()), None)
        if src is None:
            src = lib.add_source(spec["source"], save=False)
        clip.source = src.id
    clip.finished = trimmed
    lib.save()
    return clip.name


# ---------------------------------------------------------------- download

def fetch_one(lib: Library | None, spec: dict) -> str:
    start, end = spec.get("start"), spec.get("end")
    trimmed = start is not None and end is not None

    with tempfile.TemporaryDirectory() as tmp:
        cmd = YTDLP + ["-q", "--no-playlist", "-x", "--audio-format", "mp3",
                       "-o", str(Path(tmp) / "clip.%(ext)s")]
        if trimmed:
            # Fetch only the span. force-keyframes makes the cut land where
            # asked rather than at the nearest keyframe, which can be seconds
            # away and would take the first word off.
            cmd += ["--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts"]
        cmd.append(spec["url"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip()[:300])

        got = list(Path(tmp).glob("clip.*"))
        if not got:
            raise RuntimeError("yt-dlp produced no file")

        if lib is None:
            return _import_via_server(spec, got[0], trimmed)
        return _import_direct(lib, spec, got[0], trimmed)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--start", type=float, help="seconds into the source")
    ap.add_argument("--end", type=float)
    ap.add_argument("--name")
    ap.add_argument("--source", help="e.g. 'The Office'; created if new")
    ap.add_argument("--trigger", action="append", dest="triggers", default=[])
    ap.add_argument("--batch", type=Path, help="JSON list of clip specs")
    args = ap.parse_args()

    if not args.batch and not args.url:
        ap.error("give --url or --batch")

    specs = (json.loads(args.batch.read_text(encoding="utf-8-sig"))
             if args.batch else
             [{"url": args.url, "start": args.start, "end": args.end,
               "name": args.name, "source": args.source,
               "triggers": args.triggers}])

    online = server_up()
    print("soundboard is running - importing through it" if online
          else "soundboard is not running - writing the library directly")
    lib = None if online else Library()

    ok = failed = 0
    for spec in specs:
        try:
            print(f"  added: {fetch_one(lib, spec)}")
            ok += 1
        except Exception as exc:
            print(f"  FAILED {spec.get('name') or spec.get('url')}: {exc}")
            failed += 1
    print(f"\n{ok} added, {failed} failed")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
