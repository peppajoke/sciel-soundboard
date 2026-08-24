"""
Web UI + JSON API for the soundboard, plus the NAS inbox watcher.

Binds to 0.0.0.0 so the board can be driven from a phone or a second machine
on the LAN -- handy when the gaming laptop is full-screen in a game.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import config
import player as player_mod
from engine import Engine
from library import AUDIO_EXTS

ROOT = Path(__file__).resolve().parent
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOGDIR / "soundboard.log", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("soundboard.server")

app = Flask(__name__, static_folder=None)
engine = Engine()

# Auto mode always starts OFF, whatever the config file says. It was being
# persisted, so a restart mid-stream could bring it back unannounced -- the
# one behaviour that must never surprise you.
if engine.cfg.get("auto_enabled"):
    engine.update_config({"auto_enabled": False})


# ---------------------------------------------------------------- NAS inbox

def watch_nas(stop: threading.Event):
    """Import anything dropped in the NAS inbox, then move it to _imported/.

    Files are size-stable-checked before import: a clip copied over SMB is
    visible on the share long before the copy finishes, and importing a
    half-written file yields a truncated clip with no error.
    """
    seen: dict[str, int] = {}
    while not stop.is_set():
        inbox = Path(engine.cfg.get("nas_inbox", ""))
        try:
            if inbox.exists():
                done = inbox / "_imported"
                for src in inbox.iterdir():
                    if not src.is_file() or src.suffix.lower() not in AUDIO_EXTS:
                        continue
                    # Skip macOS sidecar junk. Copying from a Mac to an SMB
                    # share leaves an AppleDouble "._name.mp3" next to every
                    # real file -- it has an audio extension but no audio in
                    # it, so ffmpeg fails on every poll and fills the log.
                    if src.name.startswith("._") or src.name.startswith("."):
                        continue
                    size = src.stat().st_size
                    if seen.get(src.name) != size:
                        seen[src.name] = size       # still growing; wait a tick
                        continue
                    try:
                        clip = engine.library.import_file(src)
                        done.mkdir(exist_ok=True)
                        shutil.move(str(src), str(done / src.name))
                        seen.pop(src.name, None)
                        engine.warm_cache()
                        engine._event("info", f"imported from NAS: {clip.name}")
                        log.info("imported from NAS: %s", clip.name)
                    except Exception as exc:
                        log.error("NAS import failed for %s: %s", src.name, exc)
                        engine._event("error", f"NAS import failed: {src.name}: {exc}")
                        seen[src.name] = -1          # don't retry every poll
        except Exception as exc:
            log.warning("NAS inbox unreachable: %s", exc)
        stop.wait(float(engine.cfg.get("nas_poll_s", 20)))


# ---------------------------------------------------------------- API

def _clip_json(clip):
    return {
        "id": clip.id, "name": clip.name, "triggers": clip.triggers,
        "tags": clip.tags, "threshold": clip.threshold,
        "duration": clip.duration,
        "start": clip.start, "end": clip.end, "random_ok": clip.random_ok,
        "play_duration": round(clip.play_duration, 3),
        "finished": clip.finished, "source": clip.source, "plays": clip.plays,
        "gain": clip.gain,
    }


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/clips")
def list_clips():
    q = request.args.get("q", "")
    state = request.args.get("state", "all")
    items = engine.library.search(q, request.args.get("sort", "name"))
    source = request.args.get("source")
    if source == "none":
        items = [c for c in items if not c.source]
    elif source:
        items = [c for c in items if c.source == source]
    if state == "todo":
        items = [c for c in items if not c.finished]
    elif state == "done":
        items = [c for c in items if c.finished]
    # Counts are over the whole library, not the filtered view, so the tab
    # badges stay meaningful while a search is active.
    return jsonify({"clips": [_clip_json(c) for c in items],
                    "counts": engine.library.counts()})


@app.get("/sources")
def sources_page():
    return send_from_directory(ROOT / "static", "sources.html")


@app.get("/map")
def map_page():
    return send_from_directory(ROOT / "static", "map.html")


@app.get("/api/sources")
def list_sources():
    counts = engine.library.source_counts()
    items = [{"id": s.id, "name": s.name, "clips": counts.get(s.id, 0),
              "image": s.image}
             for s in sorted(engine.library.sources.values(),
                             key=lambda s: s.name.lower())]
    return jsonify({"sources": items, "unassigned": counts.get("", 0)})


@app.post("/api/sources")
def add_source():
    body = request.get_json(force=True) or {}
    try:
        src = engine.library.add_source(str(body.get("name", "")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": src.id, "name": src.name, "clips": 0})


@app.patch("/api/sources/<source_id>")
def rename_source(source_id):
    if source_id not in engine.library.sources:
        return jsonify({"error": "not found"}), 404
    body = request.get_json(force=True) or {}
    src = engine.library.rename_source(source_id, str(body.get("name", "")))
    return jsonify({"id": src.id, "name": src.name})


SOURCE_IMAGES = ROOT / "static" / "source_images"


@app.get("/source_images/<path:name>")
def source_image(name):
    return send_from_directory(SOURCE_IMAGES, name)


@app.post("/api/sources/<source_id>/image")
def set_source_image(source_id):
    """Accept an image for a source and normalise it to a 128px square PNG.

    Resizing on upload rather than in CSS keeps the repo small -- a handful of
    phone-camera JPEGs would outweigh the entire clip library otherwise -- and
    guarantees every button is the same size regardless of what was dropped in.
    ffmpeg does the work; it is already a hard dependency for clip import.
    """
    if source_id not in engine.library.sources:
        return jsonify({"error": "not found"}), 404
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "no image supplied"}), 400

    SOURCE_IMAGES.mkdir(parents=True, exist_ok=True)
    tmp = LOGDIR / f"_img_{int(time.time()*1000)}{Path(f.filename).suffix.lower()}"
    out_name = f"{source_id}.png"
    try:
        f.save(tmp)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
             # Cover-crop to a square so faces are not squashed.
             # 128 rather than the 38px the button draws: HiDPI screens ask
             # for 2x, and an upscaled avatar looks obviously soft.
             "-vf", "scale=128:128:force_original_aspect_ratio=increase,crop=128:128",
             "-frames:v", "1", str(SOURCE_IMAGES / out_name)],
            check=True)
    except subprocess.CalledProcessError:
        return jsonify({"error": "could not read that image"}), 400
    finally:
        tmp.unlink(missing_ok=True)

    engine.library.set_source_image(source_id, out_name)
    return jsonify({"id": source_id, "image": out_name})


@app.delete("/api/sources/<source_id>/image")
def clear_source_image(source_id):
    if source_id not in engine.library.sources:
        return jsonify({"error": "not found"}), 404
    src = engine.library.sources[source_id]
    if src.image:
        (SOURCE_IMAGES / src.image).unlink(missing_ok=True)
    engine.library.set_source_image(source_id, None)
    return jsonify({"ok": True})


@app.delete("/api/sources/<source_id>")
def delete_source(source_id):
    if source_id not in engine.library.sources:
        return jsonify({"error": "not found"}), 404
    orphaned = engine.library.delete_source(source_id)
    return jsonify({"ok": True, "unassigned_now": orphaned})


@app.get("/api/clips/<clip_id>/audio")
def clip_audio(clip_id):
    """Serve the clip so the browser can preview it locally (not on stream)."""
    clip = engine.library.clips.get(clip_id)
    if not clip:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(clip.path.parent, clip.path.name)


@app.get("/api/clips/<clip_id>/waveform")
def clip_waveform(clip_id):
    """Downsampled peak envelope for drawing the trim editor.

    Sent as small integers rather than raw samples -- a 30 s clip is 1.4M
    samples per channel, which is pointless to ship when the canvas is a few
    hundred pixels wide.
    """
    clip = engine.library.clips.get(clip_id)
    if not clip:
        return jsonify({"error": "not found"}), 404
    import numpy as np
    import soundfile as sf

    buckets = min(1200, max(200, int(request.args.get("buckets", 600))))
    data, rate = sf.read(str(clip.path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < buckets:
        buckets = max(1, len(mono))
    edges = np.linspace(0, len(mono), buckets + 1, dtype=int)
    peaks = [round(float(np.abs(mono[a:b]).max() if b > a else 0.0), 4)
             for a, b in zip(edges[:-1], edges[1:])]
    return jsonify({"peaks": peaks, "duration": len(mono) / rate})


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    added, failed = [], []
    for f in files:
        if not f.filename:
            continue
        tmp = ROOT / "logs" / f"_upload_{int(time.time()*1000)}_{Path(f.filename).name}"
        try:
            f.save(tmp)
            clip = engine.library.import_file(tmp, name=Path(f.filename).stem)
            added.append(_clip_json(clip))
        except Exception as exc:
            failed.append({"file": f.filename, "error": str(exc)})
        finally:
            tmp.unlink(missing_ok=True)
    engine.warm_cache()
    engine._event("info", f"uploaded {len(added)} clip(s)")
    return jsonify({"added": added, "failed": failed})


@app.post("/api/clips/<clip_id>/play")
def play_clip(clip_id):
    try:
        clip = engine.play(clip_id)
    except KeyError:
        return jsonify({"error": "not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(_clip_json(clip))


@app.post("/api/clips/<clip_id>/duplicate")
def duplicate_clip(clip_id):
    if clip_id not in engine.library.clips:
        return jsonify({"error": "not found"}), 404
    return jsonify(_clip_json(engine.library.duplicate(clip_id)))


@app.patch("/api/clips/<clip_id>")
def edit_clip(clip_id):
    if clip_id not in engine.library.clips:
        return jsonify({"error": "not found"}), 404
    body = request.get_json(force=True) or {}
    fields = {}
    if "name" in body:
        fields["name"] = str(body["name"]).strip()
    for key in ("triggers", "tags"):
        if key in body:
            fields[key] = [s.strip() for s in body[key] if str(s).strip()]
    if "start" in body or "end" in body:
        clip = engine.library.clips[clip_id]
        start = float(body.get("start", clip.start) or 0.0)
        raw_end = body.get("end", clip.end)
        end = None if raw_end in (None, "") else float(raw_end)
        # Validate here rather than at play time: a bad range saved now would
        # otherwise surface as a dead pad in the middle of a stream.
        if end is not None:
            end = min(end, clip.duration)
            if end - start < 0.05:
                return jsonify({"error": "trim range is too short"}), 400
        if not (0 <= start < clip.duration):
            return jsonify({"error": "start is outside the clip"}), 400
        fields["start"] = round(start, 3)
        fields["end"] = None if end is None else round(end, 3)

    if "source" in body:
        value = body["source"]
        if value in (None, ""):
            fields["source"] = None
        elif value in engine.library.sources:
            fields["source"] = value
        else:
            return jsonify({"error": "unknown source"}), 400

    if "finished" in body:
        fields["finished"] = bool(body["finished"])

    if "gain" in body:
        try:
            gain = float(body["gain"])
        except (TypeError, ValueError):
            return jsonify({"error": "gain must be a number"}), 400
        # Cap the boost: the mixer hard-clips above unity, and anything past
        # +12 dB on an already loudness-normalised clip is pure distortion.
        fields["gain"] = round(max(0.0, min(4.0, gain)), 4)

    if "random_ok" in body:
        fields["random_ok"] = bool(body["random_ok"])

    if "threshold" in body:
        value = body["threshold"]
        fields["threshold"] = None if value in (None, "") else float(value)
    return jsonify(_clip_json(engine.library.update(clip_id, **fields)))


@app.delete("/api/clips/<clip_id>")
def delete_clip(clip_id):
    engine.library.delete(clip_id)
    return jsonify({"ok": True})


@app.post("/api/stop")
def stop_all():
    engine.stop()
    return jsonify({"ok": True})


@app.post("/api/auto")
def set_auto():
    body = request.get_json(force=True) or {}
    want = bool(body.get("enabled"))
    engine.update_config({"auto_enabled": want})
    if want:
        engine.start_listener()
    # The listener keeps running when auto is switched off: the live transcript
    # is what you use to write triggers, and reloading whisper each toggle
    # costs ~10 s. on_line() checks auto_enabled before firing anything.
    return jsonify(engine.status())


@app.post("/api/listener")
def set_listener():
    body = request.get_json(force=True) or {}
    if body.get("running"):
        engine.start_listener()
    else:
        engine.stop_listener()
    return jsonify(engine.status())


@app.post("/api/random")
def set_random():
    body = request.get_json(force=True) or {}
    patch = {"random": {}}
    if "enabled" in body:
        patch["random"]["enabled"] = bool(body["enabled"])
    for key in ("min_minutes", "max_minutes"):
        if key in body:
            patch["random"][key] = float(body[key])
    if "use_budget" in body:
        patch["random"]["use_budget"] = bool(body["use_budget"])
    engine.update_config(patch)
    if engine.cfg["random"].get("enabled"):
        engine.start_random()
    else:
        engine.stop_random()
    return jsonify(engine.status())


@app.post("/api/reload")
def reload_library():
    """Re-read library.json from disk.

    Needed because fetch.py (or any other process) writes the index directly,
    and this process would otherwise overwrite it from stale memory.
    """
    engine.library.load()
    engine.warm_cache()
    engine._event("info", f"library reloaded: {len(engine.library.clips)} clips")
    return jsonify({"clips": len(engine.library.clips),
                    "sources": len(engine.library.sources)})


@app.get("/api/status")
def status():
    return jsonify(engine.status())


@app.get("/api/events")
def events():
    since = float(request.args.get("since", 0))
    return jsonify([e for e in list(engine.events) if e["at"] > since])


@app.get("/api/devices")
def devices():
    return jsonify({"outputs": player_mod.list_outputs(),
                    "inputs": player_mod.list_inputs()})


@app.route("/api/config", methods=["GET", "PATCH"])
def api_config():
    if request.method == "PATCH":
        return jsonify(engine.update_config(request.get_json(force=True) or {}))
    return jsonify(engine.cfg)


def main():
    stop = threading.Event()
    threading.Thread(target=watch_nas, args=(stop,), daemon=True).start()
    # Random dropping DOES persist across restarts, unlike auto mode: it is a
    # slow background flourish rather than something that can misfire on a
    # sentence, so surprise on boot is not a risk.
    if engine.cfg.get("random", {}).get("enabled"):
        engine.start_random()
        log.info("random dropper on")
    port = int(engine.cfg.get("port", 8770))
    log.info("soundboard on http://localhost:%s", port)
    log.info("outputs: %s", ", ".join(engine.player.devices) or "NONE")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
