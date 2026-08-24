# soundboard

Manual + auto-triggered soundboard for the stream. Clips play into VoiceMeeter,
which feeds both Streamlabs and Discord. Auto mode transcribes game audio (or
your mic) and fires a clip when what it hears is close enough to a clip's
trigger phrase.

## Cloning onto another machine

The repo carries the clips, `library.json` and `config.json`, so a clone is a
working soundboard, not just the code. Only the virtualenv is excluded (2.3 GB,
machine-specific).

```
git clone <your-repo-url> soundboard
cd soundboard
powershell -ExecutionPolicy Bypass -File install.ps1
```

`install.ps1` installs anything missing (Python, ffmpeg — via winget), builds
the virtualenv, self-tests the imports, and puts a **Soundboard** shortcut on
the Desktop and in the Start Menu. It is safe to re-run, so it doubles as a
repair command. Add `-WithVoicemeeter` to install VoiceMeeter too (only needed
to route clips into Discord).

Then start it from the shortcut, or `run.cmd` — which opens the browser for
you.

Check **Settings -> Outputs** first: audio device names differ between machines
and the committed `config.json` holds the previous machine's choices.

## Run

```
run.cmd
```

Then open <http://localhost:8770>. It binds to `0.0.0.0`, so a phone on the LAN
can drive it while the game is full-screen.

**Use `run.cmd`, not `python server.py`.** The launcher puts the bundled NVIDIA
cuBLAS/cuDNN directories on PATH before Python starts, which is the only way
faster-whisper finds them. Starting `server.py` directly falls back to CPU
transcription (it still works, just slower — the fallback is deliberate so a
broken CUDA install never takes the soundboard down mid-stream).

## Adding clips

- **Web UI** — drag files onto the drop zone.
- **NAS** — drop files in `\\192.168.1.169\media\soundboard\inbox`. Picked up
  within ~20 s and moved to `inbox\_imported\` once indexed.

Everything is transcoded on import to 48 kHz stereo WAV at −16 LUFS, so clips
from wildly different sources all sit at the same level on stream. Originals
are kept in `clips\_originals\`.

## Needs clipping vs Finished

The board opens on **Needs clipping**, because a raw drop is nearly always
longer than the soundbite inside it. Every new clip starts unfinished — from
upload, from the NAS inbox, and every duplicate, since a duplicate exists to
become a *different* cut.

Tick "Finished" in the clip editor to move one over. Dragging a trim handle
ticks it for you (visibly, so you can untick it). Tab counts are always over
the whole library, so they stay meaningful while a search is active, and the
selected tab is remembered.

## Choosing what auto mode listens to

Settings lists every capture device as a checkbox, the same way outputs work.
Tick as many as you like — **each is scanned independently**, so you can run
your mic and the game at once, or several mics.

Two kinds appear in the list:

- **Microphones** — what a device *hears*. Pick the specific one (e.g. the
  NVIDIA Broadcast mic) rather than relying on the Windows default.
- **`[what it plays]`** — a loopback capture of an *output*, which is how game
  audio gets scanned. "System audio (game)" follows the default output.

Each capture is labelled by device name in the transcript feed, so you can tell
which one heard what.

Note that every capture runs on its own thread and COM is initialised per
thread — see `_com_init` in `listener.py`. Without it a second capture dies
instantly with `0x800401F0`.

## Settings

Settings save themselves as you change them — there is no Save button. They
persist server-side in `config.json` and are mirrored to localStorage so the
panel paints instantly on load instead of flashing stale values.

Saving is gated on the panel having finished loading. Without that gate an
early change event serialises a half-built DOM and writes empty device lists
over a good config.

## Sources

Every clip can be tagged with what it was clipped from. Manage the list at
**/sources** (link in the header) — add, rename, delete. Seeded with Wendy
Williams, Tim Robinson, Trump, DJ Khaled and Arrested Development.

Deleting a source never deletes clips; they just become unassigned. The board
has a source filter next to the tabs.

## Sorting

A–Z by default — with dozens of pads it is the only order you can build muscle
memory against, because it doesn't move. Also sortable by most played, newest
and oldest. The choice is remembered.

Play counts exist **only** to drive that sort and are not shown on the pads.
They are counted in memory and flushed to disk every 15 s, never on the press
itself, so counting can't sit in the click-to-sound path.

## Trimming

Upload the whole thing and cut it down afterwards — drag the handles on the
waveform in the clip editor. **The file is never cut.** Start/end are stored as
numbers and applied at playback, so the bounds stay adjustable forever and a
bad trim costs one drag instead of a re-upload. "Preview trim" plays just the
kept region, locally, not to the stream.

While trimming you can hear what you are doing: click anywhere on the waveform
to play from that point, hit Space to play the kept region, and releasing a
handle auto-previews 1.2 s of that edge so you hear whether the cut lands right.
A playhead tracks position.

Pads show the trimmed length with a ✂ when a clip is trimmed. Cooldowns and the
self-hear gate use the trimmed length too, so a 2 s bite cut from a 30 s upload
doesn't deafen auto mode for 30 seconds.

## Auto mode

Off at every launch, by design. Turn it on from the header.

Give a clip trigger phrases (edit → one phrase per line). Matching is fuzzy —
whisper mishears constantly, so "you shall not pas" still fires "you shall not
pass". How close is close enough is the **match threshold** slider:

| Threshold | Behaviour |
|-----------|-----------|
| 0.70      | Fires loosely. Expect false positives. |
| 0.82      | Default. Real hits fire, unrelated speech scores ~0.5. |
| 0.95+     | Near-exact only. |

Very short transcripts (under `min_words`) are **not discarded** — they are
held to `short_line_threshold` (0.95) instead. A loose match on one word is
noise, but "Kiss" against a `kiss` trigger is an exact hit and must still fire.

Short triggers need a stricter threshold than long ones — a two-word trigger
hits 0.8 against unrelated speech far more often than a six-word one. Set a
per-clip override in the clip editor for those.

The transcript panel has a **level meter per input**, because a silent room and
a wrongly-picked device look identical in an empty feed. It also prints an
occasional "(sound, no speech)" line when audio arrives but whisper finds no
words — again, so a working listener never looks dead.

**The live transcript panel is the tuning tool.** It shows every line heard and
the best score it got, including near-misses, so you can see *why* something
didn't fire instead of guessing.

### Restraints

- **Budget** — default 10 auto-fires per 5 minutes. A chatty cutscene would
  otherwise dump a dozen clips in a minute. Manual presses don't count against
  it. The header pill shows how much is used.
- **Per-clip cooldown** (3 min) — a soundbite lands once and then rests.
  Applies to trigger matches *and* random drops. Manual presses are exempt.
- **Global cooldown** (2 s) — stops pile-ups.
- **Self-hear gate** — the loopback listener is muted while a clip plays plus
  2.5 s after. Without it the soundboard hears its own output and re-triggers.

## Several cuts from one upload

**Duplicate** in the clip editor makes another entry from the *same* audio
file, so one long upload can yield as many soundbites as you want — duplicate,
drag new bounds, rename. It saves your current edits first, then opens the
copy, so the trim you were just working on is not lost.

Duplicates cost nothing: no re-encode, no extra disk, and they share the same
decoded samples in memory. Deleting one only removes the underlying file once
no other cut still points at it.

## Random drops

Settings → "Drop a random clip on an interval". Fires a clip unprompted every
4–8 minutes by default.

**Only whitelisted clips are eligible.** Tick "Include in random drops" in a
clip's editor; whitelisted pads show a 🎲. Nothing is whitelisted by default,
because dropping an arbitrary soundbite unprompted is only funny for the
handful you actually picked.

- The interval is a *range*, not a fixed number — a clip landing exactly every
  5:00 reads as a machine and the audience starts predicting it.
- Never repeats the previous pick when there is more than one eligible clip.
- Skips if something is already playing.
- Draws from the same budget as auto-fires (`use_budget` in `config.json`), so
  the two features can't conspire to make the stream noisy.
- Unlike auto mode, this setting **persists across restarts** — it is a slow
  background flourish, not something that can misfire on a sentence.

## One sound at a time

A new clip always cuts whatever is still playing. Two soundbites at once is
noise, not comedy, so there is deliberately **no setting to turn this off** —
the mixer holds at most one voice per device by construction. Any trim preview
also stops the moment a real clip fires.

## Latency

Click-to-sound is roughly **25 ms**, which is below the threshold where a press
feels delayed. It breaks down as ~22 ms output stream + ~2 ms HTTP + <0.1 ms to
queue the samples.

Four things get it there, and all four are easy to undo by accident:

- **WASAPI, not MME.** The same speaker is exposed under both. MME measured
  **192 ms** of output latency here; WASAPI is 22 ms. `_resolve()` in
  `player.py` deliberately prefers the WASAPI copy of a name match — without
  that, picking a device by name silently lands on MME, since it comes first.
- **`latency="low"`** on the output stream. The default is "high".
- **The whole library is decoded into memory at startup** (and after any
  import), so no press pays a disk read.
- **Streams stay open.** Opening a device per press would cost 50–200 ms alone.

The UI helps too: pads fire on *pointer-down*, not click, which skips the
50–150 ms wait for the mouse button to come back up, and the request is
fire-and-forget rather than awaited.

If presses ever start feeling late, check the output device first — that
MME/WASAPI difference is the whole budget many times over.

## Audio routing

Clips play to every device listed under Outputs. Default is `Voicemeeter
Input`; add your headphones to monitor.

- **Streamlabs** — add the VoiceMeeter output as an audio source.
- **Discord** — set its input to the VoiceMeeter B1 output, so your mic and the
  soundboard arrive mixed as one "microphone".

Note that installing VoiceMeeter made itself the default playback device. If
Windows audio sounds wrong, that's why.

## Layout

| File | Role |
|------|------|
| `server.py` | Flask API, web UI, NAS inbox watcher. Entry point. |
| `engine.py` | Wires everything; owns cooldowns, budget, event feed. |
| `listener.py` | Audio capture → faster-whisper → transcript lines. |
| `matcher.py` | Fuzzy trigger matching. Pure logic, no IO. |
| `player.py` | Multi-device playback. |
| `library.py` | Clip index + import/transcode pipeline. |
| `config.py` | Defaults + load/save. |

## Tests

```
.venv\Scripts\python.exe test_matcher.py
```

Matcher only — no mic, no GPU. Covers the firing decisions, since that's where
tuning mistakes actually hurt.
