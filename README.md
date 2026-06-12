# gemma-local-voice-dictation

A fully local, Wispr-Flow-style voice dictation tool for macOS (Apple Silicon).
Hold a key, speak, release — Whisper transcribes and a local Gemma 4 model cleans
it up, then the text is pasted at your cursor. Also does **code dictation** and
**audio-direct** transcription. No cloud, no API keys.

```
hold key → mic capture → ASR (Whisper or Gemma) → Gemma cleanup → paste at cursor
```

## Prerequisites

- macOS on Apple Silicon, [`uv`](https://docs.astral.sh/uv/), and a running
  Gemma 4 server via LiteRT-LM.
- Start the model server in its own terminal (small model so it fits ~18 GB):

  ```bash
  litert-lm serve gemma4-e4b,gpu          # OpenAI-compatible server on :9379
  ```

## Running the app

```bash
uv run dictation.py                                  # default: Whisper ASR
ASR_BACKEND=gemma uv run dictation.py                # audio-direct, no Whisper
PROSE_LANGUAGE=Spanish uv run dictation.py           # translate prose → Spanish
PROSE_LANGUAGE=French ASR_BACKEND=gemma uv run dictation.py   # combine them
```

Stop with `Ctrl-C`. The first Whisper run downloads `large-v3-turbo` (~1.5 GB)
and is slow to warm up; later runs are fast.

### Environment variables

| Variable        | Default   | Values / example            | Effect                                                  |
|-----------------|-----------|-----------------------------|---------------------------------------------------------|
| `ASR_BACKEND`   | `whisper` | `whisper`, `gemma`          | Which engine turns speech into text (see below).        |
| `PROSE_LANGUAGE`| _(unset)_ | `Spanish`, `French`, …      | Translate prose output into this language.              |

## Hotkeys & modes

| Key                       | Mode  | What it does                                                        |
|---------------------------|-------|--------------------------------------------------------------------|
| **Hold Right Option (⌥)** | prose | Dictate prose; Gemma cleans punctuation, fillers, formatting, tone. |
| **Hold Right Command (⌘)**| code  | Dictate code; builds a file line-by-line, multi-turn (see below).  |
| **Tap Right Control (⌃)** | —     | Reset the code conversation (start a fresh function/file).          |

Push-to-talk: hold the key while speaking, release to transcribe. (Modifier keys
are used instead of F-keys so they don't collide with the macOS media row.)

### Prose mode

Speak naturally; Gemma returns cleaned text. It also understands spoken commands
and self-corrections:

- **Formatting:** "new paragraph", "new line", "make that a bulleted list",
  "numbered list", "all caps", "lowercase", "quote that", "in parentheses".
- **Self-correction:** "send it Friday, no wait, Monday" → *"Send it Monday."*
- **Translation:** say "dictate in Spanish" to start, "dictate in English" to stop.
- **Undo:** say "undo" / "scratch that" / "delete that line" (as the whole
  utterance) to revert the last paste via the editor's ⌘Z.

> Voice commands that need an intermediate transcript — **undo** and the
> **translation toggle** — work in **Whisper mode only**. In Gemma mode, use the
> editor's own ⌘Z and set `PROSE_LANGUAGE` at launch instead.

### Code mode

Speak instructions; Gemma emits the code to insert at the cursor, keeping context
across turns so it stays consistent (names, indentation, style):

```
Hold ⌘: "a function fib that takes n"          → def fib(n):
Hold ⌘: "if n is less than 2 return n"         →     if n < 2:
                                                          return n
Hold ⌘: "otherwise return fib n-1 plus fib n-2"→     return fib(n - 1) + fib(n - 2)
Tap  ⌃: (reset for the next snippet)
```

Default language is Python (`CODE_LANGUAGE` in `config.py`); override by saying
"in Rust, ...".

## ASR backends

| | **Whisper** (`ASR_BACKEND=whisper`, default) | **Gemma** (`ASR_BACKEND=gemma`) |
|---|---|---|
| Pipeline | Whisper transcribes → Gemma cleans (2 stages) | Audio → Gemma in **one call** |
| Accuracy | Highest (ASR-specialized) | Lower (4B generalist) |
| Speed    | Fast (optimized turbo model) | Slower (audio encodes on CPU) |
| Filler removal / commands | Reliable | Weaker — E4B can leak "um" and miss commands |
| Tone/prosody | Text-only | Hears your voice directly |
| Voice undo / translation toggle | ✅ | ❌ (use ⌘Z / `PROSE_LANGUAGE`) |

**Recommendation:** use Whisper mode for everyday dictation (best quality and
speed). Gemma mode is the simplest single-model path and the only one that hears
tone, but it trades away reliable cleanup and the voice-command layer.

## Required macOS permissions

System Settings → Privacy & Security — grant these to **whichever app launches the
process** (Terminal/iTerm, or your IDE if you run it from its integrated terminal),
then fully restart that app:

- **Microphone** — capture audio.
- **Accessibility** — paste (⌘V) and undo (⌘Z) into other apps.
- **Input Monitoring** — listen for the push-to-talk keys globally.

If the hotkeys do nothing, this is almost always a missing permission on the host
app.

## Roadmap

- **Custom vocabulary:** inject a glossary of names/jargon/acronyms so common
  mishearings are fixed deterministically.
- **Rewrite selection:** a hotkey to copy the selected text → Gemma → paste, for
  instant rephrase/summarize of anything on screen.
- **App-aware formatting:** detect the frontmost app and format accordingly
  (Slack-casual vs email-formal vs commit message).
- **UX:** VAD auto-stop (end on silence), a live-transcript HUD, a menu-bar app.
