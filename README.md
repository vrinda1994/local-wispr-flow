# gemma-local-voice-dictation

A fully local, Wispr-Flow-style voice dictation tool for macOS. Press **F12**,
speak, press **F12** again — cleaned-up text is pasted at your cursor.

## Pipeline

```
For raw text: right option → mic capture → Whisper (ASR) → Gemma 4 (cleanup/tone) → paste at cursor
For code: right command → mic capture → Whisper (ASR) → Gemma 4 (cleanup/tone) → paste at cursor

```

- **ASR:** `mlx-whisper` (`large-v3-turbo`) — fast + accurate on Apple Silicon.
- **LLM:** your local Gemma 4 server (LiteRT-LM) does punctuation, filler
  removal, formatting, and tone. Edit `SYSTEM_PROMPT` in `dictation.py` to change
  its behaviour.

## Setup

```bash
# 1. Start the Gemma server (separate terminal) — small model so it fits 18GB:
litert-lm serve gemma4-e4b,gpu        # serves on :9379

# 2. Run the dictation daemon:
uv run dictation.py
```

## Required macOS permissions

System Settings → Privacy & Security, grant your terminal app (e.g. Terminal /
iTerm) access to:

- **Microphone** — to capture audio.
- **Accessibility** — to paste (⌘V) into other apps.
- **Input Monitoring** — to listen for the F12 hotkey globally.

> If F12 does nothing, your Mac may be sending it as a media key. Enable
> Settings → Keyboard → "Use F1, F2, etc. keys as standard function keys", or
> change `HOTKEY` in `dictation.py` to another key.

## Tuning

- Model: change `GEMMA_MODEL` / `WHISPER_REPO` in `dictation.py`.
- Behaviour: edit `SYSTEM_PROMPT`.
- Determinism: requests use `top_k: 1` (greedy) and omit `temperature` — do NOT
  set `temperature: 0` (it crashes the GPU sampler in this LiteRT-LM build).

## Future areas
First-run gotchas (all in the README): macOS will need Microphone, Accessibility, and Input Monitoring permissions for your terminal app — grant them in System
Settings → Privacy & Security, then restart the terminal. If F12 does nothing, enable "Use F1, F2… as standard function keys" or change HOTKEY. The very first
transcription is slow (Whisper downloads large-v3-turbo ~1.5GB and warms up); subsequent ones are fast.

AI feature roadmap


Tier 1 — polish 
- Punctuation, capitalization, filler/false-start removal, faithful wording.

Tier 2 — tone & format (your "tone/emotion" ask)
- Tone modes: bind different hotkeys to professional / casual / concise prompts (e.g. F12 = verbatim, F13 = "rewrite professionally").
- App-aware formatting: detect the frontmost app (NSWorkspace.frontmostApplication via pyobjc — already installed) and format accordingly — Slack-casual vs
  email-formal vs commit-message vs code comment.
- Emotion/tone tags: have Gemma annotate or adapt (add emphasis/emoji when excited, keep terse when curt).

Tier 3 — commands & intelligence
- Spoken commands: "new paragraph", "make that a bulleted list", "scratch that", "all caps" — the prompt already handles simple cases; expand it.
- Self-correction: "...send it Friday, no wait, Monday" → Gemma resolves the correction instead of typing both.
- Translation: "dictate in Spanish" → speak English, paste Spanish.
- Custom vocabulary: inject a glossary of your names/jargon/acronyms into the system prompt so transcription errors get fixed deterministically.
- Modes beyond dictation: a "rewrite selection" hotkey (copy selected text → Gemma → paste) for instant rephrase/summarize of anything on screen.

Tier 4 — UX polish
- Streaming feedback (HUD overlay showing the live transcript).
- VAD auto-stop (end recording on silence instead of a second F12 press).
- A menu-bar app (rewrite the daemon in Swift later for nicer hotkey/permission UX).

