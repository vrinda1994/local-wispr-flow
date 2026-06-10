"""Local voice dictation with two modes — prose and code.

Two-stage pipeline (like Wispr Flow):
  1. mlx-whisper turns speech into an accurate raw transcript.
  2. Your local Gemma 4 server turns that transcript into the final text:
       • prose mode → cleaned-up dictation (punctuation, filler removal, tone)
       • code mode  → idiomatic source code in the target language
  The result is pasted at the cursor.

Hold a key to talk, release to transcribe:
  • Right Option (⌥)  → prose dictation
  • Right Command (⌘) → code dictation

Run:  uv run dictation.py
Stop: Ctrl-C in this terminal.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import namedtuple

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from pynput import keyboard

# --- Config ----------------------------------------------------------------
SAMPLE_RATE = 16_000            # Whisper's native rate
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
SERVER_URL = "http://127.0.0.1:9379/v1/chat/completions"
GEMMA_MODEL = "gemma4-e4b,gpu"

# Default language for code mode; override at any time by saying "in <language>".
CODE_LANGUAGE = "python"

# --- The two "intelligence" prompts (edit these to tune behaviour) ----------
PROSE_PROMPT = (
    "You are a dictation post-processor. You receive a raw, automatically "
    "transcribed snippet of someone speaking. Return a cleaned-up version that "
    "is ready to paste into whatever they are writing. Rules:\n"
    "- Fix punctuation, capitalization, and obvious transcription errors.\n"
    "- Remove filler words (um, uh, like, you know) and false starts.\n"
    "- Preserve the speaker's meaning, tone, and wording. Do NOT summarize, "
    "answer questions, or add anything.\n"
    "- If they clearly issue a formatting instruction (e.g. 'make this a bulleted "
    "list', 'new paragraph'), apply it.\n"
    "- Output ONLY the cleaned text, with no preamble, quotes, or explanation."
)

CODE_PROMPT = (
    f"You are pair-programming by voice, building up a file one instruction at a "
    f"time. Default language: {CODE_LANGUAGE}. If the speaker names a different "
    "language, use that one. Each user turn is a rough, auto-transcribed spoken "
    "instruction. You can see the code you produced on previous turns.\n"
    "Rules:\n"
    "- Output ONLY the NEW source code to insert at the cursor for THIS "
    "instruction — usually a single line or a small block. Do NOT repeat code "
    "you already emitted on earlier turns.\n"
    "- Stay consistent with the code so far: reuse the same variable/function "
    "names, indentation level, and style.\n"
    "- No markdown fences or backticks, no prose, no explanation — raw code only.\n"
    "- Interpret spoken operators sensibly (e.g. 'equals equals' -> ==, "
    "'not equal' -> !=, 'arrow' -> =>, 'new line' -> a line break).\n"
    "- Honor spoken edits: 'scratch that' / 'undo' means discard the previous "
    "instruction; reissue corrected code if they restate it.\n"
    "- The output is pasted directly into the editor, so it must be ready to run."
)

# Each push-to-talk key maps to a mode.
Mode = namedtuple("Mode", "label key_name system max_tokens is_code")
MODES = {
    keyboard.Key.alt_r: Mode("prose", "Right Option (⌥)", PROSE_PROMPT, 512, False),
    keyboard.Key.cmd_r: Mode("code", "Right Command (⌘)", CODE_PROMPT, 1024, True),
}

# Tap (press + release, no hold) this key to start a fresh code conversation.
RESET_KEY = keyboard.Key.ctrl_r          # Right Control
# Keep at most this many past turns of code context (user+assistant messages).
MAX_HISTORY_MESSAGES = 30

# If the whole utterance is just one of these, undo the last paste (editor ⌘Z)
# instead of sending it to the model. Matched on the full, cleaned transcript so
# a mid-sentence "scratch that" still reaches the LLM as a correction.
UNDO_PHRASES = {
    "undo", "undo that", "undo last", "scratch", "scratch that",
    "delete that", "delete that line", "delete line", "delete last line",
    "remove that", "remove that line",
}

# --- Lazy model load (first transcription is slow; later ones are fast) -----
import mlx_whisper  # noqa: E402  (imported here so the startup banner prints first)


class Recorder:
    """Captures mic audio into memory while active."""

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def start(self):
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda data, *_: self._frames.append(data.copy()),
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    """Speech -> raw text via local Whisper.

    Pass the float32 samples straight to Whisper. Giving it a *file path* would
    make it shell out to ffmpeg to decode; passing the in-memory array (already
    16 kHz mono float32 from sounddevice) skips ffmpeg entirely.
    """
    result = mlx_whisper.transcribe(
        audio.astype(np.float32), path_or_hf_repo=WHISPER_REPO
    )
    return result["text"].strip()


def run_llm(messages: list[dict], max_tokens: int) -> str:
    """Send a full chat message list to the local Gemma server, return the reply."""
    payload = {
        "model": GEMMA_MODEL,
        "top_k": 1,            # greedy/deterministic. (temp omitted on purpose:
        "max_tokens": max_tokens,  # temperature=0 crashes this GPU sampler build.)
        "messages": messages,
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def strip_code_fences(text: str) -> str:
    """Remove a leading ```lang fence and trailing ``` if the model added them."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]               # drop opening fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                   # drop closing fence
        t = "\n".join(lines)
    return t.strip("\n")


def is_undo_command(text: str) -> bool:
    """True if the whole utterance is a delete/undo command (not mid-sentence)."""
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()
    return cleaned in UNDO_PHRASES


def undo_in_editor():
    """Trigger the frontmost editor's own undo (⌘Z) — removes the last paste."""
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.cmd):
        kb.press("z")
        kb.release("z")


def paste(text: str):
    """Insert text at the cursor in the frontmost app (clipboard + Cmd-V)."""
    previous = ""
    try:
        previous = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(0.05)
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.1)
    if previous:  # restore the user's old clipboard
        pyperclip.copy(previous)


def _block(label: str, text: str):
    """Print a labelled, multi-line block with real line breaks (not repr)."""
    width = 52
    print(f"   ┌─ {label} " + "─" * max(2, width - len(label) - 5))
    for line in text.splitlines() or [""]:
        print(f"   │ {line}")
    print("   └" + "─" * (width - 1))


# --- State machine ---------------------------------------------------------
_recorder = Recorder()
_recording = False
_busy = False
_active_key = None
_lock = threading.Lock()
_code_history: list[dict] = []   # multi-turn context for code mode (user+assistant)


def _reset_code_session():
    if _code_history:
        _code_history.clear()
        print("🧹 code conversation reset — next code turn starts fresh\n")


def _process(audio: np.ndarray, mode: Mode):
    global _busy
    try:
        if audio.size < SAMPLE_RATE * 0.3:  # <0.3s -> ignore stray taps
            print("… too short, skipped")
            return
        print(f"⏳ transcribing… [{mode.label}]")
        raw = transcribe(audio)
        if not raw:
            print("… nothing heard")
            return
        _block("raw", raw)

        if is_undo_command(raw):
            undo_in_editor()                 # remove the last paste via editor ⌘Z
            del _code_history[-2:]           # and forget that turn (no-op if empty)
            print("↩️  undo — reverted the last paste\n")
            return

        print(f"🧠 {mode.label}…")

        if mode.is_code:
            # Multi-turn: the model sees prior turns so it can build code
            # line by line, keeping names/indentation consistent.
            messages = (
                [{"role": "system", "content": mode.system}]
                + _code_history
                + [{"role": "user", "content": raw}]
            )
            out = strip_code_fences(run_llm(messages, mode.max_tokens))
            _code_history.append({"role": "user", "content": raw})
            _code_history.append({"role": "assistant", "content": out})
            del _code_history[:-MAX_HISTORY_MESSAGES]  # cap context growth
            turns = len(_code_history) // 2
            _block(f"{mode.label} (turn {turns})", out)
        else:
            messages = [
                {"role": "system", "content": mode.system},
                {"role": "user", "content": raw},
            ]
            out = run_llm(messages, mode.max_tokens)
            _block(mode.label, out)

        paste(out)
        print("✅ pasted\n")
    except Exception as e:  # keep the daemon alive on any failure
        print(f"❌ {type(e).__name__}: {e}\n", file=sys.stderr)
    finally:
        with _lock:
            _busy = False


def _start(key):
    global _recording, _active_key
    with _lock:
        if _recording or _busy:   # already recording, busy, or held-key autorepeat
            return
        _recording = True
        _active_key = key
        _recorder.start()
        print(f"🔴 recording [{MODES[key].label}]… (release to stop)")


def _stop(key):
    global _recording, _busy, _active_key
    with _lock:
        if not _recording or key != _active_key:
            return
        _recording = False
        _busy = True
        mode = MODES[_active_key]
        _active_key = None
        audio = _recorder.stop()
        threading.Thread(target=_process, args=(audio, mode), daemon=True).start()


def _on_press(key):
    if key in MODES:
        _start(key)


def _on_release(key):
    if key in MODES:
        _stop(key)
    elif key == RESET_KEY:        # tap Right Control to clear the code session
        _reset_code_session()


def main():
    print("🎙️  Voice dictation ready.")
    for mode in MODES.values():
        print(f"    Hold {mode.key_name:18} → {mode.label}")
    print(f"    Tap  Right Control (⌃)   → reset code conversation")
    print(f"    Code language: {CODE_LANGUAGE}  (say 'in <language>' to override)")
    print(f"    ASR: {WHISPER_REPO}   LLM: {GEMMA_MODEL}")
    print("    Gemma server must be running on :9379. Ctrl-C to quit.\n")
    with keyboard.Listener(on_press=_on_press, on_release=_on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
