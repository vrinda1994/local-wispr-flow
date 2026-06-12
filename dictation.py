"""Entry point: wires the modules into the push-to-talk dictation loop.

Hold a key to talk, release to transcribe:
  • Right Option (⌥)  → prose dictation
  • Right Command (⌘) → code dictation (multi-turn)
Tap Right Control (⌃) → reset the code conversation.

Prose voice commands (handled in commands.py / prompts.py):
  • formatting    — "new paragraph", "make that a bulleted list", "all caps"
  • self-correct  — "...send it Friday, no wait, Monday" → keeps only Monday
  • translation   — "dictate in Spanish" → speak English, paste Spanish
  • undo          — "undo" / "scratch that" → revert the last paste (⌘Z)

Run:  uv run dictation.py
Stop: Ctrl-C in this terminal.
"""

from __future__ import annotations

import sys
import threading
from collections import namedtuple

from pynput import keyboard

import commands
import config
import prompts

# Pick the ASR engine from the env var. Importing conditionally means that in
# `gemma` mode Whisper/mlx is never loaded at all.
if config.ASR_BACKEND == "gemma":
    import asr_gemma as asr_backend
else:
    import asr as asr_backend

from editor import paste, strip_code_fences, undo_in_editor
from llm import run_llm
from recorder import Recorder
from ui import show_block

Mode = namedtuple("Mode", "label key_name max_tokens kind")
MODES = {
    config.PROSE_KEY: Mode("prose", "Right Option (⌥)", config.PROSE_MAX_TOKENS, "prose"),
    config.CODE_KEY: Mode("code", "Right Command (⌘)", config.CODE_MAX_TOKENS, "code"),
}

# --- Session state ---------------------------------------------------------
_recorder = Recorder()
_recording = False
_busy = False
_active_key = None
_lock = threading.Lock()
_code_history: list[dict] = []          # multi-turn code context (whisper mode)
_code_so_far = ""                       # accumulated code context (gemma one-shot)
_prose_lang = config.DEFAULT_PROSE_LANGUAGE   # current prose translation target


def _reset_code_session():
    global _code_so_far
    if _code_history or _code_so_far:
        _code_history.clear()
        _code_so_far = ""
        print("🧹 code conversation reset — next code turn starts fresh\n")


def _apply_language_command(cmd):
    """cmd is ('set', '<Language>') or ('off', None) from commands.py."""
    global _prose_lang
    action, lang = cmd
    if action == "off":
        _prose_lang = None
        print("🌐 translation off — prose stays in English\n")
    else:
        _prose_lang = lang
        print(f"🌐 prose will be translated → {lang}\n")


def _handle_prose(raw: str, mode: Mode):
    # Whole-utterance translation switch is a command, not dictation.
    cmd = commands.parse_language_command(raw)
    if cmd:
        _apply_language_command(cmd)
        return
    system = prompts.build_prose_system(_prose_lang)
    out = run_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": raw}],
        mode.max_tokens,
    )
    label = mode.label + (f" → {_prose_lang}" if _prose_lang else "")
    show_block(label, out)
    paste(out)
    print("✅ pasted\n")


def _handle_code(raw: str, mode: Mode):
    messages = (
        [{"role": "system", "content": prompts.CODE_PROMPT}]
        + _code_history
        + [{"role": "user", "content": raw}]
    )
    out = strip_code_fences(run_llm(messages, mode.max_tokens))
    _code_history.append({"role": "user", "content": raw})
    _code_history.append({"role": "assistant", "content": out})
    del _code_history[:-config.MAX_HISTORY_MESSAGES]   # cap context growth
    show_block(f"{mode.label} (turn {len(_code_history) // 2})", out)
    paste(out)
    print("✅ pasted\n")


def _run_whisper(audio, mode: Mode):
    """Two-stage path: Whisper transcribes, then Gemma cleans (full command layer)."""
    print(f"⏳ transcribing… [{mode.label}]")
    raw = asr_backend.transcribe(audio)
    if not raw:
        print("… nothing heard")
        return
    show_block("raw", raw)

    # Voice commands only work here, where we have a transcript to inspect.
    if commands.is_undo(raw):
        undo_in_editor()
        del _code_history[-2:]              # forget that turn (no-op if empty)
        print("↩️  undo — reverted the last paste\n")
        return

    if mode.kind == "prose":
        _handle_prose(raw, mode)
    else:
        _handle_code(raw, mode)


def _run_gemma(audio, mode: Mode):
    """One-shot path: audio + system prompt go to Gemma in a single call.

    No intermediate transcript, so voice commands (undo, translation toggle) are
    not available here — but cleanup, formatting, and self-correction still work
    because the model applies them in-prompt, and it can hear your tone directly.
    """
    global _code_so_far
    print(f"🧠 {mode.label} (audio → Gemma, single call)…")
    if audio.size > 30 * config.SAMPLE_RATE:        # model card: 30s audio max
        print("   ⚠️  >30s of audio — Gemma may truncate it (model card limit)")
    if mode.kind == "prose":
        out = asr_backend.process_audio(
            audio, prompts.GEMMA_AUDIO_SYSTEM, mode.max_tokens,
            instruction=prompts.gemma_prose_instruction(_prose_lang),
        )
        label = mode.label + (f" → {_prose_lang}" if _prose_lang else "")
        show_block(label, out)      # raw Gemma output, pasted as-is (no extraction)
    else:
        instruction = (
            "You are pair-programming by voice. The code written so far is:\n"
            f"{_code_so_far or '(none yet)'}\n\n"
            "The user just spoke a new instruction (attached audio). Output ONLY "
            "the new code to append for it, consistent with the code above."
        )
        out = strip_code_fences(
            asr_backend.process_audio(
                audio, prompts.CODE_PROMPT, mode.max_tokens, instruction=instruction
            )
        )
        _code_so_far = (_code_so_far + "\n" + out).strip()
        show_block(f"{mode.label} (audio)", out)
    paste(out)
    print("✅ pasted\n")


def _process(audio, mode: Mode):
    global _busy
    try:
        if audio.size < config.SAMPLE_RATE * 0.3:   # <0.3s -> ignore stray taps
            print("… too short, skipped")
            return
        if config.ASR_BACKEND == "gemma":
            _run_gemma(audio, mode)
        else:
            _run_whisper(audio, mode)
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
    elif key == config.RESET_KEY:
        _reset_code_session()


def main():
    print("🎙️  Voice dictation ready.")
    for mode in MODES.values():
        print(f"    Hold {mode.key_name:18} → {mode.label}")
    print("    Tap  Right Control (⌃)   → reset code conversation")
    print(f"    Code language: {config.CODE_LANGUAGE}")
    print(f"    Prose translation: {_prose_lang or 'off'}  (say 'dictate in <language>')")
    if config.ASR_BACKEND == "gemma":
        print("    ASR backend: Gemma (audio-direct, single call, no Whisper)")
        print("      ↳ voice undo / 'dictate in <lang>' unavailable; use ⌃ to reset code,")
        print("        and PROSE_LANGUAGE=<lang> to translate")
    else:
        print(f"    ASR backend: Whisper ({config.WHISPER_REPO})")
    print("    Gemma server must be running on :9379. Ctrl-C to quit.\n")
    with keyboard.Listener(on_press=_on_press, on_release=_on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
