"""User-tunable settings for the voice dictation app."""

import os

from pynput import keyboard

# --- Audio / models --------------------------------------------------------
SAMPLE_RATE = 16_000            # Whisper's native rate
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
SERVER_URL = "http://127.0.0.1:9379/v1/chat/completions"
GEMMA_MODEL = "gemma4-e4b,gpu"

# Which engine turns speech into text. Override at launch:
#   ASR_BACKEND=gemma uv run dictation.py   → audio sent straight to Gemma (no Whisper)
#   (default)                               → mlx-whisper does ASR
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()

# --- Push-to-talk keys (hold to talk, release to transcribe) ---------------
PROSE_KEY = keyboard.Key.alt_r    # Right Option (⌥)
CODE_KEY = keyboard.Key.cmd_r     # Right Command (⌘)
RESET_KEY = keyboard.Key.ctrl_r   # Right Control (⌃) — tap to reset code session

# --- Request limits --------------------------------------------------------
PROSE_MAX_TOKENS = 512
CODE_MAX_TOKENS = 1024

# --- Code mode -------------------------------------------------------------
CODE_LANGUAGE = "python"
MAX_HISTORY_MESSAGES = 30        # cap multi-turn code context (user+assistant)

# --- Prose translation -----------------------------------------------------
# Default target language for prose; None = no translation (output English).
# In whisper mode you can also toggle at runtime by saying "dictate in Spanish".
# In gemma mode there is no transcript to intercept, so set it here / via env:
#   PROSE_LANGUAGE=Spanish ASR_BACKEND=gemma uv run dictation.py
DEFAULT_PROSE_LANGUAGE = os.getenv("PROSE_LANGUAGE") or None
