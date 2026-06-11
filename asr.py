"""Speech-to-text via local Whisper (mlx-whisper, Apple Silicon)."""

import numpy as np
import mlx_whisper

from config import WHISPER_REPO


def transcribe(audio: np.ndarray) -> str:
    """Speech -> raw text.

    Pass the float32 samples straight to Whisper. Giving it a *file path* would
    make it shell out to ffmpeg to decode; passing the in-memory array (already
    16 kHz mono float32 from sounddevice) skips ffmpeg entirely.
    """
    result = mlx_whisper.transcribe(
        audio.astype(np.float32), path_or_hf_repo=WHISPER_REPO
    )
    return result["text"].strip()
