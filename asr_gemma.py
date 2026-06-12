"""Direct audio → Gemma path (no Whisper).

Gemma 4 E4B is natively multimodal and accepts audio, and the LiteRT-LM server's
OpenAI endpoint exposes that via the `input_audio` content part. This module
sends the recorded mic audio straight to Gemma, so a single model does the
transcription (and cleanup) in one shot — eliminating the Whisper ASR stage.

Trade-off vs. the Whisper path (asr.py):
  + One model, no Whisper download, and Gemma hears tone/prosody directly.
  - A dedicated ASR is usually more accurate word-for-word, and audio encoding
    runs on CPU in this build, so it can be slower.

Public functions:
  • transcribe(audio)                         → verbatim text (drop-in for asr.transcribe)
  • process_audio(audio, system, max_tokens)  → one-shot transcribe + apply system prompt
"""

import base64
import io
import wave

import numpy as np
import requests

from config import GEMMA_MODEL, SAMPLE_RATE, SERVER_URL

# Model-card-aligned defaults (https://ai.google.dev/gemma/docs/core/model_card_4
# §6 Audio): concise system, canonical ASR instruction, audio placed after text.
_VERBATIM_SYSTEM = (
    "You are Gemma operating in speech mode. Output only the requested text — no "
    "commentary, labels, or quotation marks."
)
_ASR_INSTRUCTION = (
    "Transcribe the following speech segment in English into English text. "
    "Write numbers as digits and keep it on a single line."
)


def _wav_base64(audio: np.ndarray) -> str:
    """Encode float32 [-1, 1] mono samples as a base64 16-bit PCM WAV string."""
    pcm16 = (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _audio_message(instruction: str, audio: np.ndarray) -> dict:
    """Build the OpenAI user message carrying a text instruction + the audio."""
    return {"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "input_audio",
         "input_audio": {"data": _wav_base64(audio), "format": "wav"}},
    ]}


def _post(messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": GEMMA_MODEL,
        "top_k": 1,            # greedy/deterministic (temperature=0 crashes sampler)
        "max_tokens": max_tokens,
        "messages": messages,
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def transcribe(audio: np.ndarray) -> str:
    """Verbatim speech -> text via Gemma (canonical ASR prompt). Drop-in for
    asr.transcribe."""
    return _post(
        [{"role": "system", "content": _VERBATIM_SYSTEM},
         _audio_message(_ASR_INSTRUCTION, audio)],
        max_tokens=512,
    )


def process_audio(audio: np.ndarray, system_prompt: str, max_tokens: int = 512,
                  instruction: str | None = None) -> str:
    """One-shot audio call. `system_prompt` + `instruction` should follow the
    model card's audio templates (see prompts.gemma_prose_instruction)."""
    return _post(
        [{"role": "system", "content": system_prompt},
         _audio_message(instruction or _ASR_INSTRUCTION, audio)],
        max_tokens,
    )
