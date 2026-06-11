"""Microphone capture into an in-memory buffer while push-to-talk is held."""

import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE


class Recorder:
    """Captures mic audio into memory between start() and stop()."""

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
