from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import numpy as np

from .audio import (Audio, as_audio_array, ms_to_samples, samples_to_ms,
                    write_wav)

Logger = Callable[[str], None]


class BufferedWavSink:
    def __init__(self, *, path: str, sr: int):
        self.path = path
        self.sr = sr
        self._chunks: list[Audio] = []

    @property
    def audio(self) -> Audio:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32, copy=False)

    async def write(self, chunk: Audio | Sequence[float]) -> None:
        chunk_audio = as_audio_array(chunk)
        if chunk_audio.size:
            self._chunks.append(chunk_audio.copy())
        await asyncio.sleep(0)

    def save(self) -> int:
        audio = self.audio
        write_wav(self.path, audio, self.sr)
        return len(audio)


async def write_paced(
    *,
    sink: BufferedWavSink,
    audio: Audio,
    sr: int,
    label: str,
    chunk_ms: float,
    logger: Logger | None = None,
) -> None:
    chunk_n = max(1, ms_to_samples(sr, chunk_ms))
    total_ms = samples_to_ms(sr, len(audio))

    if logger is not None:
        logger(f"[stream] start {label}, {total_ms:.1f} ms")

    for start in range(0, len(audio), chunk_n):
        chunk = audio[start : start + chunk_n]
        await sink.write(chunk)
        await asyncio.sleep(len(chunk) / sr)

    if logger is not None:
        logger(f"[stream] end {label}")
