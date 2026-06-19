from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np

from ._audio import (Audio, ms_to_samples, samples_to_ms, to_mono_float32,
                     write_wav)
from .config import AudioSink

Logger = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


class BufferedWavSink:
    def __init__(self, *, path: str | Path, sample_rate: int):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._chunks: list[Audio] = []

    @property
    def audio(self) -> Audio:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32, copy=False)

    async def write(self, chunk: Audio) -> None:
        audio = to_mono_float32(chunk)
        if audio.size:
            self._chunks.append(audio.copy())
        await asyncio.sleep(0)

    def save(self) -> int:
        audio = self.audio
        write_wav(self.path, audio, self.sample_rate)
        return len(audio)


async def stream_audio(
    *,
    sink: AudioSink,
    audio: Audio,
    sample_rate: int,
    chunk_ms: float,
    label: str | None = None,
    logger: Logger | None = None,
    sleep: Sleep = asyncio.sleep,
) -> None:
    samples = to_mono_float32(audio)
    chunk_n = max(1, ms_to_samples(sample_rate, chunk_ms))

    if logger is not None and label is not None:
        logger(
            f"[stream] start {label}, {samples_to_ms(sample_rate, len(samples)):.1f} ms"
        )

    for start in range(0, len(samples), chunk_n):
        chunk = samples[start : start + chunk_n]
        await sink.write(chunk)
        await sleep(len(chunk) / sample_rate)

    if logger is not None and label is not None:
        logger(f"[stream] end {label}")
