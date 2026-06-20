from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._audio import (Audio, ms_to_samples, samples_to_ms, to_mono_float32,
                     write_wav)
from .config import AudioSink

Logger = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class _QueueEnd:
    error: BaseException | None = None


class QueueAudioSink:
    def __init__(self, *, max_chunks: int = 16):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        self._queue: asyncio.Queue[Audio | _QueueEnd] = asyncio.Queue(
            maxsize=max_chunks
        )
        self._closed = False

    async def write(self, chunk: Audio) -> None:
        if self._closed:
            raise RuntimeError("audio sink is closed")

        audio = to_mono_float32(chunk)
        if audio.size:
            await self._queue.put(audio.copy())

    async def close(self, error: BaseException | None = None) -> None:
        if self._closed:
            return

        self._closed = True
        await self._queue.put(_QueueEnd(error))

    def abort(self, error: BaseException | None = None) -> None:
        if self._closed:
            return

        self._closed = True

        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._queue.put_nowait(_QueueEnd(error))

    async def chunks(self) -> AsyncIterator[Audio]:
        while True:
            item = await self._queue.get()

            if isinstance(item, _QueueEnd):
                if item.error is not None:
                    raise item.error
                return

            yield item

    def __aiter__(self) -> AsyncIterator[Audio]:
        return self.chunks()


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
    pace: bool = True,
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

    start_time = 0.0
    if pace:
        start_time = time.perf_counter()

    elapsed_audio_s = 0.0

    for start in range(0, len(samples), chunk_n):
        chunk = samples[start : start + chunk_n]
        await sink.write(chunk)

        if pace:
            elapsed_audio_s += len(chunk) / sample_rate
            target_time = start_time + elapsed_audio_s
            now = time.perf_counter()
            if target_time > now:
                await sleep(target_time - now)

    if logger is not None and label is not None:
        logger(f"[stream] end {label}")
