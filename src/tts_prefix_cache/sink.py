from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ._audio import Audio, ms_to_samples, samples_to_ms, to_mono_float32
from .config import AudioSink

Logger = Callable[[str], None]


@dataclass(frozen=True)
class _QueueEnd:
    error: BaseException | None = None


class _QueueAudioSink:
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


async def _write_audio_chunks(
    *,
    sink: AudioSink,
    audio: Audio,
    sample_rate: int,
    chunk_ms: float,
    label: str | None = None,
    logger: Logger | None = None,
) -> None:
    samples = to_mono_float32(audio)
    chunk_n = max(1, ms_to_samples(sample_rate, chunk_ms))

    if logger is not None and label is not None:
        logger(
            f"[stream] start {label}, {samples_to_ms(sample_rate, len(samples)):.1f} ms"
        )

    for start in range(0, len(samples), chunk_n):
        await sink.write(samples[start : start + chunk_n])

    if logger is not None and label is not None:
        logger(f"[stream] end {label}")
