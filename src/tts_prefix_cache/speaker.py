from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable
from contextlib import suppress
from types import TracebackType

from ._audio import (Audio, ms_to_samples, samples_to_ms, silence,
                     to_mono_float32)
from .cache import MemoryPrefixCache
from .config import (AudioSink, CacheStatus, LiveSpeakResult,
                     PrefixSpeakerConfig, RenderResult, Synthesizer)
from .events import PrefixSpeakerEvent, PrefixSpeakerLogger
from .sink import _QueueAudioSink, _write_audio_chunks
from .splice import (BoundaryResult, PreparedPrefix, SpliceResult,
                     _prepare_prefix_audio, _splice_from_full_audio,
                     _stitch_audio)

Sleep = Callable[[float], Awaitable[None]]
TimedAudioTask = asyncio.Task[tuple[Audio, float]]


class SpeechAudioStream:
    def __init__(
        self,
        producer: Callable[[AudioSink], Awaitable[LiveSpeakResult]],
    ) -> None:
        self._producer = producer
        self._sink = _QueueAudioSink()
        self._task: asyncio.Task[LiveSpeakResult] | None = None
        self._result: LiveSpeakResult | None = None
        self._started = False

    def __aiter__(self) -> AsyncIterator[Audio]:
        if self._started:
            raise RuntimeError("stream can only be iterated once")

        self._started = True
        return self._chunks()

    async def __aenter__(self) -> SpeechAudioStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def result(self) -> LiveSpeakResult:
        if self._task is None:
            raise RuntimeError("stream must be consumed before result is available")

        return await self._finish_task()

    async def aclose(self) -> None:
        self._sink.abort()

        if self._task is None:
            return

        if not self._task.done():
            self._task.cancel()

        with suppress(asyncio.CancelledError, Exception):
            await self._task

    def _ensure_task(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> LiveSpeakResult:
        error: BaseException | None = None

        try:
            return await self._producer(self._sink)
        except BaseException as exc:
            error = exc
            raise
        finally:
            with suppress(asyncio.CancelledError):
                await self._sink.close(error)

    async def _chunks(self) -> AsyncIterator[Audio]:
        completed = False
        self._ensure_task()

        try:
            async for chunk in self._sink.chunks():
                yield chunk

            await self._finish_task()
            completed = True
        finally:
            if not completed:
                await self.aclose()

    async def _finish_task(self) -> LiveSpeakResult:
        if self._result is not None:
            return self._result

        if self._task is None:
            raise RuntimeError("stream has not been started")

        self._result = await self._task
        return self._result


class PrefixSpeaker:
    def __init__(
        self,
        *,
        tts: Synthesizer,
        config: PrefixSpeakerConfig | None = None,
        cache: MemoryPrefixCache[Hashable, PreparedPrefix] | None = None,
        logger: PrefixSpeakerLogger | None = None,
        sleep: Sleep = asyncio.sleep,
    ):
        self.tts = tts
        self.config = config or PrefixSpeakerConfig()
        self.cache: MemoryPrefixCache[Hashable, PreparedPrefix] = (
            cache or MemoryPrefixCache()
        )
        self.logger = logger
        self._sleep = sleep
        self._default_cache_namespace: Hashable = object()

    async def render(
        self,
        *,
        prefix: str,
        continuation: str,
        cache_key: Hashable | None = None,
    ) -> tuple[Audio, RenderResult]:
        start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize_timed(prefix + continuation))

        self._emit("synthesis_started", mode="render")

        try:
            prepared, cache_status = await self._get_prepared_prefix(
                prefix,
                cache_key=cache_key,
            )
            prefix_ready_ms = (time.perf_counter() - start) * 1000.0

            full_audio, full_synthesis_ms = await full_task

            splice_start = time.perf_counter()
            audio, splice = await asyncio.to_thread(
                _stitch_audio,
                prefix=prepared,
                full_audio=full_audio,
                sample_rate=self.config.sample_rate,
                config=self.config.splice,
            )
            splice_ms = (time.perf_counter() - splice_start) * 1000.0

            total_elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._emit("render_completed", total_elapsed_ms=total_elapsed_ms)
            self._emit_boundary(splice.boundary)

            return audio, RenderResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                prefix_ready_ms=prefix_ready_ms,
                full_synthesis_ms=full_synthesis_ms,
                splice_ms=splice_ms,
                total_elapsed_ms=total_elapsed_ms,
            )

        finally:
            if not full_task.done():
                full_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await full_task

    async def speak(
        self,
        *,
        prefix: str,
        continuation: str,
        sink: AudioSink,
        cache_key: Hashable | None = None,
    ) -> LiveSpeakResult:
        start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize_timed(prefix + continuation))

        self._emit("synthesis_started", mode="live")

        try:
            prepared, cache_status = await self._get_prepared_prefix(
                prefix,
                cache_key=cache_key,
            )
            prefix_ready_ms = (time.perf_counter() - start) * 1000.0

            (
                splice,
                silence_samples,
                full_synthesis_ms,
                splice_ms,
            ) = await self._speak_live(
                prepared=prepared,
                full_task=full_task,
                sink=sink,
            )

            if silence_samples:
                self._emit(
                    "latency_silence_added",
                    silence_samples=silence_samples,
                    silence_ms=samples_to_ms(self.config.sample_rate, silence_samples),
                )

            self._emit_boundary(splice.boundary)

            continuation_start = time.perf_counter()
            await _write_audio_chunks(
                sink=sink,
                audio=splice.continuation,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="generated_continuation",
                logger=self.logger,
            )
            continuation_write_ms = (time.perf_counter() - continuation_start) * 1000.0

            total_elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._emit("speak_completed", total_elapsed_ms=total_elapsed_ms)

            return LiveSpeakResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                silence_samples=silence_samples,
                prefix_ready_ms=prefix_ready_ms,
                full_synthesis_ms=full_synthesis_ms,
                splice_ms=splice_ms,
                continuation_write_ms=continuation_write_ms,
                total_elapsed_ms=total_elapsed_ms,
            )

        finally:
            if not full_task.done():
                full_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await full_task

    async def _speak_live(
        self,
        *,
        prepared: PreparedPrefix,
        full_task: TimedAudioTask,
        sink: AudioSink,
    ) -> tuple[SpliceResult, int, float, float]:
        prefix_head, held_tail = self._split_prefix_holdback(prepared.playback_audio)
        timeline_start = time.perf_counter()
        queued_samples = 0

        await _write_audio_chunks(
            sink=sink,
            audio=prefix_head,
            sample_rate=self.config.sample_rate,
            chunk_ms=self.config.chunk_ms,
            label="cached_prefix",
            logger=self.logger,
        )
        queued_samples += len(prefix_head)

        if await self._full_task_ready_by_playout_deadline(
            full_task,
            timeline_start=timeline_start,
            queued_samples=queued_samples,
        ):
            full_audio, full_synthesis_ms = await full_task
            splice, splice_ms = await self._splice_timed(
                prefix=prepared,
                full_audio=full_audio,
                held_tail=held_tail,
            )
            return splice, 0, full_synthesis_ms, splice_ms

        if held_tail is not None and held_tail.size:
            await _write_audio_chunks(
                sink=sink,
                audio=held_tail,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="cached_prefix_tail",
                logger=self.logger,
            )
            queued_samples += len(held_tail)

        if await self._full_task_ready_by_playout_deadline(
            full_task,
            timeline_start=timeline_start,
            queued_samples=queued_samples,
        ):
            full_audio, full_synthesis_ms = await full_task
            silence_samples = 0
        else:
            (
                full_audio,
                silence_samples,
                full_synthesis_ms,
            ) = await self._await_full_audio(
                task=full_task,
                sink=sink,
            )

        splice, splice_ms = await self._splice_timed(
            prefix=prepared,
            full_audio=full_audio,
        )
        return splice, silence_samples, full_synthesis_ms, splice_ms

    async def _synthesize(self, text: str) -> Audio:
        return to_mono_float32(
            await self.tts.synthesize(
                text,
                sample_rate=self.config.sample_rate,
            )
        )

    async def _synthesize_timed(self, text: str) -> tuple[Audio, float]:
        start = time.perf_counter()
        audio = await self._synthesize(text)
        return audio, (time.perf_counter() - start) * 1000.0

    async def _get_prepared_prefix(
        self,
        prefix: str,
        *,
        cache_key: Hashable | None,
    ) -> tuple[PreparedPrefix, CacheStatus]:
        resolved_cache_key = (
            cache_key if cache_key is not None else self._default_cache_key(prefix)
        )

        prepared, status = await self.cache.get_or_create(
            resolved_cache_key,
            lambda: self._prepare_prefix(prefix),
        )

        self._emit(
            "prefix_cache",
            status=status,
            playback_samples=len(prepared.playback_audio),
            playback_duration_ms=samples_to_ms(
                self.config.sample_rate,
                len(prepared.playback_audio),
            ),
        )

        return prepared, status

    def _default_cache_key(self, prefix: str) -> Hashable:
        return (
            self._default_cache_namespace,
            self.config.sample_rate,
            self.config.splice,
            prefix,
        )

    async def _prepare_prefix(self, prefix: str) -> PreparedPrefix:
        audio = await self._synthesize(prefix)
        return await asyncio.to_thread(
            _prepare_prefix_audio,
            audio,
            sample_rate=self.config.sample_rate,
            config=self.config.splice,
        )

    async def _splice_from_full_audio(
        self,
        *,
        prefix: PreparedPrefix,
        full_audio: Audio,
        held_tail: object | None = None,
    ) -> SpliceResult:
        return await asyncio.to_thread(
            _splice_from_full_audio,
            prefix=prefix,
            full_audio=full_audio,
            sample_rate=self.config.sample_rate,
            config=self.config.splice,
            held_tail=held_tail,
        )

    async def _splice_timed(
        self,
        *,
        prefix: PreparedPrefix,
        full_audio: Audio,
        held_tail: object | None = None,
    ) -> tuple[SpliceResult, float]:
        start = time.perf_counter()
        splice = await self._splice_from_full_audio(
            prefix=prefix,
            full_audio=full_audio,
            held_tail=held_tail,
        )
        return splice, (time.perf_counter() - start) * 1000.0

    async def _await_full_audio(
        self,
        *,
        task: TimedAudioTask,
        sink: AudioSink,
    ) -> tuple[Audio, int, float]:
        silence_chunk = silence(
            self.config.sample_rate,
            30.0,
        )
        silence_samples = 0
        start_time = time.perf_counter()
        elapsed_audio_s = 0.0

        while not task.done():
            if silence_samples == 0:
                self._emit("latency_silence_started")

            await sink.write(silence_chunk)
            silence_samples += len(silence_chunk)

            elapsed_audio_s += len(silence_chunk) / self.config.sample_rate
            target_time = start_time + elapsed_audio_s
            now = time.perf_counter()
            if target_time > now:
                await self._sleep(target_time - now)

        audio, full_synthesis_ms = await task
        return audio, silence_samples, full_synthesis_ms

    async def _full_task_ready_by_playout_deadline(
        self,
        task: TimedAudioTask,
        *,
        timeline_start: float,
        queued_samples: int,
    ) -> bool:
        playout_seconds = queued_samples / self.config.sample_rate
        output_lead_seconds = self.config.output_lead_ms / 1000.0
        deadline = timeline_start + playout_seconds - output_lead_seconds

        if task.done():
            await task
            return True

        timeout = deadline - time.perf_counter()
        if timeout > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                pass

        if task.done():
            await task
            return True

        return False

    def _split_prefix_holdback(
        self, playback_audio: Audio
    ) -> tuple[Audio, Audio | None]:
        holdback = min(
            ms_to_samples(self.config.sample_rate, self.config.splice.holdback_ms),
            playback_audio.size,
        )

        if holdback <= 0:
            return playback_audio, None

        return playback_audio[:-holdback], playback_audio[-holdback:]

    def _emit_boundary(self, boundary: BoundaryResult) -> None:
        self._emit(
            "splice_boundary",
            cut_sample=boundary.cut_sample,
            cut_ms=samples_to_ms(self.config.sample_rate, boundary.cut_sample),
            expected_sample=boundary.expected_sample,
            expected_ms=samples_to_ms(
                self.config.sample_rate,
                boundary.expected_sample,
            ),
            method=boundary.method,
            confidence=boundary.confidence,
            normalized_cost=boundary.normalized_cost,
            duration_ratio=boundary.duration_ratio,
        )

    def _emit(self, name: str, **data: object) -> None:
        if self.logger is not None:
            self.logger(PrefixSpeakerEvent(name=name, data=data))
