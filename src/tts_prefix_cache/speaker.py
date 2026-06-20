from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable
from contextlib import suppress
from types import TracebackType

from ._audio import (Audio, ms_to_samples, pcm16le_bytes, samples_to_ms,
                     silence, to_mono_float32)
from .cache import MemoryPrefixCache
from .config import (AudioSink, CacheStatus, LiveSpeakResult,
                     PrefixSpeakerConfig, RenderResult, Synthesizer)
from .sink import _QueueAudioSink, _write_audio_chunks
from .splice import (PreparedPrefix, SpliceResult, _prepare_prefix_audio,
                     _splice_from_full_audio, _stitch_audio)

Logger = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


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


class SpeechPcm16leStream:
    def __init__(self, audio_stream: SpeechAudioStream) -> None:
        self._audio_stream = audio_stream

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._chunks()

    async def __aenter__(self) -> SpeechPcm16leStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def result(self) -> LiveSpeakResult:
        return await self._audio_stream.result()

    async def aclose(self) -> None:
        await self._audio_stream.aclose()

    async def _chunks(self) -> AsyncIterator[bytes]:
        completed = False

        try:
            async for chunk in self._audio_stream:
                data = pcm16le_bytes(chunk)
                if data:
                    yield data

            completed = True
        finally:
            if not completed:
                await self._audio_stream.aclose()


class PrefixSpeaker:
    def __init__(
        self,
        *,
        tts: Synthesizer,
        config: PrefixSpeakerConfig | None = None,
        cache: MemoryPrefixCache[Hashable, PreparedPrefix] | None = None,
        logger: Logger | None = None,
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
        full_task = asyncio.create_task(self._synthesize(prefix + continuation))

        self._log("[synth] started full synthesis for render")

        try:
            prepared, cache_status = await self._get_prepared_prefix(
                prefix,
                cache_key=cache_key,
            )
            full_audio = await full_task

            audio, splice = await asyncio.to_thread(
                _stitch_audio,
                prefix=prepared,
                full_audio=full_audio,
                sample_rate=self.config.sample_rate,
                config=self.config.splice,
            )

            wall_elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._log(f"[render] completed after {wall_elapsed_ms:.1f} ms")
            self._log(
                "[boundary] cut at "
                f"{samples_to_ms(self.config.sample_rate, splice.boundary.cut_sample):.1f} ms "
                f"using {splice.boundary.method}, confidence {splice.boundary.confidence:.3f}"
            )

            return audio, RenderResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                wall_elapsed_ms=wall_elapsed_ms,
            )

        finally:
            if not full_task.done():
                full_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await full_task

    def audio_stream(
        self,
        *,
        prefix: str,
        continuation: str,
        cache_key: Hashable | None = None,
    ) -> SpeechAudioStream:
        return SpeechAudioStream(
            lambda sink: self.speak(
                prefix=prefix,
                continuation=continuation,
                sink=sink,
                cache_key=cache_key,
            )
        )

    def pcm16le_stream(
        self,
        *,
        prefix: str,
        continuation: str,
        cache_key: Hashable | None = None,
    ) -> SpeechPcm16leStream:
        return SpeechPcm16leStream(
            self.audio_stream(
                prefix=prefix,
                continuation=continuation,
                cache_key=cache_key,
            )
        )

    async def speak(
        self,
        *,
        prefix: str,
        continuation: str,
        sink: AudioSink,
        cache_key: Hashable | None = None,
    ) -> LiveSpeakResult:
        start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize(prefix + continuation))

        self._log("[synth] started full synthesis in background")

        try:
            prepared, cache_status = await self._get_prepared_prefix(
                prefix,
                cache_key=cache_key,
            )

            splice, silence_samples = await self._speak_live(
                prepared=prepared,
                full_task=full_task,
                sink=sink,
            )

            if silence_samples:
                self._log(
                    "[stream] latency silence added, "
                    f"{samples_to_ms(self.config.sample_rate, silence_samples):.1f} ms"
                )

            self._log(
                "[boundary] cut at "
                f"{samples_to_ms(self.config.sample_rate, splice.boundary.cut_sample):.1f} ms "
                f"using {splice.boundary.method}, confidence {splice.boundary.confidence:.3f}"
            )

            await _write_audio_chunks(
                sink=sink,
                audio=splice.continuation,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="generated continuation",
                logger=self.logger,
            )

            wall_elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._log(f"[speak] completed after {wall_elapsed_ms:.1f} ms")

            return LiveSpeakResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                silence_samples=silence_samples,
                wall_elapsed_ms=wall_elapsed_ms,
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
        full_task: asyncio.Task[Audio],
        sink: AudioSink,
    ) -> tuple[SpliceResult, int]:
        prefix_head, held_tail = self._split_prefix_holdback(prepared.playback_audio)
        timeline_start = time.perf_counter()
        queued_samples = 0

        await _write_audio_chunks(
            sink=sink,
            audio=prefix_head,
            sample_rate=self.config.sample_rate,
            chunk_ms=self.config.chunk_ms,
            label="cached prefix",
            logger=self.logger,
        )
        queued_samples += len(prefix_head)

        if await self._full_task_ready_by_playout_deadline(
            full_task,
            timeline_start=timeline_start,
            queued_samples=queued_samples,
        ):
            full_audio = await full_task
            return (
                await self._splice_from_full_audio(
                    prefix=prepared,
                    full_audio=full_audio,
                    held_tail=held_tail,
                ),
                0,
            )

        if held_tail is not None and held_tail.size:
            await _write_audio_chunks(
                sink=sink,
                audio=held_tail,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="cached prefix tail",
                logger=self.logger,
            )
            queued_samples += len(held_tail)

        if await self._full_task_ready_by_playout_deadline(
            full_task,
            timeline_start=timeline_start,
            queued_samples=queued_samples,
        ):
            full_audio = await full_task
            silence_samples = 0
        else:
            full_audio, silence_samples = await self._await_full_audio(
                task=full_task,
                sink=sink,
            )

        return (
            await self._splice_from_full_audio(
                prefix=prepared,
                full_audio=full_audio,
            ),
            silence_samples,
        )

    async def _synthesize(self, text: str) -> Audio:
        return to_mono_float32(
            await self.tts.synthesize(
                text,
                sample_rate=self.config.sample_rate,
            )
        )

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

        if status == "hit":
            self._log("[prefix cache] hit")
        elif status == "joined":
            self._log("[prefix cache] joined in-flight prefix synthesis")
        else:
            self._log(
                "[prefix cache] miss, synthesized and stored prefix, "
                f"{samples_to_ms(self.config.sample_rate, len(prepared.playback_audio)):.1f} ms"
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

    async def _await_full_audio(
        self,
        *,
        task: asyncio.Task[Audio],
        sink: AudioSink,
    ) -> tuple[Audio, int]:
        silence_chunk = silence(
            self.config.sample_rate,
            30.0,
        )
        silence_samples = 0
        start_time = time.perf_counter()
        elapsed_audio_s = 0.0

        while not task.done():
            if silence_samples == 0:
                self._log("[stream] full synth not ready, adding same-stream silence")

            await sink.write(silence_chunk)
            silence_samples += len(silence_chunk)

            elapsed_audio_s += len(silence_chunk) / self.config.sample_rate
            target_time = start_time + elapsed_audio_s
            now = time.perf_counter()
            if target_time > now:
                await self._sleep(target_time - now)

        return await task, silence_samples

    async def _full_task_ready_by_playout_deadline(
        self,
        task: asyncio.Task[Audio],
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

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
