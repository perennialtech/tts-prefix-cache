from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable
from contextlib import suppress

from ._audio import (Audio, ms_to_samples, pcm16le_bytes, samples_to_ms,
                     silence, to_mono_float32)
from .cache import MemoryPrefixCache
from .config import (AudioSink, CacheStatus, PrefixSpeakerConfig, SpeakResult,
                     Synthesizer)
from .sink import QueueAudioSink, stream_audio
from .splice import (PreparedPrefix, SpliceResult, prepare_prefix_audio,
                     splice_from_full_audio)

Logger = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


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

    async def prewarm_prefix(
        self,
        prefix: str,
        *,
        key: Hashable | None = None,
    ) -> None:
        await self._get_prepared_prefix(prefix, key=key)

    async def audio_stream(
        self,
        *,
        prefix: str,
        rest: str,
        key: Hashable | None = None,
    ) -> AsyncIterator[Audio]:
        sink = QueueAudioSink()

        async def produce() -> None:
            error: BaseException | None = None

            try:
                await self.speak(
                    prefix=prefix,
                    rest=rest,
                    sink=sink,
                    key=key,
                )
            except BaseException as exc:
                error = exc
                raise
            finally:
                with suppress(asyncio.CancelledError):
                    await sink.close(error)

        task = asyncio.create_task(produce())

        try:
            async for chunk in sink.chunks():
                yield chunk
        except BaseException:
            if not task.done():
                task.cancel()

            sink.abort()

            with suppress(BaseException):
                await task

            raise

        await task

    async def pcm16le_stream(
        self,
        *,
        prefix: str,
        rest: str,
        key: Hashable | None = None,
    ) -> AsyncIterator[bytes]:
        async for chunk in self.audio_stream(prefix=prefix, rest=rest, key=key):
            data = pcm16le_bytes(chunk)
            if data:
                yield data

    async def speak(
        self,
        *,
        prefix: str,
        rest: str,
        sink: AudioSink,
        key: Hashable | None = None,
    ) -> SpeakResult:
        synth_start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize(prefix + rest))
        silence_samples = 0

        self._log("[synth] started full synthesis in background")

        try:
            prepared, cache_status = await self._get_prepared_prefix(prefix, key=key)

            holdback = min(
                ms_to_samples(self.config.sample_rate, self.config.splice.holdback_ms),
                prepared.playback_audio.size,
            )

            if holdback:
                prefix_head = prepared.playback_audio[:-holdback]
                held_tail = prepared.playback_audio[-holdback:]
            else:
                prefix_head = prepared.playback_audio
                held_tail = None

            await stream_audio(
                sink=sink,
                audio=prefix_head,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                pace=self.config.pace_audio,
                label="cached prefix",
                logger=self.logger,
                sleep=self._sleep,
            )

            if full_task.done():
                full_audio = await full_task
                splice = await self._splice_from_full_audio(
                    prefix=prepared,
                    full_audio=full_audio,
                    held_tail=held_tail,
                )
            else:
                if held_tail is not None and held_tail.size:
                    await stream_audio(
                        sink=sink,
                        audio=held_tail,
                        sample_rate=self.config.sample_rate,
                        chunk_ms=self.config.chunk_ms,
                        pace=self.config.pace_audio,
                        label="cached prefix tail",
                        logger=self.logger,
                        sleep=self._sleep,
                    )

                full_audio, silence_samples = await self._await_full_audio(
                    task=full_task,
                    sink=sink,
                )
                splice = await self._splice_from_full_audio(
                    prefix=prepared,
                    full_audio=full_audio,
                )

            synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0
            self._log(f"[synth] full audio ready after {synth_elapsed_ms:.1f} ms")

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

            await stream_audio(
                sink=sink,
                audio=splice.continuation,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                pace=self.config.pace_audio,
                label="generated continuation",
                logger=self.logger,
                sleep=self._sleep,
            )

            return SpeakResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                silence_samples=silence_samples,
                synth_elapsed_ms=synth_elapsed_ms,
            )

        finally:
            if not full_task.done():
                full_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await full_task

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
        key: Hashable | None,
    ) -> tuple[PreparedPrefix, CacheStatus]:
        cache_key = prefix if key is None else key

        prepared, status = await self.cache.get_or_create(
            cache_key,
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

    async def _prepare_prefix(self, prefix: str) -> PreparedPrefix:
        audio = await self._synthesize(prefix)
        return await asyncio.to_thread(
            prepare_prefix_audio,
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
            splice_from_full_audio,
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
            self.config.wait_silence_chunk_ms,
        )
        silence_samples = 0

        while not task.done():
            if silence_samples == 0:
                self._log("[stream] full synth not ready, adding same-stream silence")

            await sink.write(silence_chunk)
            silence_samples += len(silence_chunk)
            await self._sleep(self.config.wait_silence_chunk_ms / 1000.0)

        return await task, silence_samples

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
