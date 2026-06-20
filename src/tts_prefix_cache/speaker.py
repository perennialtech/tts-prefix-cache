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
from .sink import QueueAudioSink, stream_audio, write_audio_chunks
from .splice import (PreparedPrefix, SpliceResult, prepare_prefix_audio,
                     splice_from_full_audio, stitch_audio)

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

    async def render(
        self,
        *,
        prefix: str,
        rest: str,
        key: Hashable | None = None,
    ) -> tuple[Audio, SpeakResult]:
        synth_start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize(prefix + rest))

        self._log("[synth] started full synthesis for render")

        try:
            prepared, cache_status = await self._get_prepared_prefix(prefix, key=key)
            full_audio = await full_task

            audio, splice = await asyncio.to_thread(
                stitch_audio,
                prefix=prepared,
                full_audio=full_audio,
                sample_rate=self.config.sample_rate,
                config=self.config.splice,
            )

            synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0
            self._log(f"[synth] render audio ready after {synth_elapsed_ms:.1f} ms")
            self._log(
                "[boundary] cut at "
                f"{samples_to_ms(self.config.sample_rate, splice.boundary.cut_sample):.1f} ms "
                f"using {splice.boundary.method}, confidence {splice.boundary.confidence:.3f}"
            )

            return audio, SpeakResult(
                boundary=splice.boundary,
                cache_status=cache_status,
                silence_samples=0,
                synth_elapsed_ms=synth_elapsed_ms,
            )

        finally:
            if not full_task.done():
                full_task.cancel()

            with suppress(asyncio.CancelledError, Exception):
                await full_task

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

        self._log("[synth] started full synthesis in background")

        try:
            prepared, cache_status = await self._get_prepared_prefix(prefix, key=key)

            if self.config.playback_clock == "source":
                splice, silence_samples = await self._speak_source_clock(
                    prepared=prepared,
                    full_task=full_task,
                    sink=sink,
                )
            elif self.config.playback_clock == "buffered_timeline":
                splice, silence_samples = await self._speak_buffered_timeline(
                    prepared=prepared,
                    full_task=full_task,
                    sink=sink,
                )
            else:
                splice, silence_samples = await self._speak_sink_clock(
                    prepared=prepared,
                    full_task=full_task,
                    sink=sink,
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

            await write_audio_chunks(
                sink=sink,
                audio=splice.continuation,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="generated continuation",
                logger=self.logger,
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

    async def _speak_source_clock(
        self,
        *,
        prepared: PreparedPrefix,
        full_task: asyncio.Task[Audio],
        sink: AudioSink,
    ) -> tuple[SpliceResult, int]:
        prefix_head, held_tail = self._split_prefix_holdback(prepared.playback_audio)

        await stream_audio(
            sink=sink,
            audio=prefix_head,
            sample_rate=self.config.sample_rate,
            chunk_ms=self.config.chunk_ms,
            label="cached prefix",
            logger=self.logger,
            sleep=self._sleep,
        )

        if full_task.done():
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
            await stream_audio(
                sink=sink,
                audio=held_tail,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="cached prefix tail",
                logger=self.logger,
                sleep=self._sleep,
            )

        full_audio, silence_samples = await self._await_full_audio(
            task=full_task,
            sink=sink,
            source_paced=True,
        )
        return (
            await self._splice_from_full_audio(
                prefix=prepared,
                full_audio=full_audio,
            ),
            silence_samples,
        )

    async def _speak_buffered_timeline(
        self,
        *,
        prepared: PreparedPrefix,
        full_task: asyncio.Task[Audio],
        sink: AudioSink,
    ) -> tuple[SpliceResult, int]:
        prefix_head, held_tail = self._split_prefix_holdback(prepared.playback_audio)
        timeline_start = time.perf_counter()
        queued_samples = 0

        await write_audio_chunks(
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
            await write_audio_chunks(
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
                source_paced=True,
            )

        return (
            await self._splice_from_full_audio(
                prefix=prepared,
                full_audio=full_audio,
            ),
            silence_samples,
        )

    async def _speak_sink_clock(
        self,
        *,
        prepared: PreparedPrefix,
        full_task: asyncio.Task[Audio],
        sink: AudioSink,
    ) -> tuple[SpliceResult, int]:
        prefix_head, held_tail = self._split_prefix_holdback(prepared.playback_audio)

        await write_audio_chunks(
            sink=sink,
            audio=prefix_head,
            sample_rate=self.config.sample_rate,
            chunk_ms=self.config.chunk_ms,
            label="cached prefix",
            logger=self.logger,
        )

        if full_task.done():
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
            await write_audio_chunks(
                sink=sink,
                audio=held_tail,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="cached prefix tail",
                logger=self.logger,
            )

        full_audio, silence_samples = await self._await_full_audio(
            task=full_task,
            sink=sink,
            source_paced=False,
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
        source_paced: bool,
    ) -> tuple[Audio, int]:
        silence_chunk = silence(
            self.config.sample_rate,
            self.config.wait_silence_chunk_ms,
        )
        silence_samples = 0
        start_time = time.perf_counter()
        elapsed_audio_s = 0.0

        while not task.done():
            if silence_samples == 0:
                self._log("[stream] full synth not ready, adding same-stream silence")

            await sink.write(silence_chunk)
            silence_samples += len(silence_chunk)

            if source_paced:
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
