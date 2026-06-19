from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

from ._audio import (Audio, samples_to_ms, silence, to_mono_float32,
                     trim_trailing_silence_keep, write_wav)
from ._boundary import PrefixClip, _frame_size_and_hop, splice_continuation
from .cache import MemoryPrefixCache, PrefixKey
from .config import (AudioSink, BoundaryConfig, CacheStatus,
                     PrefixSpeakerConfig, SpeakResult, Synthesizer, Voice)
from .sink import stream_audio

Logger = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


async def _build_prefix_clip(
    *,
    tts: Synthesizer,
    prefix: str,
    voice: Voice,
    sample_rate: int,
    boundary_config: BoundaryConfig,
) -> PrefixClip:
    raw = to_mono_float32(await tts.synthesize(prefix, voice, sample_rate=sample_rate))
    frame_size, hop = _frame_size_and_hop(sample_rate, boundary_config)

    playback_audio = trim_trailing_silence_keep(
        raw,
        sample_rate,
        threshold_db=boundary_config.silence_threshold_db,
        keep_ms=boundary_config.prefix_trim_keep_ms,
        frame_size=frame_size,
        hop=hop,
    )
    match_audio = trim_trailing_silence_keep(
        raw,
        sample_rate,
        threshold_db=boundary_config.silence_threshold_db,
        keep_ms=boundary_config.dtw_trim_keep_ms,
        frame_size=frame_size,
        hop=hop,
    )

    return PrefixClip(playback_audio=playback_audio, match_audio=match_audio)


def _dump_debug_wavs(
    *,
    debug_dir: Path,
    sample_rate: int,
    prefix: PrefixClip,
    full_audio: Audio,
    continuation: Audio,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_wav(debug_dir / "debug_cached_prefix.wav", prefix.playback_audio, sample_rate)
    write_wav(debug_dir / "debug_full_raw.wav", full_audio, sample_rate)
    write_wav(debug_dir / "debug_continuation_after_cut.wav", continuation, sample_rate)


class PrefixSpeaker:
    def __init__(
        self,
        *,
        tts: Synthesizer,
        voice: Voice,
        config: PrefixSpeakerConfig | None = None,
        cache: MemoryPrefixCache | None = None,
        logger: Logger | None = None,
        sleep: Sleep = asyncio.sleep,
    ):
        self.tts = tts
        self.voice = voice
        self.config = config or PrefixSpeakerConfig()
        self.cache = cache or MemoryPrefixCache()
        self.logger = logger
        self._sleep = sleep

    async def prewarm_prefix(self, prefix: str) -> None:
        await self._get_prefix_clip(prefix)

    async def speak(
        self,
        *,
        prefix: str,
        rest: str,
        sink: AudioSink,
    ) -> SpeakResult:
        full_text = prefix + rest
        synth_start = time.perf_counter()
        full_task = asyncio.create_task(self._synthesize(full_text))

        self._log("[synth] started full synthesis in background")

        try:
            prefix_clip, cache_status = await self._get_prefix_clip(prefix)

            await stream_audio(
                sink=sink,
                audio=prefix_clip.playback_audio,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
                label="cached prefix",
                logger=self.logger,
                sleep=self._sleep,
            )

            full_audio, silence_samples = await self._await_full_audio(
                task=full_task,
                sink=sink,
            )
            synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0

            self._log(f"[synth] full audio ready after {synth_elapsed_ms:.1f} ms")

            if silence_samples:
                self._log(
                    "[stream] latency silence added, "
                    f"{samples_to_ms(self.config.sample_rate, silence_samples):.1f} ms"
                )

            splice = splice_continuation(
                prefix=prefix_clip,
                full_audio=full_audio,
                sample_rate=self.config.sample_rate,
                config=self.config.boundary,
            )

            self._log(
                "[boundary] cut at "
                f"{samples_to_ms(self.config.sample_rate, splice.boundary.cut_sample):.1f} ms "
                f"into full audio using {splice.boundary.method}"
            )

            if self.config.debug_dir is not None:
                _dump_debug_wavs(
                    debug_dir=self.config.debug_dir,
                    sample_rate=self.config.sample_rate,
                    prefix=prefix_clip,
                    full_audio=full_audio,
                    continuation=splice.continuation,
                )
                self._log(f"[debug] wrote debug WAVs to {self.config.debug_dir}")

            await stream_audio(
                sink=sink,
                audio=splice.continuation,
                sample_rate=self.config.sample_rate,
                chunk_ms=self.config.chunk_ms,
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
                self.voice,
                sample_rate=self.config.sample_rate,
            )
        )

    async def _get_prefix_clip(self, prefix: str) -> tuple[PrefixClip, CacheStatus]:
        key = PrefixKey(
            text=prefix,
            voice=self.voice,
            sample_rate=self.config.sample_rate,
            prefix_trim_keep_ms=self.config.boundary.prefix_trim_keep_ms,
            dtw_trim_keep_ms=self.config.boundary.dtw_trim_keep_ms,
            silence_threshold_db=self.config.boundary.silence_threshold_db,
        )

        clip, status = await self.cache.get_or_create(
            key,
            lambda: _build_prefix_clip(
                tts=self.tts,
                prefix=prefix,
                voice=self.voice,
                sample_rate=self.config.sample_rate,
                boundary_config=self.config.boundary,
            ),
        )

        if status == "hit":
            self._log("[prefix cache] hit")
        elif status == "joined":
            self._log("[prefix cache] joined in-flight prefix synthesis")
        else:
            self._log(
                "[prefix cache] miss, synthesized and stored prefix, "
                f"{samples_to_ms(self.config.sample_rate, len(clip.playback_audio)):.1f} ms"
            )

        return clip, status

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
