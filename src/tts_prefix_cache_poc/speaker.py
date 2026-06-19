from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from contextlib import suppress

from .audio import Audio, fade_in, ms_to_samples, samples_to_ms, silence, write_wav
from .boundary import BoundaryDetector
from .cache import PrefixCache, prefix_cache_key
from .config import (
    BoundaryConfig,
    BoundaryResult,
    PlaybackConfig,
    Synthesizer,
    VoiceParams,
)
from .sink import BufferedWavSink, write_paced

Logger = Callable[[str], None]


async def synthesize_cached_prefix_audio(
    *,
    tts: Synthesizer,
    prefix: str,
    voice: VoiceParams,
    sr: int,
    boundary_config: BoundaryConfig,
) -> Audio:
    audio = await tts.synthesize(prefix, voice)

    from .audio import trim_trailing_silence_keep

    return trim_trailing_silence_keep(
        audio,
        sr,
        threshold_db=boundary_config.silence_threshold_db,
        keep_ms=boundary_config.prefix_trim_keep_ms,
        frame_size=boundary_config.frame_size,
        hop=boundary_config.hop,
    )


def dump_debug_wavs(
    *,
    debug_dir: str,
    sr: int,
    tracks: dict[str, Audio],
) -> None:
    os.makedirs(debug_dir, exist_ok=True)

    for filename, audio in tracks.items():
        write_wav(os.path.join(debug_dir, filename), audio, sr)


class CachedPrefixSpeaker:
    def __init__(
        self,
        *,
        tts: Synthesizer,
        cache: PrefixCache[Audio],
        sink: BufferedWavSink,
        voice: VoiceParams,
        playback_config: PlaybackConfig,
        boundary_config: BoundaryConfig,
        logger: Logger | None = None,
    ):
        self.tts = tts
        self.cache = cache
        self.sink = sink
        self.voice = voice
        self.playback = playback_config
        self.boundary_config = boundary_config
        self.boundary_detector = BoundaryDetector(
            sr=playback_config.sr,
            config=boundary_config,
        )
        self.logger = logger

    async def prewarm_prefix(self, prefix: str) -> None:
        await self._get_cached_prefix(prefix)

    async def speak(self, *, prefix: str, rest: str) -> BoundaryResult:
        full_text = prefix + rest
        synth_start = time.perf_counter()
        full_task = asyncio.create_task(self.tts.synthesize(full_text, self.voice))
        full_audio: Audio | None = None
        synth_elapsed_ms: float | None = None

        self._log("[synth] started full synthesis in background")

        try:
            cached_prefix_audio = await self._get_cached_prefix(prefix)

            if full_task.done():
                full_audio = await full_task
                synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0

            await write_paced(
                sink=self.sink,
                audio=cached_prefix_audio,
                sr=self.playback.sr,
                label="cached prefix",
                chunk_ms=self.playback.chunk_ms,
                logger=self.logger,
            )

            silence_written = 0
            if full_audio is None:
                silence_written = await self._write_silence_until_ready(full_task)
                full_audio = await full_task
                synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0

            self._log(f"[synth] full audio ready after {synth_elapsed_ms:.1f} ms")

            if silence_written:
                self._log(
                    "[stream] latency silence added, "
                    f"{samples_to_ms(self.playback.sr, silence_written):.1f} ms"
                )

            boundary = self.boundary_detector.find_cut(
                cached_prefix_audio=cached_prefix_audio,
                full_audio=full_audio,
            )

            self._log(
                "[boundary] cut at "
                f"{samples_to_ms(self.playback.sr, boundary.cut_sample):.1f} ms "
                f"into full audio using {boundary.method}"
            )

            continuation = fade_in(
                full_audio[boundary.cut_sample :],
                fade_samples=ms_to_samples(
                    self.playback.sr,
                    self.boundary_config.continuation_fade_in_ms,
                ),
            )

            if self.playback.dump_debug:
                dump_debug_wavs(
                    debug_dir=self.playback.debug_dir,
                    sr=self.playback.sr,
                    tracks={
                        "debug_cached_prefix.wav": cached_prefix_audio,
                        "debug_full_raw.wav": full_audio,
                        "debug_continuation_after_cut.wav": continuation,
                    },
                )
                self._log(f"[debug] wrote debug WAVs to {self.playback.debug_dir}")

            await write_paced(
                sink=self.sink,
                audio=continuation,
                sr=self.playback.sr,
                label="generated continuation",
                chunk_ms=self.playback.chunk_ms,
                logger=self.logger,
            )

            return boundary

        finally:
            if full_task.done():
                with suppress(asyncio.CancelledError, Exception):
                    full_task.result()
            else:
                full_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await full_task

    async def _get_cached_prefix(self, prefix: str) -> Audio:
        key = prefix_cache_key(prefix, self.voice)
        was_cached = key in self.cache

        if was_cached:
            self._log("[goblin cache] hit, cached prefix is ready immediately")
        else:
            self._log("[goblin cache] miss, synthesizing prefix once")

        audio = await self.cache.get_or_create(
            key,
            lambda: synthesize_cached_prefix_audio(
                tts=self.tts,
                prefix=prefix,
                voice=self.voice,
                sr=self.playback.sr,
                boundary_config=self.boundary_config,
            ),
        )

        if not was_cached:
            self._log(
                f"[goblin cache] stored prefix, "
                f"{samples_to_ms(self.playback.sr, len(audio)):.1f} ms"
            )

        return audio

    async def _write_silence_until_ready(self, task: asyncio.Task[Audio]) -> int:
        silence_chunk = silence(
            self.playback.sr,
            self.playback.latency_silence_chunk_ms,
        )
        silence_written = 0

        while not task.done():
            if silence_written == 0:
                self._log("[stream] full synth not ready, adding same-stream silence")

            await self.sink.write(silence_chunk)
            silence_written += len(silence_chunk)
            await asyncio.sleep(self.playback.latency_silence_chunk_ms / 1000.0)

        return silence_written

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
