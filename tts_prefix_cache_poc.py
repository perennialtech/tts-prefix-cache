#!/usr/bin/env python3
"""
tts_prefix_cache_poc.py

Standalone proof-of-concept for fixed cached-prefix TTS playback with
audio-only continuation splicing. The fake TTS backend keeps this file runnable
without external services; replace FakeTTS.synthesize() when integrating.
"""

import argparse
import asyncio
import hashlib
import math
import os
import struct
import time
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Protocol, TypeVar

Audio = list[float]
T = TypeVar("T")
Number = TypeVar("Number", int, float)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceParams:
    model: str = "fake-goblin-tts"
    voice_id: str = "raccoon-narrator"
    speed: float = 1.0
    pitch: float = 1.0


@dataclass(frozen=True)
class FakeTTSConfig:
    sr: int = 24000
    base_latency_ms: float = 250.0
    per_char_latency_ms: float = 8.0


@dataclass(frozen=True)
class BoundaryConfig:
    silence_threshold_db: float = -43.0
    prefix_trim_keep_ms: float = 160.0
    dtw_trim_keep_ms: float = 20.0

    search_before_ms: float = 350.0
    search_after_ms: float = 900.0
    min_quiet_ms: float = 70.0
    keep_quiet_ms: float = 25.0
    low_amp_window_ms: float = 80.0
    continuation_fade_in_ms: float = 20.0

    frame_size: int = 1024
    hop: int = 256

    dtw_hop_ms: float = 10.0
    dtw_win_ms: float = 30.0
    dtw_max_search_multiplier: float = 1.8
    dtw_max_search_extra_ms: float = 600.0
    dtw_min_endpoint_ratio: float = 0.55


@dataclass(frozen=True)
class RunConfig:
    prefix: str = "Someone says: "
    rest: str = "The tiny raccoon found a shiny snack in the moonlit alley."
    out: str = "poc_output.wav"
    sr: int = 24000
    chunk_ms: float = 40.0
    use_dtw: bool = False
    dump_debug: bool = False
    debug_dir: str = "poc_debug"
    prewarm: bool = True
    latency_silence_chunk_ms: float = 30.0


@dataclass(frozen=True)
class BoundaryResult:
    cut_sample: int
    method: str


class Synthesizer(Protocol):
    async def synthesize(self, text: str, voice: VoiceParams) -> Audio: ...


# ---------------------------------------------------------------------------
# Basic audio utilities
# ---------------------------------------------------------------------------


def ms_to_samples(sr: int, ms: float) -> int:
    return int(sr * ms / 1000.0)


def samples_to_ms(sr: int, samples: int) -> float:
    return samples * 1000.0 / sr


def clamp(x: Number, lo: Number, hi: Number) -> Number:
    return max(lo, min(hi, x))


def silence(sr: int, ms: float) -> Audio:
    return [0.0] * ms_to_samples(sr, ms)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def write_wav(path: str, audio: Audio, sr: int) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)

        frames = bytearray()
        for x in audio:
            frames.extend(struct.pack("<h", int(clamp(x, -1.0, 1.0) * 32767.0)))

        wf.writeframes(bytes(frames))


# ---------------------------------------------------------------------------
# Fake TTS
# ---------------------------------------------------------------------------


class FakeTTS:
    def __init__(self, config: FakeTTSConfig):
        self.sr = config.sr
        self.base_latency_ms = config.base_latency_ms
        self.per_char_latency_ms = config.per_char_latency_ms

    async def synthesize(self, text: str, voice: VoiceParams) -> Audio:
        delay = self.base_latency_ms + self.per_char_latency_ms * len(text)
        await asyncio.sleep(delay / 1000.0)
        return self._render_fake_speech(text, voice)

    def _render_fake_speech(self, text: str, voice: VoiceParams) -> Audio:
        tokens = tokenize_text(text)
        words = [t for t in tokens if t[0] == "word"]
        total_words = max(1, len(words))

        base_seed = stable_int(voice.voice_id + "|" + text)
        base_f0 = 140.0 + (stable_int(voice.voice_id) % 70)

        audio: Audio = []
        word_i = 0

        for kind, value in tokens:
            if kind == "word":
                progress = word_i / max(1, total_words - 1)

                # Full-text synthesis includes the prefix so continuation prosody
                # can use the same context a real TTS provider would see.
                contour = 1.06 - 0.12 * progress
                text_context_shift = ((base_seed % 31) - 15) * 0.7

                word_seed = stable_int(value.lower())
                word_shift = (word_seed % 85) - 35

                f0 = (base_f0 + text_context_shift + word_shift) * contour * voice.pitch
                dur_ms = (105.0 + 34.0 * len(value)) / max(0.1, voice.speed)
                dur_ms = clamp(dur_ms, 100.0, 460.0)

                audio.extend(
                    render_word_tone(
                        word=value,
                        sr=self.sr,
                        duration_ms=dur_ms,
                        f0=f0,
                        seed=word_seed,
                    )
                )
                audio.extend(silence(self.sr, 32.0 / voice.speed))
                word_i += 1

            elif kind == "punct":
                audio.extend(
                    silence(self.sr, punctuation_pause_ms(value) / voice.speed)
                )

        return audio


def tokenize_text(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush_word() -> None:
        if buf:
            tokens.append(("word", "".join(buf)))
            buf.clear()

    for ch in text:
        if ch.isalnum() or ch == "'":
            buf.append(ch)
        else:
            flush_word()
            if ch in ":,.;!?":
                tokens.append(("punct", ch))

    flush_word()
    return tokens


def punctuation_pause_ms(ch: str) -> float:
    if ch == ":":
        return 190.0
    if ch == ",":
        return 120.0
    if ch == ";":
        return 140.0
    if ch in ".?":
        return 230.0
    if ch == "!":
        return 210.0
    return 0.0


def render_word_tone(
    *,
    word: str,
    sr: int,
    duration_ms: float,
    f0: float,
    seed: int,
) -> Audio:
    n = max(1, ms_to_samples(sr, duration_ms))
    fade_n = max(1, min(ms_to_samples(sr, 9.0), n // 2))

    phase = 0.0
    out: Audio = []

    vib_rate = 4.0 + (seed % 40) / 20.0
    syllable_rate = 2.0 + (len(word) % 4)

    for i in range(n):
        t = i / sr

        vibrato = 1.0 + 0.025 * math.sin(2.0 * math.pi * vib_rate * t + (seed % 17))
        contour = 1.0 + 0.025 * math.sin(
            2.0 * math.pi * t / max(0.001, duration_ms / 1000.0)
        )
        freq = f0 * vibrato * contour

        phase += 2.0 * math.pi * freq / sr

        sample = (
            math.sin(phase)
            + 0.35 * math.sin(2.0 * phase + 0.2)
            + 0.12 * math.sin(3.0 * phase + 0.5)
        )

        if i < fade_n:
            env = i / fade_n
        elif i >= n - fade_n:
            env = (n - i) / fade_n
        else:
            env = 1.0

        syllable = 0.78 + 0.22 * math.sin(2.0 * math.pi * syllable_rate * t)
        out.append(0.19 * sample * env * syllable)

    return out


# ---------------------------------------------------------------------------
# Cache and sink
# ---------------------------------------------------------------------------


class PrefixCache(Generic[T]):
    def __init__(self):
        self._values: dict[str, T] = {}
        self._inflight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self._values

    async def get_or_create(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if key in self._values:
                return self._values[key]

            task = self._inflight.get(key)
            if task is None:
                task = asyncio.ensure_future(factory())
                self._inflight[key] = task

        try:
            value = await task
        except BaseException:
            async with self._lock:
                if self._inflight.get(key) is task:
                    del self._inflight[key]
            raise

        async with self._lock:
            if key not in self._values:
                self._values[key] = value
            if self._inflight.get(key) is task:
                del self._inflight[key]
            return self._values[key]


def prefix_cache_key(prefix: str, voice: VoiceParams) -> str:
    raw = f"{voice.model}|{voice.voice_id}|{voice.speed}|{voice.pitch}|{prefix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BufferedWavSink:
    def __init__(self, *, path: str, sr: int):
        self.path = path
        self.sr = sr
        self.audio: Audio = []

    async def write(self, chunk: Audio) -> None:
        self.audio.extend(chunk)
        await asyncio.sleep(0)

    def save(self) -> None:
        write_wav(self.path, self.audio, self.sr)
        print(
            f"[stream] saved {self.path}, "
            f"{samples_to_ms(self.sr, len(self.audio)):.1f} ms"
        )


async def write_paced(
    *,
    sink: BufferedWavSink,
    audio: Audio,
    sr: int,
    label: str,
    chunk_ms: float,
) -> None:
    chunk_n = max(1, ms_to_samples(sr, chunk_ms))
    total_ms = samples_to_ms(sr, len(audio))

    print(f"[stream] start {label}, {total_ms:.1f} ms")

    for start in range(0, len(audio), chunk_n):
        chunk = audio[start : start + chunk_n]
        await sink.write(chunk)
        await asyncio.sleep(len(chunk) / sr)

    print(f"[stream] end {label}")


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------


def frame_db(
    audio: Audio,
    *,
    frame_size: int,
    hop: int,
) -> list[float]:
    if not audio:
        return []

    values: list[float] = []
    limit = max(1, len(audio) - frame_size + 1)

    for start in range(0, limit, hop):
        end = min(start + frame_size, len(audio))
        energy = 0.0

        for i in range(start, end):
            energy += audio[i] * audio[i]

        energy /= frame_size
        rms = math.sqrt(max(energy, 0.0))
        values.append(20.0 * math.log10(max(rms, 1e-8)))

    return values


def trim_trailing_silence_keep(
    audio: Audio,
    sr: int,
    *,
    threshold_db: float,
    keep_ms: float,
    frame_size: int,
    hop: int,
) -> Audio:
    dbs = frame_db(audio, frame_size=frame_size, hop=hop)
    voiced = [i for i, db in enumerate(dbs) if db > threshold_db]

    if not voiced:
        return audio

    last_voiced_frame = voiced[-1]
    last_voiced_sample = last_voiced_frame * hop + frame_size
    keep_samples = ms_to_samples(sr, keep_ms)
    end = min(len(audio), last_voiced_sample + keep_samples)

    return audio[:end]


def fade_in(audio: Audio, fade_samples: int) -> Audio:
    if not audio or fade_samples <= 0:
        return audio

    n = min(fade_samples, len(audio))

    for i in range(n):
        audio[i] *= i / max(1, n - 1)

    return audio


def envelope_features(
    audio: Audio,
    sr: int,
    *,
    hop_ms: float,
    win_ms: float,
) -> tuple[list[tuple[float, float]], int]:
    hop = max(1, ms_to_samples(sr, hop_ms))
    win = max(1, ms_to_samples(sr, win_ms))

    if not audio:
        return [], hop

    feats: list[tuple[float, float]] = []

    for start in range(0, len(audio), hop):
        end = min(start + win, len(audio))

        energy = 0.0
        for i in range(start, end):
            energy += audio[i] * audio[i]

        energy /= win
        rms = math.sqrt(max(energy, 0.0))
        db = 20.0 * math.log10(max(rms, 1e-8))
        db_norm = (clamp(db, -80.0, 0.0) + 80.0) / 80.0

        crossings = 0
        prev = audio[start]

        for offset in range(1, win):
            idx = start + offset
            x = audio[idx] if idx < len(audio) else 0.0

            if (prev >= 0.0 and x < 0.0) or (prev < 0.0 and x >= 0.0):
                crossings += 1

            prev = x

        zcr = crossings / max(1, win - 1)
        feats.append((db_norm, zcr))

    return feats, hop


def feature_cost(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + 0.25 * abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------


class BoundaryDetector:
    def __init__(self, *, sr: int, config: BoundaryConfig):
        self.sr = sr
        self.config = config

    def find_cut(
        self,
        *,
        cached_prefix_audio: Audio,
        full_audio: Audio,
        use_dtw: bool,
    ) -> BoundaryResult:
        expected = len(cached_prefix_audio)
        expected_method = "cached-prefix-duration"

        if use_dtw:
            try:
                expected = self._envelope_dtw_prefix_end_estimate(
                    cached_prefix_audio=cached_prefix_audio,
                    full_audio=full_audio,
                )
                expected_method = "rough-envelope-dtw"
            except Exception as exc:
                print(
                    f"[boundary] DTW failed, falling back to duration estimate: {exc}"
                )

        # The splice is deliberately searched inside/near silence instead of
        # cutting arbitrary speech samples.
        cut = self._find_quiet_boundary_near(
            audio=full_audio,
            expected_sample=expected,
        )

        if cut is not None:
            return BoundaryResult(cut, f"{expected_method}+nearby-silence")

        cut = self._snap_to_low_amplitude_point(
            audio=full_audio,
            center_sample=expected,
        )

        return BoundaryResult(cut, f"{expected_method}+low-amplitude-fallback")

    def _find_quiet_boundary_near(
        self,
        *,
        audio: Audio,
        expected_sample: int,
    ) -> int | None:
        dbs = frame_db(
            audio,
            frame_size=self.config.frame_size,
            hop=self.config.hop,
        )
        if not dbs:
            return None

        search_start_sample = max(
            0,
            expected_sample - ms_to_samples(self.sr, self.config.search_before_ms),
        )
        search_end_sample = min(
            len(audio),
            expected_sample + ms_to_samples(self.sr, self.config.search_after_ms),
        )

        start_frame = max(0, search_start_sample // self.config.hop)
        end_frame = min(
            len(dbs),
            max(start_frame, search_end_sample // self.config.hop),
        )

        if end_frame <= start_frame:
            return None

        min_quiet_frames = max(
            1,
            ms_to_samples(self.sr, self.config.min_quiet_ms) // self.config.hop,
        )
        keep_quiet_samples = ms_to_samples(self.sr, self.config.keep_quiet_ms)

        quiet = [
            db < self.config.silence_threshold_db for db in dbs[start_frame:end_frame]
        ]

        best_cut: int | None = None
        best_distance: int | None = None
        run_start: int | None = None

        def consider_run(local_start: int, local_end: int) -> None:
            nonlocal best_cut, best_distance

            if local_end - local_start < min_quiet_frames:
                return

            absolute_run_start = start_frame + local_start
            absolute_run_end = start_frame + local_end

            run_start_sample = absolute_run_start * self.config.hop
            run_end_sample = absolute_run_end * self.config.hop

            cut = max(run_start_sample, run_end_sample - keep_quiet_samples)
            cut = clamp(cut, 0, len(audio))
            distance = abs(cut - expected_sample)

            if best_distance is None or distance < best_distance:
                best_cut = cut
                best_distance = distance

        for i, is_quiet in enumerate(quiet):
            if is_quiet and run_start is None:
                run_start = i
            elif not is_quiet and run_start is not None:
                consider_run(run_start, i)
                run_start = None

        if run_start is not None:
            consider_run(run_start, len(quiet))

        return best_cut

    def _snap_to_low_amplitude_point(
        self,
        *,
        audio: Audio,
        center_sample: int,
    ) -> int:
        radius = ms_to_samples(self.sr, self.config.low_amp_window_ms)
        lo = max(0, center_sample - radius)
        hi = min(len(audio), center_sample + radius)

        if hi <= lo:
            return clamp(center_sample, 0, len(audio))

        best_i = lo
        best_abs = abs(audio[lo])

        for i in range(lo + 1, hi):
            value = abs(audio[i])
            if value < best_abs:
                best_abs = value
                best_i = i

        return best_i

    def _envelope_dtw_prefix_end_estimate(
        self,
        *,
        cached_prefix_audio: Audio,
        full_audio: Audio,
    ) -> int:
        # DTW is only a rough audio-only estimate; silence search still chooses
        # the final splice point.
        prefix_for_match = trim_trailing_silence_keep(
            cached_prefix_audio,
            self.sr,
            threshold_db=self.config.silence_threshold_db,
            keep_ms=self.config.dtw_trim_keep_ms,
            frame_size=self.config.frame_size,
            hop=self.config.hop,
        )

        max_search_len = min(
            len(full_audio),
            int(
                len(prefix_for_match) * self.config.dtw_max_search_multiplier
                + ms_to_samples(self.sr, self.config.dtw_max_search_extra_ms)
            ),
        )
        full_head = full_audio[:max_search_len]

        x_feats, hop = envelope_features(
            prefix_for_match,
            self.sr,
            hop_ms=self.config.dtw_hop_ms,
            win_ms=self.config.dtw_win_ms,
        )
        y_feats, _ = envelope_features(
            full_head,
            self.sr,
            hop_ms=self.config.dtw_hop_ms,
            win_ms=self.config.dtw_win_ms,
        )

        m = len(x_feats)
        n = len(y_feats)

        if m == 0 or n == 0:
            raise ValueError("not enough audio for DTW")

        inf = 1e18
        prev = [inf] * n

        for i in range(m):
            curr = [inf] * n

            for j in range(n):
                c = feature_cost(x_feats[i], y_feats[j])

                if i == 0 and j == 0:
                    best = 0.0
                else:
                    best = inf

                    if i > 0:
                        best = min(best, prev[j])
                    if j > 0:
                        best = min(best, curr[j - 1])
                    if i > 0 and j > 0:
                        best = min(best, prev[j - 1])

                curr[j] = c + best

            prev = curr

        min_j = min(n - 1, max(0, int(m * self.config.dtw_min_endpoint_ratio)))
        best_j = min_j
        best_score = inf

        for j in range(min_j, n):
            score = prev[j] / max(1, m + j)
            if score < best_score:
                best_score = score
                best_j = j

        return clamp(best_j * hop, 0, len(full_audio))


# ---------------------------------------------------------------------------
# Cached-prefix speaker
# ---------------------------------------------------------------------------


async def synthesize_cached_prefix_audio(
    *,
    tts: Synthesizer,
    prefix: str,
    voice: VoiceParams,
    sr: int,
    boundary_config: BoundaryConfig,
) -> Audio:
    audio = await tts.synthesize(prefix, voice)
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

    print(f"[debug] wrote debug WAVs to {debug_dir}")


class CachedPrefixSpeaker:
    def __init__(
        self,
        *,
        tts: Synthesizer,
        cache: PrefixCache[Audio],
        sink: BufferedWavSink,
        voice: VoiceParams,
        run_config: RunConfig,
        boundary_config: BoundaryConfig,
        boundary_detector: BoundaryDetector,
    ):
        self.tts = tts
        self.cache = cache
        self.sink = sink
        self.voice = voice
        self.run = run_config
        self.boundary_config = boundary_config
        self.boundary_detector = boundary_detector

    async def prewarm_prefix(self) -> None:
        await self._get_cached_prefix(self.run.prefix)

    async def speak(self, *, prefix: str, rest: str) -> None:
        full_text = prefix + rest

        synth_start = time.perf_counter()
        full_task = asyncio.create_task(self.tts.synthesize(full_text, self.voice))
        print("[synth] started full synthesis in background")

        try:
            cached_prefix_audio = await self._get_cached_prefix(prefix)
        except BaseException:
            full_task.cancel()
            raise

        await write_paced(
            sink=self.sink,
            audio=cached_prefix_audio,
            sr=self.run.sr,
            label="cached prefix",
            chunk_ms=self.run.chunk_ms,
        )

        silence_written = await self._write_silence_until_ready(full_task)

        full_audio = await full_task
        synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0
        print(f"[synth] full audio ready after {synth_elapsed_ms:.1f} ms")

        if silence_written:
            print(
                f"[stream] latency silence added, "
                f"{samples_to_ms(self.run.sr, silence_written):.1f} ms"
            )

        boundary = self.boundary_detector.find_cut(
            cached_prefix_audio=cached_prefix_audio,
            full_audio=full_audio,
            use_dtw=self.run.use_dtw,
        )

        print(
            "[boundary] cut at "
            f"{samples_to_ms(self.run.sr, boundary.cut_sample):.1f} ms "
            f"into full audio using {boundary.method}"
        )

        continuation = fade_in(
            full_audio[boundary.cut_sample :],
            fade_samples=ms_to_samples(
                self.run.sr,
                self.boundary_config.continuation_fade_in_ms,
            ),
        )

        if self.run.dump_debug:
            dump_debug_wavs(
                debug_dir=self.run.debug_dir,
                sr=self.run.sr,
                tracks={
                    "debug_cached_prefix.wav": cached_prefix_audio,
                    "debug_full_raw.wav": full_audio,
                    "debug_continuation_after_cut.wav": continuation,
                },
            )

        await write_paced(
            sink=self.sink,
            audio=continuation,
            sr=self.run.sr,
            label="generated continuation",
            chunk_ms=self.run.chunk_ms,
        )

    async def _get_cached_prefix(self, prefix: str) -> Audio:
        key = prefix_cache_key(prefix, self.voice)
        was_cached = key in self.cache

        if was_cached:
            print("[goblin cache] hit, cached prefix is ready immediately")
        else:
            print("[goblin cache] miss, synthesizing prefix once")

        audio = await self.cache.get_or_create(
            key,
            lambda: synthesize_cached_prefix_audio(
                tts=self.tts,
                prefix=prefix,
                voice=self.voice,
                sr=self.run.sr,
                boundary_config=self.boundary_config,
            ),
        )

        if not was_cached:
            print(
                f"[goblin cache] stored prefix, "
                f"{samples_to_ms(self.run.sr, len(audio)):.1f} ms"
            )

        return audio

    async def _write_silence_until_ready(self, task: asyncio.Task[Audio]) -> int:
        silence_chunk = silence(self.run.sr, self.run.latency_silence_chunk_ms)
        silence_written = 0

        # If the full synth is late, keep one continuous output stream alive
        # with silence instead of closing/reopening the stream.
        while not task.done():
            if silence_written == 0:
                print("[stream] full synth not ready, adding same-stream silence")

            await self.sink.write(silence_chunk)
            silence_written += len(silence_chunk)
            await asyncio.sleep(self.run.latency_silence_chunk_ms / 1000.0)

        return silence_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[RunConfig, VoiceParams, FakeTTSConfig, BoundaryConfig]:
    run_config = RunConfig(
        prefix=args.prefix,
        rest=args.rest,
        out=args.out,
        sr=args.sr,
        chunk_ms=args.chunk_ms,
        use_dtw=args.use_dtw,
        dump_debug=args.dump_debug,
        debug_dir=args.debug_dir,
        prewarm=args.prewarm,
    )

    voice = VoiceParams(
        model="fake-goblin-tts",
        voice_id=args.voice,
        speed=args.speed,
        pitch=args.pitch,
    )

    fake_tts_config = FakeTTSConfig(
        sr=args.sr,
        base_latency_ms=args.base_latency_ms,
        per_char_latency_ms=args.per_char_latency_ms,
    )

    boundary_config = BoundaryConfig()

    return run_config, voice, fake_tts_config, boundary_config


async def main_async(args: argparse.Namespace) -> None:
    run_config, voice, fake_tts_config, boundary_config = configs_from_args(args)

    tts = FakeTTS(fake_tts_config)
    cache: PrefixCache[Audio] = PrefixCache()
    sink = BufferedWavSink(path=run_config.out, sr=run_config.sr)
    boundary_detector = BoundaryDetector(sr=run_config.sr, config=boundary_config)

    speaker = CachedPrefixSpeaker(
        tts=tts,
        cache=cache,
        sink=sink,
        voice=voice,
        run_config=run_config,
        boundary_config=boundary_config,
        boundary_detector=boundary_detector,
    )

    if run_config.prewarm:
        print("[main] prewarming prefix cache before request")
        await speaker.prewarm_prefix()
        print("[main] prewarm done, next request can start with cached audio")

    request_start = time.perf_counter()

    await speaker.speak(prefix=run_config.prefix, rest=run_config.rest)

    elapsed_ms = (time.perf_counter() - request_start) * 1000.0
    sink.save()

    print(f"[main] request wall time, {elapsed_ms:.1f} ms")
    print("[main] goblin PoC complete")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone cached-prefix TTS splice PoC, no timestamps, no phonemes.",
    )

    p.add_argument(
        "--prefix",
        default="Someone says: ",
        help="Fixed prefix to cache. Punctuation like ':' helps create a spliceable pause.",
    )

    p.add_argument(
        "--rest",
        default="The tiny raccoon found a shiny snack in the moonlit alley.",
        help="Changing request text after the fixed prefix.",
    )

    p.add_argument(
        "--out",
        default="poc_output.wav",
        help="Output WAV path.",
    )

    p.add_argument(
        "--sr",
        type=int,
        default=24000,
        help="Sample rate.",
    )

    p.add_argument(
        "--voice",
        default="raccoon-narrator",
        help="Fake voice id.",
    )

    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Fake voice speed.",
    )

    p.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Fake voice pitch multiplier.",
    )

    p.add_argument(
        "--base-latency-ms",
        type=float,
        default=250.0,
        help="Fake TTS base latency.",
    )

    p.add_argument(
        "--per-char-latency-ms",
        type=float,
        default=8.0,
        help="Fake TTS per-character latency.",
    )

    p.add_argument(
        "--chunk-ms",
        type=float,
        default=40.0,
        help="Simulated streaming chunk size.",
    )

    p.add_argument(
        "--use-dtw",
        action="store_true",
        help="Use rough dependency-free envelope DTW before silence search.",
    )

    p.add_argument(
        "--dump-debug",
        action="store_true",
        help=(
            "Write debug_cached_prefix.wav, debug_full_raw.wav, "
            "and debug_continuation_after_cut.wav."
        ),
    )

    p.add_argument(
        "--debug-dir",
        default="poc_debug",
        help="Directory for debug WAVs.",
    )

    p.add_argument(
        "--no-prewarm",
        dest="prewarm",
        action="store_false",
        help="Do not prewarm prefix cache. First request will pay prefix synthesis cost.",
    )

    p.set_defaults(prewarm=True)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
