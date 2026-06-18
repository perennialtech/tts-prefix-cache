#!/usr/bin/env python3
"""
tts_prefix_cache_poc.py

Standalone goblin proof-of-concept for:

- fixed TTS prefix cache
- immediate cached-prefix playback
- full TTS synthesis in the background with the prefix included
- no word timestamps
- no char timestamps
- no phoneme/viseme alignment
- audio-only splice by detecting silence near the expected prefix boundary
- optional rough envelope-DTW estimate before silence search
- one continuous output stream written to a WAV file

This uses a fake TTS backend so the file is fully standalone.
Replace FakeTTS.synthesize() with your real provider call when integrating.

Run:
    python tts_prefix_cache_poc.py

Try:
    python tts_prefix_cache_poc.py --use-dtw
    python tts_prefix_cache_poc.py --prefix "New message: " --rest "The tiny raccoon stole the goblin sandwich."
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

Audio = list[float]


# ---------------------------------------------------------------------------
# Goblin basics
# ---------------------------------------------------------------------------


def ms_to_samples(sr: int, ms: float) -> int:
    return int(sr * ms / 1000.0)


def samples_to_ms(sr: int, samples: int) -> float:
    return samples * 1000.0 / sr


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def silence(sr: int, ms: float) -> Audio:
    return [0.0] * ms_to_samples(sr, ms)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_wav(path: str, audio: Audio, sr: int) -> None:
    ensure_parent_dir(path)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)

        frames = bytearray()
        for x in audio:
            x = clamp(x, -1.0, 1.0)
            frames.extend(struct.pack("<h", int(x * 32767.0)))

        wf.writeframes(bytes(frames))


# ---------------------------------------------------------------------------
# Voice and fake TTS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceParams:
    model: str = "fake-goblin-tts"
    voice_id: str = "raccoon-narrator"
    speed: float = 1.0
    pitch: float = 1.0


class FakeTTS:
    """
    Fake standalone TTS.

    It deliberately:
    - has artificial latency
    - creates word-ish tones
    - creates punctuation pauses, especially after ":"
    - changes pitch contour from full text context

    The splice logic below is the real goblin thing.
    This fake renderer is just here so the PoC runs without external services.
    """

    def __init__(
        self,
        *,
        sr: int,
        base_latency_ms: float = 250.0,
        per_char_latency_ms: float = 8.0,
    ):
        self.sr = sr
        self.base_latency_ms = base_latency_ms
        self.per_char_latency_ms = per_char_latency_ms

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

                # Context-dependent contour, this mimics why full-text synthesis
                # may sound better after the prefix than rest-only synthesis.
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

                # Tiny inter-word gap, the goblin breathing room.
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
    if ch == ".":
        return 230.0
    if ch == "?":
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

        # Soft fake voice-ish harmonics.
        sample = (
            math.sin(phase)
            + 0.35 * math.sin(2.0 * phase + 0.2)
            + 0.12 * math.sin(3.0 * phase + 0.5)
        )

        # Word envelope to avoid clicks.
        if i < fade_n:
            env = i / fade_n
        elif i > n - fade_n:
            env = (n - i) / fade_n
        else:
            env = 1.0

        # Little amplitude wobble to make it less laser-beepy.
        syllable = 0.78 + 0.22 * math.sin(2.0 * math.pi * syllable_rate * t)

        out.append(0.19 * sample * env * syllable)

    return out


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------


class PrefixCache:
    def __init__(self):
        self._cache: dict[str, Audio] = {}

    def key(self, prefix: str, voice: VoiceParams) -> str:
        raw = f"{voice.model}|{voice.voice_id}|{voice.speed}|{voice.pitch}|{prefix}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(
        self,
        *,
        tts: FakeTTS,
        prefix: str,
        voice: VoiceParams,
        sr: int,
    ) -> Audio:
        key = self.key(prefix, voice)

        if key in self._cache:
            print("[goblin cache] hit, cached prefix is ready immediately")
            return self._cache[key]

        print("[goblin cache] miss, synthesizing prefix once")
        audio = await tts.synthesize(prefix, voice)

        # Normalize the cached prefix ending.
        # Keep a small controlled pause where the future splice can hide.
        audio = trim_trailing_silence_keep(
            audio,
            sr,
            threshold_db=-43.0,
            keep_ms=160.0,
        )

        self._cache[key] = audio
        print(f"[goblin cache] stored prefix, {samples_to_ms(sr, len(audio)):.1f} ms")
        return audio


# ---------------------------------------------------------------------------
# Output stream
# ---------------------------------------------------------------------------


class WavOutputStream:
    """
    A same-stream sink.

    For a real app this would be:
    - WebRTC track
    - websocket audio stream
    - HTTP chunked audio
    - local sound device
    - RTP sender

    For this standalone goblin PoC, it accumulates one continuous WAV.
    """

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
            f"[stream] saved {self.path}, {samples_to_ms(self.sr, len(self.audio)):.1f} ms"
        )


async def play_audio_realtime(
    *,
    stream: WavOutputStream,
    audio: Audio,
    sr: int,
    label: str,
    chunk_ms: float = 40.0,
) -> None:
    chunk_n = max(1, ms_to_samples(sr, chunk_ms))
    total_ms = samples_to_ms(sr, len(audio))

    print(f"[stream] start {label}, {total_ms:.1f} ms")

    for start in range(0, len(audio), chunk_n):
        chunk = audio[start : start + chunk_n]
        await stream.write(chunk)
        await asyncio.sleep(len(chunk) / sr)

    print(f"[stream] end {label}")


# ---------------------------------------------------------------------------
# Audio analysis, silence detection
# ---------------------------------------------------------------------------


def frame_db(
    audio: Audio,
    *,
    frame_size: int = 1024,
    hop: int = 256,
) -> list[float]:
    if not audio:
        return []

    padded = audio[:]
    if len(padded) < frame_size:
        padded.extend([0.0] * (frame_size - len(padded)))

    values: list[float] = []

    for start in range(0, len(padded) - frame_size + 1, hop):
        frame = padded[start : start + frame_size]
        energy = sum(x * x for x in frame) / frame_size
        rms = math.sqrt(max(energy, 0.0))
        db = 20.0 * math.log10(max(rms, 1e-8))
        values.append(db)

    return values


def trim_trailing_silence_keep(
    audio: Audio,
    sr: int,
    *,
    threshold_db: float = -45.0,
    keep_ms: float = 160.0,
    frame_size: int = 1024,
    hop: int = 256,
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


def find_quiet_boundary_near(
    audio: Audio,
    sr: int,
    expected_sample: int,
    *,
    search_before_ms: float = 350.0,
    search_after_ms: float = 900.0,
    min_quiet_ms: float = 70.0,
    keep_quiet_ms: float = 25.0,
    threshold_db: float = -43.0,
    frame_size: int = 1024,
    hop: int = 256,
) -> int | None:
    """
    Find a quiet run near expected_sample.

    The goblin trick is to force or encourage a pause after the prefix,
    then splice inside that pause instead of cutting speech.
    """
    dbs = frame_db(audio, frame_size=frame_size, hop=hop)
    if not dbs:
        return None

    search_start_sample = max(
        0,
        expected_sample - ms_to_samples(sr, search_before_ms),
    )
    search_end_sample = min(
        len(audio),
        expected_sample + ms_to_samples(sr, search_after_ms),
    )

    start_frame = max(0, search_start_sample // hop)
    end_frame = min(len(dbs), max(start_frame, search_end_sample // hop))

    if end_frame <= start_frame:
        return None

    min_quiet_frames = max(1, ms_to_samples(sr, min_quiet_ms) // hop)
    keep_quiet_samples = ms_to_samples(sr, keep_quiet_ms)

    quiet = [db < threshold_db for db in dbs[start_frame:end_frame]]

    best_cut = None
    best_distance = None
    run_start = None

    def consider_run(local_start: int, local_end: int) -> None:
        nonlocal best_cut, best_distance

        run_len = local_end - local_start
        if run_len < min_quiet_frames:
            return

        absolute_run_start = start_frame + local_start
        absolute_run_end = start_frame + local_end

        run_start_sample = absolute_run_start * hop
        run_end_sample = absolute_run_end * hop

        # Cut near the end of the silence, leaving a tiny bit of quiet
        # before the continuation starts.
        cut = max(run_start_sample, run_end_sample - keep_quiet_samples)
        cut = clamp_int(cut, 0, len(audio))

        distance = abs(cut - expected_sample)

        if best_cut is None or distance < best_distance:
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


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def snap_to_low_amplitude_point(
    audio: Audio,
    sr: int,
    center_sample: int,
    *,
    window_ms: float = 80.0,
) -> int:
    """
    Fallback goblin method.

    This is not as good as cutting in silence, but it avoids blindly cutting
    at an arbitrary sample.
    """
    radius = ms_to_samples(sr, window_ms)

    lo = max(0, center_sample - radius)
    hi = min(len(audio), center_sample + radius)

    if hi <= lo:
        return clamp_int(center_sample, 0, len(audio))

    best_i = lo
    best_abs = abs(audio[lo])

    for i in range(lo + 1, hi):
        value = abs(audio[i])
        if value < best_abs:
            best_abs = value
            best_i = i

    return best_i


def fade_in(audio: Audio, fade_samples: int) -> Audio:
    if not audio or fade_samples <= 0:
        return audio

    out = audio[:]
    n = min(fade_samples, len(out))

    for i in range(n):
        out[i] *= i / max(1, n - 1)

    return out


def crossfade(left: Audio, right: Audio, fade_samples: int) -> Audio:
    """
    Included for completeness.

    This PoC mostly hides the seam in silence and fade-ins the continuation.
    In a lower-latency setup where the full synth is ready before the cached
    prefix ends, you can hold back a tail and crossfade instead.
    """
    if fade_samples <= 0 or not left or not right:
        return left + right

    n = min(fade_samples, len(left), len(right))

    out = left[:-n]

    for i in range(n):
        a = 1.0 - i / max(1, n - 1)
        b = i / max(1, n - 1)
        out.append(left[len(left) - n + i] * a + right[i] * b)

    out.extend(right[n:])
    return out


# ---------------------------------------------------------------------------
# Optional rough audio-only DTW
# ---------------------------------------------------------------------------


def envelope_features(
    audio: Audio,
    sr: int,
    *,
    hop_ms: float = 10.0,
    win_ms: float = 30.0,
) -> tuple[list[tuple[float, float]], int]:
    """
    Very rough features:
    - clipped dB energy
    - zero crossing rate

    This is not real MFCC alignment, but it is dependency-free and useful as
    a standalone goblin approximation.
    """
    hop = max(1, ms_to_samples(sr, hop_ms))
    win = max(1, ms_to_samples(sr, win_ms))

    feats: list[tuple[float, float]] = []

    if not audio:
        return feats, hop

    for start in range(0, len(audio), hop):
        frame = audio[start : start + win]
        if len(frame) < win:
            frame = frame + [0.0] * (win - len(frame))

        energy = sum(x * x for x in frame) / win
        rms = math.sqrt(max(energy, 0.0))
        db = 20.0 * math.log10(max(rms, 1e-8))

        # Clip to [-80, 0], then map to [0, 1].
        db_norm = (clamp(db, -80.0, 0.0) + 80.0) / 80.0

        crossings = 0
        prev = frame[0]
        for x in frame[1:]:
            if (prev >= 0.0 and x < 0.0) or (prev < 0.0 and x >= 0.0):
                crossings += 1
            prev = x

        zcr = crossings / max(1, win - 1)
        feats.append((db_norm, zcr))

    return feats, hop


def feature_cost(a: tuple[float, float], b: tuple[float, float]) -> float:
    energy_cost = abs(a[0] - b[0])
    zcr_cost = abs(a[1] - b[1])
    return energy_cost + 0.25 * zcr_cost


def envelope_dtw_prefix_end_estimate(
    *,
    cached_prefix_audio: Audio,
    full_audio: Audio,
    sr: int,
) -> int:
    """
    Dependency-free rough DTW.

    Estimate where the cached prefix ends inside the beginning of full_audio.
    This still uses no provider timestamps and no phoneme/viseme data.

    Then the caller should search for nearby silence, because silence is where
    the goblin splice wants to live.
    """
    prefix_for_match = trim_trailing_silence_keep(
        cached_prefix_audio,
        sr,
        threshold_db=-43.0,
        keep_ms=20.0,
    )

    max_search_len = min(
        len(full_audio),
        int(len(prefix_for_match) * 1.8 + sr * 0.6),
    )
    full_head = full_audio[:max_search_len]

    x_feats, hop = envelope_features(prefix_for_match, sr)
    y_feats, _ = envelope_features(full_head, sr)

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

    min_j = max(0, int(m * 0.55))
    max_j = n - 1

    best_j = min_j
    best_score = inf

    for j in range(min_j, max_j + 1):
        # Normalize a little so longer endpoints are not automatically worse.
        score = prev[j] / max(1, m + j)
        if score < best_score:
            best_score = score
            best_j = j

    return clamp_int(best_j * hop, 0, len(full_audio))


# ---------------------------------------------------------------------------
# Boundary estimation
# ---------------------------------------------------------------------------


def estimate_cut_without_metadata(
    *,
    cached_prefix_audio: Audio,
    full_audio: Audio,
    sr: int,
    use_dtw: bool,
) -> tuple[int, str]:
    """
    Returns:
        cut_sample, method_name

    This is the core no-timestamps goblin workaround.
    """
    expected = len(cached_prefix_audio)
    expected_method = "cached-prefix-duration"

    if use_dtw:
        try:
            expected = envelope_dtw_prefix_end_estimate(
                cached_prefix_audio=cached_prefix_audio,
                full_audio=full_audio,
                sr=sr,
            )
            expected_method = "rough-envelope-dtw"
        except Exception as exc:
            print(f"[boundary] DTW failed, falling back to duration estimate: {exc}")

    cut = find_quiet_boundary_near(
        full_audio,
        sr,
        expected_sample=expected,
        search_before_ms=350.0,
        search_after_ms=900.0,
        min_quiet_ms=70.0,
        keep_quiet_ms=25.0,
        threshold_db=-43.0,
    )

    if cut is not None:
        return cut, f"{expected_method}+nearby-silence"

    cut = snap_to_low_amplitude_point(
        full_audio,
        sr,
        center_sample=expected,
        window_ms=80.0,
    )

    return cut, f"{expected_method}+low-amplitude-fallback"


# ---------------------------------------------------------------------------
# The actual cached-prefix TTS flow
# ---------------------------------------------------------------------------


async def speak_with_cached_prefix_no_metadata(
    *,
    tts: FakeTTS,
    cache: PrefixCache,
    stream: WavOutputStream,
    prefix: str,
    rest: str,
    voice: VoiceParams,
    sr: int,
    use_dtw: bool,
    chunk_ms: float,
    dump_debug: bool,
    debug_dir: str,
) -> None:
    full_text = prefix + rest

    # 1. Get cached prefix audio.
    cached_prefix_audio = await cache.get(
        tts=tts,
        prefix=prefix,
        voice=voice,
        sr=sr,
    )

    # 2. Start full synthesis in background.
    # Important goblin detail: include prefix in the full text for better prosody.
    synth_start = time.perf_counter()
    full_task = asyncio.create_task(tts.synthesize(full_text, voice))
    print("[synth] started full synthesis in background")

    # 3. Play cached prefix immediately on the same stream.
    await play_audio_realtime(
        stream=stream,
        audio=cached_prefix_audio,
        sr=sr,
        label="cached prefix",
        chunk_ms=chunk_ms,
    )

    # 4. If the full synth is not ready yet, keep the same stream alive with silence.
    # This is the goblin tradeoff: instant prefix, possible pause if backend is slower.
    silence_chunk_ms = 30.0
    silence_chunk = silence(sr, silence_chunk_ms)
    silence_written = 0

    while not full_task.done():
        if silence_written == 0:
            print("[stream] full synth not ready, adding same-stream silence")
        await stream.write(silence_chunk)
        silence_written += len(silence_chunk)
        await asyncio.sleep(silence_chunk_ms / 1000.0)

    full_audio = await full_task
    synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0
    print(f"[synth] full audio ready after {synth_elapsed_ms:.1f} ms")

    if silence_written:
        print(
            f"[stream] latency silence added, {samples_to_ms(sr, silence_written):.1f} ms"
        )

    # 5. Estimate where the generated prefix ends without metadata.
    cut, method = estimate_cut_without_metadata(
        cached_prefix_audio=cached_prefix_audio,
        full_audio=full_audio,
        sr=sr,
        use_dtw=use_dtw,
    )

    print(
        "[boundary] cut at "
        f"{samples_to_ms(sr, cut):.1f} ms into full audio "
        f"using {method}"
    )

    # 6. Skip generated prefix, play generated continuation.
    continuation = full_audio[cut:]

    # Tiny fade-in helps avoid clicks if the cut is not perfectly silent.
    continuation = fade_in(
        continuation,
        fade_samples=ms_to_samples(sr, 20.0),
    )

    if dump_debug:
        os.makedirs(debug_dir, exist_ok=True)
        write_wav(
            os.path.join(debug_dir, "debug_cached_prefix.wav"), cached_prefix_audio, sr
        )
        write_wav(os.path.join(debug_dir, "debug_full_raw.wav"), full_audio, sr)
        write_wav(
            os.path.join(debug_dir, "debug_continuation_after_cut.wav"),
            continuation,
            sr,
        )
        print(f"[debug] wrote debug WAVs to {debug_dir}")

    await play_audio_realtime(
        stream=stream,
        audio=continuation,
        sr=sr,
        label="generated continuation",
        chunk_ms=chunk_ms,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    sr = args.sr

    voice = VoiceParams(
        model="fake-goblin-tts",
        voice_id=args.voice,
        speed=args.speed,
        pitch=args.pitch,
    )

    tts = FakeTTS(
        sr=sr,
        base_latency_ms=args.base_latency_ms,
        per_char_latency_ms=args.per_char_latency_ms,
    )

    cache = PrefixCache()

    if args.prewarm:
        print("[main] prewarming prefix cache before request")
        await cache.get(
            tts=tts,
            prefix=args.prefix,
            voice=voice,
            sr=sr,
        )
        print("[main] prewarm done, next request can start with cached audio")

    stream = WavOutputStream(path=args.out, sr=sr)

    request_start = time.perf_counter()

    await speak_with_cached_prefix_no_metadata(
        tts=tts,
        cache=cache,
        stream=stream,
        prefix=args.prefix,
        rest=args.rest,
        voice=voice,
        sr=sr,
        use_dtw=args.use_dtw,
        chunk_ms=args.chunk_ms,
        dump_debug=args.dump_debug,
        debug_dir=args.debug_dir,
    )

    elapsed_ms = (time.perf_counter() - request_start) * 1000.0
    stream.save()

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
        help="Write debug_cached_prefix.wav, debug_full_raw.wav, and debug_continuation_after_cut.wav.",
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
