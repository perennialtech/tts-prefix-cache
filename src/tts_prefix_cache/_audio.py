from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]


def to_mono_float32(audio: object) -> Audio:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("audio must be a mono 1D array")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("audio must contain only finite samples")
    return arr.astype(np.float32, copy=False)


def ms_to_samples(sample_rate: int, ms: float) -> int:
    return int(sample_rate * ms / 1000.0)


def samples_to_ms(sample_rate: int, samples: int) -> float:
    return samples * 1000.0 / sample_rate


def silence(sample_rate: int, ms: float) -> Audio:
    if ms <= 0:
        return np.zeros(0, dtype=np.float32)

    return np.zeros(max(1, ms_to_samples(sample_rate, ms)), dtype=np.float32)


def concatenate_audio(parts: list[object] | tuple[object, ...]) -> Audio:
    chunks = [to_mono_float32(part) for part in parts]
    chunks = [chunk for chunk in chunks if chunk.size]

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32, copy=False)


def write_wav(path: str | os.PathLike[str], audio: object, sample_rate: int) -> None:
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.clip(to_mono_float32(audio), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2", copy=False)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(struct.calcsize("<h"))
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _frame_inputs(
    audio: Audio,
    *,
    frame_size: int,
    hop: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    samples = audio.astype(np.float64, copy=False)
    if samples.size == 0:
        return samples, np.empty(0, dtype=np.int64)

    frame_count = max(1, (samples.size + hop - 1) // hop)
    padded_len = (frame_count - 1) * hop + frame_size

    if samples.size < padded_len:
        samples = np.pad(samples, (0, padded_len - samples.size))

    starts = np.arange(frame_count, dtype=np.int64) * hop
    return samples, starts


def frame_db(
    audio: Audio,
    *,
    frame_size: int,
    hop: int,
) -> npt.NDArray[np.float64]:
    samples, starts = _frame_inputs(audio, frame_size=frame_size, hop=hop)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)

    squared = samples * samples

    prefix = np.empty(squared.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])

    energy = (prefix[starts + frame_size] - prefix[starts]) / frame_size
    rms = np.sqrt(np.maximum(energy, 0.0))

    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def trim_trailing_silence_keep(
    audio: Audio,
    sample_rate: int,
    *,
    threshold_db: float,
    keep_ms: float,
    frame_size: int,
    hop: int,
) -> Audio:
    dbs = frame_db(audio, frame_size=frame_size, hop=hop)
    voiced = np.flatnonzero(dbs > threshold_db)

    if voiced.size == 0:
        return audio.copy()

    last_voiced_sample = int(voiced[-1]) * hop + frame_size
    keep_samples = ms_to_samples(sample_rate, keep_ms)
    end = min(audio.size, last_voiced_sample + keep_samples)

    return audio[:end].copy()


def fade_in(audio: Audio, fade_samples: int) -> Audio:
    out = audio.copy()
    if out.size == 0 or fade_samples <= 0:
        return out

    n = min(fade_samples, out.size)
    if n == 1:
        out[0] *= 1.0
        return out

    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def fade_out(audio: Audio, fade_samples: int) -> Audio:
    out = audio.copy()
    if out.size == 0 or fade_samples <= 0:
        return out

    n = min(fade_samples, out.size)
    if n == 1:
        out[0] *= 1.0
        return out

    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def equal_power_crossfade(left: object, right: object) -> Audio:
    left_samples = to_mono_float32(left)
    right_samples = to_mono_float32(right)
    n = min(left_samples.size, right_samples.size)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    theta = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    faded = left_samples[-n:] * np.cos(theta) + right_samples[:n] * np.sin(theta)

    return faded.astype(np.float32, copy=False)
