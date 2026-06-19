from __future__ import annotations

import os
import struct
import wave
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]


def as_audio_array(audio: Audio | Sequence[float]) -> Audio:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def ms_to_samples(sr: int, ms: float) -> int:
    return int(sr * ms / 1000.0)


def samples_to_ms(sr: int, samples: int) -> float:
    return samples * 1000.0 / sr


def silence(sr: int, ms: float) -> Audio:
    return np.zeros(ms_to_samples(sr, ms), dtype=np.float32)


def write_wav(path: str, audio: Audio | Sequence[float], sr: int) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    samples = np.clip(as_audio_array(audio), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2", copy=False)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(struct.calcsize("<h"))
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def frame_db(
    audio: Audio | Sequence[float],
    *,
    frame_size: int,
    hop: int,
) -> npt.NDArray[np.float64]:
    samples = as_audio_array(audio).astype(np.float64, copy=False)
    if samples.size == 0:
        return np.empty(0, dtype=np.float64)

    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))

    starts = np.arange(0, samples.size - frame_size + 1, hop, dtype=np.int64)
    squared = samples * samples

    prefix = np.empty(squared.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(squared, out=prefix[1:])

    energy = (prefix[starts + frame_size] - prefix[starts]) / frame_size
    rms = np.sqrt(np.maximum(energy, 0.0))

    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def trim_trailing_silence_keep(
    audio: Audio | Sequence[float],
    sr: int,
    *,
    threshold_db: float,
    keep_ms: float,
    frame_size: int,
    hop: int,
) -> Audio:
    samples = as_audio_array(audio)
    dbs = frame_db(samples, frame_size=frame_size, hop=hop)
    voiced = np.flatnonzero(dbs > threshold_db)

    if voiced.size == 0:
        return samples.copy()

    last_voiced_sample = int(voiced[-1]) * hop + frame_size
    keep_samples = ms_to_samples(sr, keep_ms)
    end = min(samples.size, last_voiced_sample + keep_samples)

    return samples[:end].copy()


def fade_in(audio: Audio | Sequence[float], fade_samples: int) -> Audio:
    out = as_audio_array(audio).copy()
    if out.size == 0 or fade_samples <= 0:
        return out

    n = min(fade_samples, out.size)
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)

    return out


def envelope_features(
    audio: Audio | Sequence[float],
    sr: int,
    *,
    hop_ms: float,
    win_ms: float,
) -> tuple[npt.NDArray[np.float64], int]:
    hop = max(1, ms_to_samples(sr, hop_ms))
    win = max(1, ms_to_samples(sr, win_ms))

    samples = as_audio_array(audio).astype(np.float64, copy=False)
    if samples.size == 0:
        return np.empty((0, 2), dtype=np.float64), hop

    frame_count = max(1, (samples.size + hop - 1) // hop)
    padded_len = (frame_count - 1) * hop + win

    if samples.size < padded_len:
        samples = np.pad(samples, (0, padded_len - samples.size))

    frames = np.lib.stride_tricks.sliding_window_view(samples, win)[::hop][:frame_count]

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    db_norm = (np.clip(db, -80.0, 0.0) + 80.0) / 80.0

    if win > 1:
        signs = np.signbit(frames)
        crossings = signs[:, 1:] != signs[:, :-1]
        zcr = np.sum(crossings, axis=1) / (win - 1)
    else:
        zcr = np.zeros(frame_count, dtype=np.float64)

    return np.column_stack((db_norm, zcr * 0.25)), hop
