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
    return arr


def ms_to_samples(sample_rate: int, ms: float) -> int:
    return int(sample_rate * ms / 1000.0)


def samples_to_ms(sample_rate: int, samples: int) -> float:
    return samples * 1000.0 / sample_rate


def silence(sample_rate: int, ms: float) -> Audio:
    count = ms_to_samples(sample_rate, ms)
    if ms > 0:
        count = max(1, count)
    return np.zeros(count, dtype=np.float32)


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
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)

    return out


def envelope_features(
    audio: Audio,
    sample_rate: int,
    *,
    hop_ms: float,
    win_ms: float,
) -> tuple[npt.NDArray[np.float64], int]:
    hop = max(1, ms_to_samples(sample_rate, hop_ms))
    win = max(1, ms_to_samples(sample_rate, win_ms))

    samples, starts = _frame_inputs(audio, frame_size=win, hop=hop)
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.float64), hop

    windows = np.lib.stride_tricks.sliding_window_view(samples, win)
    frames = windows[starts]

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    db_norm = (np.clip(db, -80.0, 0.0) + 80.0) / 80.0

    if win > 1:
        signs = np.signbit(frames)
        crossings = signs[:, 1:] != signs[:, :-1]
        zcr = np.sum(crossings, axis=1) / (win - 1)
    else:
        zcr = np.zeros(starts.size, dtype=np.float64)

    return np.column_stack((db_norm, zcr * 0.25)), hop
