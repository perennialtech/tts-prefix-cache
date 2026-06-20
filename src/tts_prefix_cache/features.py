from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ._audio import Audio, _frame_inputs, ms_to_samples, to_mono_float32


@dataclass(frozen=True)
class FeatureSet:
    data: npt.NDArray[np.float64]
    hop_samples: int
    window_samples: int
    sample_count: int


def extract_features(
    audio: Audio,
    sample_rate: int,
    *,
    hop_ms: float,
    window_ms: float,
    mel_bins: int,
) -> FeatureSet:
    samples = to_mono_float32(audio)

    return _log_mel_features(
        samples,
        sample_rate,
        hop_ms=hop_ms,
        window_ms=window_ms,
        mel_bins=mel_bins,
    )


def _log_mel_features(
    audio: Audio,
    sample_rate: int,
    *,
    hop_ms: float,
    window_ms: float,
    mel_bins: int,
) -> FeatureSet:
    hop = max(1, ms_to_samples(sample_rate, hop_ms))
    win = max(1, ms_to_samples(sample_rate, window_ms))

    samples, starts = _frame_inputs(audio, frame_size=win, hop=hop)
    if starts.size == 0:
        return FeatureSet(np.empty((0, 0), dtype=np.float64), hop, win, audio.size)

    samples = samples - np.mean(samples)
    windows = np.lib.stride_tricks.sliding_window_view(samples, win)
    frames = windows[starts]

    window = np.hanning(win).astype(np.float64, copy=False)
    windowed = frames * window

    spectrum = np.fft.rfft(windowed, axis=1)
    power = (np.abs(spectrum) ** 2) / win

    filters = _mel_filterbank(sample_rate, win, mel_bins)
    mel = power @ filters.T
    log_mel = np.log(np.maximum(mel, 1e-10))

    energy = np.log(np.maximum(np.mean(frames * frames, axis=1, keepdims=True), 1e-10))
    base = np.concatenate((log_mel, energy), axis=1)
    features = np.concatenate((base, _delta(base) * 0.5), axis=1)

    return FeatureSet(_standardize(features), hop, win, audio.size)


def _standardize(features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if features.size == 0:
        return features.copy()

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std < 1e-8] = 1.0

    return (features - mean) / std


def _delta(features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if features.shape[0] < 2:
        return np.zeros_like(features)

    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return (padded[2:] - padded[:-2]) * 0.5


@lru_cache(maxsize=64)
def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    mel_bins: int,
) -> npt.NDArray[np.float64]:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filters = np.zeros((mel_bins, freqs.size), dtype=np.float64)

    f_min = 20.0 if sample_rate > 40 else 0.0
    f_max = sample_rate / 2.0

    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), mel_bins + 2)
    hz_points = _mel_to_hz(mel_points)

    for i in range(mel_bins):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        lower = (freqs - left) / max(center - left, 1e-12)
        upper = (right - freqs) / max(right - center, 1e-12)
        filters[i] = np.maximum(0.0, np.minimum(lower, upper))

    sums = filters.sum(axis=1, keepdims=True)
    np.divide(filters, sums, out=filters, where=sums > 0.0)

    return filters


def _hz_to_mel(hz: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
