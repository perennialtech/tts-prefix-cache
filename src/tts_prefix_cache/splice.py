from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._audio import (Audio, concatenate_audio, equal_power_crossfade, fade_in,
                     frame_db, ms_to_samples, to_mono_float32,
                     trim_trailing_silence_keep)
from ._validation import require_non_negative
from .align import align_prefix_to_full
from .features import FeatureSet, extract_features

_PREFIX_PLAYBACK_KEEP_MS = 160.0
_PREFIX_MATCH_KEEP_MS = 20.0

_RMS_WINDOW_MS = 42.6666667
_RMS_HOP_MS = 10.6666667

_FEATURE_WINDOW_MS = 30.0
_FEATURE_HOP_MS = 10.0
_MEL_BINS = 32

_MAX_SEARCH_MULTIPLIER = 1.8
_MAX_SEARCH_EXTRA_MS = 600.0
_ALLOW_LEADING_PADDING_MS = 120.0

_BOUNDARY_SEARCH_BEFORE_MS = 250.0
_BOUNDARY_SEARCH_AFTER_MS = 500.0
_MIN_QUIET_MS = 50.0


@dataclass(frozen=True)
class SpliceConfig:
    silence_threshold_db: float = -43.0
    holdback_ms: float = 50.0
    crossfade_ms: float = 35.0
    continuation_fade_in_ms: float = 5.0

    def __post_init__(self) -> None:
        require_non_negative("holdback_ms", self.holdback_ms)
        require_non_negative("crossfade_ms", self.crossfade_ms)
        require_non_negative("continuation_fade_in_ms", self.continuation_fade_in_ms)


@dataclass(frozen=True)
class PreparedPrefix:
    playback_audio: Audio
    match_audio: Audio
    features: FeatureSet


@dataclass(frozen=True)
class BoundaryResult:
    cut_sample: int
    expected_sample: int
    method: str
    confidence: float
    normalized_cost: float
    duration_ratio: float


@dataclass(frozen=True)
class SpliceResult:
    boundary: BoundaryResult
    continuation: Audio


def _prepare_prefix_audio(
    audio: object,
    *,
    sample_rate: int,
    config: SpliceConfig,
) -> PreparedPrefix:
    raw = to_mono_float32(audio)
    if raw.size == 0:
        raise ValueError("prefix audio must not be empty")

    frame_size, hop = _rms_frame_size_and_hop(sample_rate)

    playback_audio = trim_trailing_silence_keep(
        raw,
        sample_rate,
        threshold_db=config.silence_threshold_db,
        keep_ms=_PREFIX_PLAYBACK_KEEP_MS,
        frame_size=frame_size,
        hop=hop,
    )
    match_audio = trim_trailing_silence_keep(
        raw,
        sample_rate,
        threshold_db=config.silence_threshold_db,
        keep_ms=_PREFIX_MATCH_KEEP_MS,
        frame_size=frame_size,
        hop=hop,
    )

    return PreparedPrefix(
        playback_audio=playback_audio,
        match_audio=match_audio,
        features=_features(match_audio, sample_rate),
    )


def _find_boundary(
    *,
    prefix: PreparedPrefix,
    full_audio: object,
    sample_rate: int,
    config: SpliceConfig,
) -> BoundaryResult:
    full = to_mono_float32(full_audio)
    if full.size == 0:
        raise ValueError("full audio must not be empty")

    allowed_leading_padding = ms_to_samples(sample_rate, _ALLOW_LEADING_PADDING_MS)
    max_search_len = min(
        full.size,
        int(
            prefix.match_audio.size * _MAX_SEARCH_MULTIPLIER
            + ms_to_samples(sample_rate, _MAX_SEARCH_EXTRA_MS)
            + allowed_leading_padding
        ),
    )
    max_search_len = max(1, max_search_len)

    full_head = full[:max_search_len]
    alignment = align_prefix_to_full(
        prefix.features,
        _features(full_head, sample_rate),
        full_sample_count=full_head.size,
    )

    if alignment.start_sample > allowed_leading_padding:
        raise ValueError(
            "prefix alignment starts after allowed leading padding "
            f"({alignment.start_sample} samples > {allowed_leading_padding} samples)"
        )

    expected = min(alignment.endpoint_sample, full.size)
    return _score_boundary(
        full=full,
        expected_sample=expected,
        sample_rate=sample_rate,
        config=config,
        normalized_cost=alignment.normalized_cost,
        alignment_confidence=alignment.confidence,
        duration_ratio=alignment.duration_ratio,
    )


def _splice_from_full_audio(
    *,
    prefix: PreparedPrefix,
    full_audio: object,
    sample_rate: int,
    config: SpliceConfig,
    held_tail: object | None = None,
) -> SpliceResult:
    full = to_mono_float32(full_audio)
    boundary = _find_boundary(
        prefix=prefix,
        full_audio=full,
        sample_rate=sample_rate,
        config=config,
    )

    cut = boundary.cut_sample
    tail = None if held_tail is None else to_mono_float32(held_tail)

    if tail is not None and tail.size:
        max_crossfade = ms_to_samples(sample_rate, config.crossfade_ms)
        crossfade_n = min(tail.size, cut, max_crossfade)

        if crossfade_n > 0:
            crossfade = equal_power_crossfade(
                tail[-crossfade_n:],
                full[cut - crossfade_n : cut],
            )
            continuation = concatenate_audio(
                (tail[:-crossfade_n], crossfade, full[cut:])
            )
        else:
            continuation = concatenate_audio(
                (
                    tail,
                    fade_in(
                        full[cut:],
                        ms_to_samples(sample_rate, config.continuation_fade_in_ms),
                    ),
                )
            )
    else:
        continuation = fade_in(
            full[cut:],
            ms_to_samples(sample_rate, config.continuation_fade_in_ms),
        )

    return SpliceResult(boundary=boundary, continuation=continuation)


def _stitch_audio(
    *,
    prefix: PreparedPrefix,
    full_audio: object,
    sample_rate: int,
    config: SpliceConfig,
) -> tuple[Audio, SpliceResult]:
    holdback = min(
        ms_to_samples(sample_rate, config.holdback_ms),
        prefix.playback_audio.size,
    )

    if holdback:
        prefix_head = prefix.playback_audio[:-holdback]
        held_tail = prefix.playback_audio[-holdback:]
    else:
        prefix_head = prefix.playback_audio
        held_tail = np.zeros(0, dtype=np.float32)

    splice = _splice_from_full_audio(
        prefix=prefix,
        full_audio=full_audio,
        sample_rate=sample_rate,
        config=config,
        held_tail=held_tail,
    )

    return concatenate_audio((prefix_head, splice.continuation)), splice


def _features(audio: Audio, sample_rate: int) -> FeatureSet:
    return extract_features(
        audio,
        sample_rate,
        hop_ms=_FEATURE_HOP_MS,
        window_ms=_FEATURE_WINDOW_MS,
        mel_bins=_MEL_BINS,
    )


def _rms_frame_size_and_hop(sample_rate: int) -> tuple[int, int]:
    return (
        max(1, ms_to_samples(sample_rate, _RMS_WINDOW_MS)),
        max(1, ms_to_samples(sample_rate, _RMS_HOP_MS)),
    )


def _score_boundary(
    *,
    full: Audio,
    expected_sample: int,
    sample_rate: int,
    config: SpliceConfig,
    normalized_cost: float,
    alignment_confidence: float,
    duration_ratio: float,
) -> BoundaryResult:
    search_before = ms_to_samples(sample_rate, _BOUNDARY_SEARCH_BEFORE_MS)
    search_after = ms_to_samples(sample_rate, _BOUNDARY_SEARCH_AFTER_MS)

    lo = max(0, expected_sample - search_before)
    hi = min(full.size, expected_sample + search_after)

    candidates = np.arange(lo, hi + 1, dtype=np.int64)

    frame_size, hop = _rms_frame_size_and_hop(sample_rate)
    dbs = frame_db(full, frame_size=frame_size, hop=hop)

    if dbs.size:
        frames = np.minimum(candidates // hop, dbs.size - 1)
        energy = (np.clip(dbs[frames], -80.0, 0.0) + 80.0) / 80.0
        quiet_frames = _quiet_frames(
            dbs,
            threshold_db=config.silence_threshold_db,
            min_frames=max(
                1, (ms_to_samples(sample_rate, _MIN_QUIET_MS) + hop - 1) // hop
            ),
        )
        quiet_bonus = quiet_frames[frames].astype(np.float64)
    else:
        energy = np.zeros(candidates.size, dtype=np.float64)
        quiet_bonus = np.zeros(candidates.size, dtype=np.float64)

    amplitude = np.zeros(candidates.size, dtype=np.float64)
    valid = candidates < full.size
    amplitude[valid] = np.abs(full[candidates[valid]])

    radius = max(search_before, search_after, 1)
    distance = np.abs(candidates - expected_sample) / radius

    score = (
        0.50 * distance
        + 0.40 * energy
        + 0.10 * np.minimum(amplitude, 1.0)
        - 0.20 * quiet_bonus
    )

    best = int(np.argmin(score))
    cut = int(candidates[best])

    distance_penalty = 1.0 - 0.5 * min(float(distance[best]), 1.0)
    confidence = min(max(alignment_confidence * distance_penalty, 0.0), 1.0)

    method = "dtw+quiet-boundary" if bool(quiet_bonus[best]) else "dtw+energy-boundary"

    return BoundaryResult(
        cut_sample=cut,
        expected_sample=expected_sample,
        method=method,
        confidence=confidence,
        normalized_cost=normalized_cost,
        duration_ratio=duration_ratio,
    )


def _quiet_frames(
    dbs: np.ndarray,
    *,
    threshold_db: float,
    min_frames: int,
) -> np.ndarray:
    quiet = dbs < threshold_db
    ok = np.zeros(quiet.size, dtype=bool)

    run_start: int | None = None

    for i, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = i
        elif not is_quiet and run_start is not None:
            if i - run_start >= min_frames:
                ok[run_start:i] = True
            run_start = None

    if run_start is not None and quiet.size - run_start >= min_frames:
        ok[run_start:] = True

    return ok
