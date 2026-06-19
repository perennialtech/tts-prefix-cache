from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from dtw import dtw as dtw_align  # pyright: ignore[reportMissingTypeStubs]

from ._audio import Audio, envelope_features, fade_in, frame_db, ms_to_samples
from .config import BoundaryConfig, BoundaryResult


@dataclass(frozen=True)
class PrefixClip:
    playback_audio: Audio
    match_audio: Audio


@dataclass(frozen=True)
class SpliceResult:
    boundary: BoundaryResult
    continuation: Audio


def _frame_size_and_hop(
    sample_rate: int,
    config: BoundaryConfig,
) -> tuple[int, int]:
    return (
        max(1, ms_to_samples(sample_rate, config.rms_window_ms)),
        max(1, ms_to_samples(sample_rate, config.rms_hop_ms)),
    )


def find_boundary(
    *,
    prefix: PrefixClip,
    full_audio: Audio,
    sample_rate: int,
    config: BoundaryConfig,
) -> BoundaryResult:
    frame_size, hop = _frame_size_and_hop(sample_rate, config)

    expected = _dtw_prefix_end_estimate(
        prefix_audio=prefix.match_audio,
        full_audio=full_audio,
        sample_rate=sample_rate,
        config=config,
    )

    cut = _find_quiet_boundary_near(
        audio=full_audio,
        expected_sample=expected,
        sample_rate=sample_rate,
        config=config,
        frame_size=frame_size,
        hop=hop,
    )
    if cut is not None:
        return BoundaryResult(cut, "dtw+nearby-silence")

    cut = _snap_to_low_amplitude_point(
        audio=full_audio,
        center_sample=expected,
        sample_rate=sample_rate,
        config=config,
    )
    return BoundaryResult(cut, "dtw+low-amplitude-fallback")


def splice_continuation(
    *,
    prefix: PrefixClip,
    full_audio: Audio,
    sample_rate: int,
    config: BoundaryConfig,
) -> SpliceResult:
    boundary = find_boundary(
        prefix=prefix,
        full_audio=full_audio,
        sample_rate=sample_rate,
        config=config,
    )
    continuation = fade_in(
        full_audio[boundary.cut_sample :],
        fade_samples=ms_to_samples(sample_rate, config.continuation_fade_in_ms),
    )
    return SpliceResult(boundary=boundary, continuation=continuation)


def _find_quiet_boundary_near(
    *,
    audio: Audio,
    expected_sample: int,
    sample_rate: int,
    config: BoundaryConfig,
    frame_size: int,
    hop: int,
) -> int | None:
    dbs = frame_db(audio, frame_size=frame_size, hop=hop)
    if dbs.size == 0:
        return None

    search_start_sample = max(
        0,
        expected_sample - ms_to_samples(sample_rate, config.search_before_ms),
    )
    search_end_sample = min(
        len(audio),
        expected_sample + ms_to_samples(sample_rate, config.search_after_ms),
    )

    start_frame = max(0, search_start_sample // hop)
    end_frame = min(len(dbs), (search_end_sample + hop - 1) // hop)

    if end_frame <= start_frame:
        return None

    quiet_sample_count = ms_to_samples(sample_rate, config.min_quiet_ms)
    min_quiet_frames = max(1, (quiet_sample_count + hop - 1) // hop)
    keep_quiet_samples = ms_to_samples(sample_rate, config.keep_quiet_ms)

    quiet = dbs[start_frame:end_frame] < config.silence_threshold_db

    best_cut: int | None = None
    best_distance: int | None = None
    run_start: int | None = None

    def consider_run(local_start: int, local_end: int) -> None:
        nonlocal best_cut, best_distance

        if local_end - local_start < min_quiet_frames:
            return

        absolute_run_start = start_frame + local_start
        absolute_run_end = start_frame + local_end

        run_start_sample = absolute_run_start * hop
        run_end_sample = absolute_run_end * hop

        cut = max(run_start_sample, run_end_sample - keep_quiet_samples)
        cut = min(max(cut, 0), len(audio))
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
    *,
    audio: Audio,
    center_sample: int,
    sample_rate: int,
    config: BoundaryConfig,
) -> int:
    radius = ms_to_samples(sample_rate, config.low_amp_window_ms)
    lo = max(0, center_sample - radius)
    hi = min(len(audio), center_sample + radius)

    if hi <= lo:
        return min(max(center_sample, 0), len(audio))

    return lo + int(np.argmin(np.abs(audio[lo:hi])))


def _dtw_prefix_end_estimate(
    *,
    prefix_audio: Audio,
    full_audio: Audio,
    sample_rate: int,
    config: BoundaryConfig,
) -> int:
    max_search_len = min(
        len(full_audio),
        int(
            len(prefix_audio) * config.dtw_max_search_multiplier
            + ms_to_samples(sample_rate, config.dtw_max_search_extra_ms)
        ),
    )
    full_head = full_audio[:max_search_len]

    x_feats, hop = envelope_features(
        prefix_audio,
        sample_rate,
        hop_ms=config.dtw_hop_ms,
        win_ms=config.dtw_win_ms,
    )
    y_feats, _ = envelope_features(
        full_head,
        sample_rate,
        hop_ms=config.dtw_hop_ms,
        win_ms=config.dtw_win_ms,
    )

    if len(x_feats) == 0 or len(y_feats) == 0:
        raise ValueError("not enough audio for DTW boundary estimation")

    alignment: Any = dtw_align(
        x_feats,
        y_feats,
        dist_method="cityblock",
        step_pattern="asymmetric",
        open_begin=False,
        open_end=True,
        keep_internals=False,
    )

    endpoint_frame = int(np.max(np.asarray(alignment.index2, dtype=np.int64)))
    return min(max(endpoint_frame * hop, 0), len(full_audio))
