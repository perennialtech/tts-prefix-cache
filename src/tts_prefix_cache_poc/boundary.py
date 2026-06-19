from __future__ import annotations

import numpy as np
from dtw import dtw as dtw_align

from .audio import (
    Audio,
    as_audio_array,
    envelope_features,
    frame_db,
    ms_to_samples,
    trim_trailing_silence_keep,
)
from .config import BoundaryConfig, BoundaryResult


class BoundaryDetector:
    def __init__(self, *, sr: int, config: BoundaryConfig):
        self.sr = sr
        self.config = config

    def find_cut(
        self,
        *,
        cached_prefix_audio: Audio,
        full_audio: Audio,
    ) -> BoundaryResult:
        expected = self._dtw_prefix_end_estimate(
            cached_prefix_audio=cached_prefix_audio,
            full_audio=full_audio,
        )

        cut = self._find_quiet_boundary_near(
            audio=full_audio,
            expected_sample=expected,
        )

        if cut is not None:
            return BoundaryResult(cut, "dtw+nearby-silence")

        cut = self._snap_to_low_amplitude_point(
            audio=full_audio,
            center_sample=expected,
        )

        return BoundaryResult(cut, "dtw+low-amplitude-fallback")

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
        if dbs.size == 0:
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

        quiet = dbs[start_frame:end_frame] < self.config.silence_threshold_db

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
        self,
        *,
        audio: Audio,
        center_sample: int,
    ) -> int:
        samples = as_audio_array(audio)
        radius = ms_to_samples(self.sr, self.config.low_amp_window_ms)
        lo = max(0, center_sample - radius)
        hi = min(len(samples), center_sample + radius)

        if hi <= lo:
            return min(max(center_sample, 0), len(samples))

        return lo + int(np.argmin(np.abs(samples[lo:hi])))

    def _dtw_prefix_end_estimate(
        self,
        *,
        cached_prefix_audio: Audio,
        full_audio: Audio,
    ) -> int:
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

        if len(x_feats) == 0 or len(y_feats) == 0:
            raise ValueError("not enough audio for DTW boundary estimation")

        alignment = dtw_align(
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
