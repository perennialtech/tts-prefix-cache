from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from dtw import dtw as dtw_align  # pyright: ignore[reportMissingTypeStubs]

from .features import FeatureSet


@dataclass(frozen=True)
class AlignmentResult:
    start_sample: int
    endpoint_sample: int
    start_frame: int
    endpoint_frame: int
    normalized_cost: float
    confidence: float
    duration_ratio: float


def align_prefix_to_full(
    prefix: FeatureSet,
    full: FeatureSet,
    *,
    full_sample_count: int,
) -> AlignmentResult:
    if prefix.data.shape[0] == 0 or full.data.shape[0] == 0:
        raise ValueError("not enough audio for DTW alignment")

    alignment: Any = dtw_align(
        prefix.data,
        full.data,
        dist_method="cityblock",
        step_pattern="asymmetric",
        open_begin=True,
        open_end=True,
        keep_internals=False,
    )

    index1 = np.asarray(alignment.index1, dtype=np.int64)
    index2 = np.asarray(alignment.index2, dtype=np.int64)

    if index1.size == 0 or index2.size == 0:
        raise ValueError("DTW returned an empty alignment path")

    start_matches = np.flatnonzero(index1 == 0)
    start_frame = (
        int(index2[start_matches[0]]) if start_matches.size else int(index2[0])
    )

    last_prefix_frame = prefix.data.shape[0] - 1
    endpoint_matches = np.flatnonzero(index1 == last_prefix_frame)

    if endpoint_matches.size:
        endpoint_frame = int(index2[endpoint_matches[-1]])
    else:
        last_seen_prefix_frame = int(np.max(index1))
        matches = np.flatnonzero(index1 == last_seen_prefix_frame)
        endpoint_frame = int(index2[matches[-1]]) if matches.size else int(index2[-1])

    start_sample = min(max(start_frame * full.hop_samples, 0), full_sample_count)
    endpoint_sample = min(
        max(endpoint_frame * full.hop_samples + full.window_samples, 0),
        full_sample_count,
    )

    diff = np.abs(prefix.data[index1] - full.data[index2])
    normalized_cost = float(
        np.mean(np.sum(diff, axis=1)) / max(prefix.data.shape[1], 1)
    )

    matched_samples = max(1, endpoint_sample - start_sample)
    duration_ratio = matched_samples / max(prefix.sample_count, 1)

    if not math.isfinite(normalized_cost):
        confidence = 0.0
    else:
        ratio_penalty = abs(math.log(max(duration_ratio, 1e-6)))
        confidence = math.exp(-max(normalized_cost, 0.0)) * math.exp(
            -2.0 * ratio_penalty
        )

    confidence = min(max(confidence, 0.0), 1.0)

    return AlignmentResult(
        start_sample=start_sample,
        endpoint_sample=endpoint_sample,
        start_frame=start_frame,
        endpoint_frame=endpoint_frame,
        normalized_cost=normalized_cost,
        confidence=confidence,
        duration_ratio=duration_ratio,
    )
