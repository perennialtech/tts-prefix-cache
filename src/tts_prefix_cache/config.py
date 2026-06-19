from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from ._audio import Audio

CacheStatus = Literal["hit", "miss", "joined"]


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Voice:
    model: str
    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.voice_id:
            raise ValueError("voice_id must not be empty")
        _require_positive("speed", self.speed)
        _require_positive("pitch", self.pitch)


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

    rms_window_ms: float = 42.6666667
    rms_hop_ms: float = 10.6666667

    dtw_hop_ms: float = 10.0
    dtw_win_ms: float = 30.0
    dtw_max_search_multiplier: float = 1.8
    dtw_max_search_extra_ms: float = 600.0

    def __post_init__(self) -> None:
        _require_positive("prefix_trim_keep_ms", self.prefix_trim_keep_ms)
        _require_positive("dtw_trim_keep_ms", self.dtw_trim_keep_ms)
        _require_positive("search_before_ms", self.search_before_ms)
        _require_positive("search_after_ms", self.search_after_ms)
        _require_positive("min_quiet_ms", self.min_quiet_ms)
        _require_positive("keep_quiet_ms", self.keep_quiet_ms)
        _require_positive("low_amp_window_ms", self.low_amp_window_ms)
        _require_positive("continuation_fade_in_ms", self.continuation_fade_in_ms)
        _require_positive("rms_window_ms", self.rms_window_ms)
        _require_positive("rms_hop_ms", self.rms_hop_ms)
        _require_positive("dtw_hop_ms", self.dtw_hop_ms)
        _require_positive("dtw_win_ms", self.dtw_win_ms)
        _require_positive("dtw_max_search_multiplier", self.dtw_max_search_multiplier)
        _require_positive("dtw_max_search_extra_ms", self.dtw_max_search_extra_ms)


@dataclass(frozen=True)
class PrefixSpeakerConfig:
    sample_rate: int = 24000
    chunk_ms: float = 40.0
    wait_silence_chunk_ms: float = 30.0
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    debug_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        _require_positive("chunk_ms", self.chunk_ms)
        _require_positive("wait_silence_chunk_ms", self.wait_silence_chunk_ms)


@dataclass(frozen=True)
class BoundaryResult:
    cut_sample: int
    method: str


@dataclass(frozen=True)
class SpeakResult:
    boundary: BoundaryResult
    cache_status: CacheStatus
    silence_samples: int
    synth_elapsed_ms: float


class Synthesizer(Protocol):
    async def synthesize(
        self,
        text: str,
        voice: Voice,
        *,
        sample_rate: int,
    ) -> Audio: ...


class AudioSink(Protocol):
    async def write(self, chunk: Audio) -> None: ...
