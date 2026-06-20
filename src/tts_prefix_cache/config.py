from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ._audio import Audio
from ._validation import require_non_negative, require_positive
from .splice import BoundaryResult, SpliceConfig

CacheStatus = Literal["hit", "miss", "joined"]
PlaybackClockMode = Literal["source", "buffered_timeline", "sink"]


class Synthesizer(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        sample_rate: int,
    ) -> Audio: ...


class AudioSink(Protocol):
    async def write(self, chunk: Audio) -> None: ...


@dataclass(frozen=True)
class PrefixSpeakerConfig:
    sample_rate: int = 24000
    chunk_ms: float = 40.0
    wait_silence_chunk_ms: float = 30.0
    playback_clock: PlaybackClockMode = "source"
    output_lead_ms: float = 20.0
    splice: SpliceConfig = field(default_factory=SpliceConfig)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        require_positive("chunk_ms", self.chunk_ms)
        require_positive("wait_silence_chunk_ms", self.wait_silence_chunk_ms)
        require_non_negative("output_lead_ms", self.output_lead_ms)

        if self.playback_clock not in ("source", "buffered_timeline", "sink"):
            raise ValueError(
                'playback_clock must be "source", "buffered_timeline", or "sink"'
            )


@dataclass(frozen=True)
class SpeakResult:
    boundary: BoundaryResult
    cache_status: CacheStatus
    silence_samples: int
    synth_elapsed_ms: float
