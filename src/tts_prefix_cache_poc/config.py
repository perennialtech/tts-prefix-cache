from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .audio import Audio


@dataclass(frozen=True)
class VoiceParams:
    model: str = "fake-goblin-tts"
    voice_id: str = "raccoon-narrator"
    speed: float = 1.0
    pitch: float = 1.0


@dataclass(frozen=True)
class PlaybackConfig:
    sr: int = 24000
    chunk_ms: float = 40.0
    latency_silence_chunk_ms: float = 30.0
    dump_debug: bool = False
    debug_dir: str = "poc_debug"


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

    frame_size: int = 1024
    hop: int = 256

    dtw_hop_ms: float = 10.0
    dtw_win_ms: float = 30.0
    dtw_max_search_multiplier: float = 1.8
    dtw_max_search_extra_ms: float = 600.0


@dataclass(frozen=True)
class BoundaryResult:
    cut_sample: int
    method: str


class Synthesizer(Protocol):
    async def synthesize(self, text: str, voice: VoiceParams) -> Audio: ...
