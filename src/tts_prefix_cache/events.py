from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrefixSpeakerEvent:
    name: str
    data: Mapping[str, object] = field(default_factory=dict)


PrefixSpeakerLogger = Callable[[PrefixSpeakerEvent], None]
