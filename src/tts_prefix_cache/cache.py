from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from ._boundary import PrefixClip
from .config import CacheStatus, Voice


@dataclass(frozen=True)
class PrefixKey:
    text: str
    voice: Voice
    sample_rate: int
    prefix_trim_keep_ms: float
    dtw_trim_keep_ms: float
    silence_threshold_db: float


class MemoryPrefixCache:
    def __init__(self) -> None:
        self._tasks: dict[PrefixKey, asyncio.Task[PrefixClip]] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        key: PrefixKey,
        factory: Callable[[], Coroutine[Any, Any, PrefixClip]],
    ) -> tuple[PrefixClip, CacheStatus]:
        async with self._lock:
            task = self._tasks.get(key)
            status: CacheStatus | None = None

            if task is not None and task.done():
                if task.cancelled() or task.exception() is not None:
                    del self._tasks[key]
                    task = None
                else:
                    status = "hit"

            if task is None:
                task = asyncio.create_task(factory())
                self._tasks[key] = task
                status = "miss"
            elif status is None:
                status = "joined"

        try:
            value = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                async with self._lock:
                    if self._tasks.get(key) is task:
                        del self._tasks[key]
            raise
        except BaseException:
            async with self._lock:
                if self._tasks.get(key) is task:
                    del self._tasks[key]
            raise

        return value, status
