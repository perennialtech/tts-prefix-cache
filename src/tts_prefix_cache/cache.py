from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Hashable
from typing import Any, Generic, TypeVar

from .config import CacheStatus

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class MemoryPrefixCache(Generic[K, V]):
    def __init__(self) -> None:
        self._tasks: dict[K, asyncio.Task[V]] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        key: K,
        factory: Callable[[], Coroutine[Any, Any, V]],
    ) -> tuple[V, CacheStatus]:
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
