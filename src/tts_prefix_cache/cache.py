from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable
from typing import Any, Generic, TypeVar

from .config import CacheStatus

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class MemoryPrefixCache(Generic[K, V]):
    def __init__(self, max_items: int | None = None) -> None:
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative or None")

        self._tasks: OrderedDict[K, asyncio.Task[V]] = OrderedDict()
        self._max_items = max_items
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
                    self._tasks.move_to_end(key)
                    status = "hit"

            if task is None:
                task = asyncio.create_task(factory())
                self._tasks[key] = task
                status = "miss"
            elif status is None:
                self._tasks.move_to_end(key)
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

        async with self._lock:
            if self._tasks.get(key) is task:
                self._tasks.move_to_end(key)
                self._evict_completed_locked()

        return value, status

    def _evict_completed_locked(self) -> None:
        if self._max_items is None:
            return

        completed_count = sum(
            1 for task in self._tasks.values() if self._is_completed_success(task)
        )
        if completed_count <= self._max_items:
            return

        for key, task in list(self._tasks.items()):
            if completed_count <= self._max_items:
                return

            if not task.done():
                continue

            if task.cancelled() or task.exception() is not None:
                del self._tasks[key]
                continue

            del self._tasks[key]
            completed_count -= 1

    @staticmethod
    def _is_completed_success(task: asyncio.Task[V]) -> bool:
        return task.done() and not task.cancelled() and task.exception() is None
