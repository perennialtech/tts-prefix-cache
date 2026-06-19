from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from .config import VoiceParams

T = TypeVar("T")


class PrefixCache(Generic[T]):
    def __init__(self) -> None:
        self._values: dict[str, T] = {}
        self._inflight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self._values

    async def get_or_create(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if key in self._values:
                return self._values[key]

            task = self._inflight.get(key)
            if task is None:
                task = asyncio.ensure_future(factory())
                self._inflight[key] = task

        try:
            value = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                async with self._lock:
                    if self._inflight.get(key) is task:
                        del self._inflight[key]
            raise
        except BaseException:
            async with self._lock:
                if self._inflight.get(key) is task:
                    del self._inflight[key]
            raise

        async with self._lock:
            self._values.setdefault(key, value)
            if self._inflight.get(key) is task:
                del self._inflight[key]
            return self._values[key]


def prefix_cache_key(prefix: str, voice: VoiceParams) -> str:
    raw = f"{voice.model}|{voice.voice_id}|{voice.speed}|{voice.pitch}|{prefix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
