"""In-process pub-sub event bus for SSE streaming.

Keyed by experiment_id. The orchestrator publishes events as the loop progresses;
the SSE endpoint subscribes and forwards events to the browser.

Note: in-process only — won't survive an API restart. For v0 that's fine.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[Event]]] = defaultdict(list)
        self._history: dict[str, list[Event]] = defaultdict(list)
        self._closed: set[str] = set()

    def subscribe(self, key: str) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        # Replay history so a late subscriber doesn't miss earlier events
        for past in self._history.get(key, []):
            queue.put_nowait(past)
        self._queues[key].append(queue)
        return queue

    def unsubscribe(self, key: str, queue: asyncio.Queue[Event]) -> None:
        with contextlib.suppress(ValueError):
            self._queues[key].remove(queue)
        if not self._queues[key]:
            self._queues.pop(key, None)

    async def publish(self, key: str, event: Event) -> None:
        self._history[key].append(event)
        # Cap history to last 1000 events per experiment
        if len(self._history[key]) > 1000:
            self._history[key] = self._history[key][-1000:]
        for q in list(self._queues.get(key, [])):
            await q.put(event)

    def close(self, key: str) -> None:
        self._closed.add(key)

    def is_closed(self, key: str) -> bool:
        return key in self._closed

    async def stream(self, key: str) -> AsyncIterator[Event]:
        """Yield events for `key` until a `done` or `error` event arrives."""
        q = self.subscribe(key)
        try:
            while True:
                event = await q.get()
                yield event
                if event.type in {"done", "error", "loop.finished", "loop.failed"}:
                    return
        finally:
            self.unsubscribe(key, q)


bus = EventBus()
