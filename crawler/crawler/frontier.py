import asyncio
import time
from urllib.parse import urlsplit

from crawler.config import Config


class Frontier:
    """In-memory async URL frontier with per-domain politeness."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._last_fetch_time: dict[str, float] = {}

    async def push(self, url: str, depth: int) -> None:
        """Push a URL into the frontier as (depth, url)."""
        await self._queue.put((depth, url))

    async def pop(self) -> tuple[str, int] | None:
        """
        Pop next crawlable item, enforcing per-domain cooldown.

        Returns:
            (url, depth) when available and allowed now, else None.
        """
        if self._queue.empty():
            return None

        deferred: list[tuple[int, str]] = []
        checked = 0

        while checked < 50 and not self._queue.empty():
            try:
                depth, url = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            checked += 1
            try:
                domain = (urlsplit(url).hostname or "").lower()
            except ValueError:
                # Skip malformed URLs instead of crashing workers.
                continue
            now = time.monotonic()
            last = self._last_fetch_time.get(domain)

            if domain and last is not None and (now - last) < self.config.DOMAIN_DELAY:
                deferred.append((depth, url))
                continue

            if domain:
                self._last_fetch_time[domain] = now

            for item in deferred:
                await self._queue.put(item)
            return (url, depth)

        for item in deferred:
            await self._queue.put(item)

        if checked >= 50 and deferred:
            await asyncio.sleep(0.05)

        return None

    def size(self) -> int:
        """Return number of queued items."""
        return self._queue.qsize()
