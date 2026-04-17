import asyncio
import random
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

from crawler.config import Config

USER_AGENTS = (
    "Mozilla/5.0 (compatible; GeoEduBot/1.0; geology research crawler)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/17.2",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
)


class Fetcher:
    """Async HTTP fetcher with robots compliance, retry, and size safeguards."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._session: aiohttp.ClientSession | None = None
        self._connector: aiohttp.TCPConnector | None = None
        self._session_lock = asyncio.Lock()
        self._robots_cache: dict[str, RobotFileParser | None] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session

        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                return self._session

            connector = aiohttp.TCPConnector(
                limit=self.config.CONCURRENCY,
                limit_per_host=8,
                ssl=False,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=20,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=self.config.REQUEST_TIMEOUT,
                connect=self.config.CONNECT_TIMEOUT,
            )
            self._connector = connector
            self._session = aiohttp.ClientSession(
                connector=connector,
                connector_owner=False,
                timeout=timeout,
            )
        return self._session

    async def _is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        if not domain:
            return False

        cached = self._robots_cache.get(domain, None)
        if domain in self._robots_cache:
            if cached is None:
                return True
            return cached.can_fetch("GeoEduBot", url)

        lock = self._robots_locks.setdefault(domain, asyncio.Lock())
        async with lock:
            cached = self._robots_cache.get(domain, None)
            if domain in self._robots_cache:
                if cached is None:
                    return True
                return cached.can_fetch("GeoEduBot", url)

            scheme = parsed.scheme.lower() or "https"
            robots_url = f"{scheme}://{domain}/robots.txt"
            try:
                session = await self._get_session()
                async with session.get(
                    robots_url,
                    allow_redirects=True,
                    max_redirects=5,
                    headers={"User-Agent": USER_AGENTS[0]},
                ) as resp:
                    if resp.status >= 400:
                        self._robots_cache[domain] = None
                        return True
                    text = await resp.text(errors="replace")
            except Exception:
                self._robots_cache[domain] = None
                return True

            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(text.splitlines())
            self._robots_cache[domain] = parser
            return parser.can_fetch("GeoEduBot", url)

    async def fetch(self, url: str) -> dict | None:
        if not await self._is_allowed(url):
            return None

        max_bytes = int(self.config.MAX_PAGE_SIZE_MB * 1024 * 1024)
        retries = self.config.MAX_RETRIES

        for attempt in range(retries + 1):
            try:
                session = await self._get_session()
                headers = {"User-Agent": random.choice(USER_AGENTS)}
                async with session.get(
                    url,
                    headers=headers,
                    allow_redirects=True,
                    max_redirects=5,
                ) as resp:
                    content_type_header = (resp.headers.get("Content-Type") or "").lower()
                    content_type = content_type_header.split(";", 1)[0].strip()
                    if content_type not in self.config.ALLOWED_CONTENT_TYPES:
                        return None

                    size_bytes = 0
                    chunks: list[bytes] = []
                    async for chunk in resp.content.iter_chunked(8192):
                        if not chunk:
                            continue
                        size_bytes += len(chunk)
                        if size_bytes > max_bytes:
                            return None
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    charset = resp.charset or "utf-8"
                    html = body.decode(charset, errors="replace")

                    return {
                        "url": str(resp.url),
                        "html": html,
                        "status": resp.status,
                        "content_type": content_type_header,
                        "size_bytes": size_bytes,
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                    }
            except (TimeoutError, aiohttp.ClientError):
                if attempt >= retries:
                    return None
                await asyncio.sleep(self.config.RETRY_BACKOFF * (attempt + 1))

        return None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        if self._connector is not None and not self._connector.closed:
            await self._connector.close()
