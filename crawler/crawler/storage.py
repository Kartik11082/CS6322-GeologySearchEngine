import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from crawler.config import Config


class Storage:
    """Async storage for crawled pages, edges, and crawl statistics."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pages_path = self.output_dir / self.config.PAGES_FILE
        self.edges_path = self.output_dir / self.config.EDGES_FILE
        self.stats_path = self.output_dir / self.config.STATS_FILE

        self._pages_file = None
        self._edges_file = None
        self._initialized = False
        self._edges_header_written = self.edges_path.exists() and self.edges_path.stat().st_size > 0

        self._init_lock = asyncio.Lock()
        self._page_lock = asyncio.Lock()
        self._edge_lock = asyncio.Lock()

        self.total_pages = 0
        self.total_edges = 0

    async def _ensure_open(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._pages_file = await aiofiles.open(self.pages_path, mode="a", encoding="utf-8")
            self._edges_file = await aiofiles.open(self.edges_path, mode="a", encoding="utf-8")
            self._initialized = True

    async def save_page(self, page_dict: dict) -> None:
        await self._ensure_open()
        line = json.dumps(page_dict, ensure_ascii=False) + "\n"
        async with self._page_lock:
            await self._pages_file.write(line)
            self.total_pages += 1

    async def save_edges(self, src_url: str, links: list[str]) -> None:
        if not links:
            return

        await self._ensure_open()
        async with self._edge_lock:
            if not self._edges_header_written:
                await self._edges_file.write("src_url,dst_url\n")
                self._edges_header_written = True

            # Write all edges for this page in one batch to reduce await/io overhead.
            rows = "".join(f"{src_url},{dst_url}\n" for dst_url in links)
            await self._edges_file.write(rows)
            self.total_edges += len(links)

    async def close(self) -> None:
        if self._pages_file is not None:
            await self._pages_file.flush()
            await self._pages_file.close()
            self._pages_file = None

        if self._edges_file is not None:
            await self._edges_file.flush()
            await self._edges_file.close()
            self._edges_file = None

        stats = {
            "total_pages": self.total_pages,
            "total_edges": self.total_edges,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        async with aiofiles.open(self.stats_path, mode="w", encoding="utf-8") as handle:
            await handle.write(json.dumps(stats, ensure_ascii=False, indent=2))
