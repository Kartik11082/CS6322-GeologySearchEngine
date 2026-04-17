import json
import os
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bloom_filter2 import BloomFilter

from crawler.config import Config


class DedupFilter:
    """Bloom-filter based URL deduplication with canonical normalization."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._filter = BloomFilter(
            max_elements=self.config.BLOOM_CAPACITY,
            error_rate=self.config.BLOOM_ERROR_RATE,
        )
        self._count = 0

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url or "")

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        path = parsed.path or ""
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query_pairs.sort(key=lambda kv: (kv[0], kv[1]))
        query = urlencode(query_pairs, doseq=True)

        return urlunparse((scheme, netloc, path, parsed.params, query, ""))

    def seen(self, url: str) -> bool:
        normalized = self._normalize_url(url)
        return normalized in self._filter

    def add(self, url: str) -> None:
        normalized = self._normalize_url(url)
        self._filter.add(normalized)
        self._count += 1

    def count(self) -> int:
        return self._count

    def load_from_existing(self, pages_file: str) -> int:
        """Pre-seed bloom filter from an existing pages.jsonl file."""
        if not os.path.exists(pages_file):
            return 0

        loaded = 0
        with open(pages_file, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    url = json.loads(line).get("url", "")
                except Exception:
                    continue
                if not url:
                    continue
                if self.seen(url):
                    continue
                self.add(url)
                loaded += 1

        print(f"Pre-loaded {loaded:,} URLs from existing crawl")
        return loaded
