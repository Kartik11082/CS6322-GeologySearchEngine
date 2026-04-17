"""Load crawled pages and edges from the new crawler data files.

New format:
  - pages.jsonl  : one JSON object per line (not gzipped)
      {url, title, text, content_type, crawled_at, status, depth}
  - web_graph.csv: CSV with columns src_url, dst_url
"""

import csv
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import PAGES_JSONL_PATH, WEB_GRAPH_CSV_PATH


def load_pages(pages_path: Path = PAGES_JSONL_PATH) -> list[dict[str, Any]]:
    """Read all pages from pages.jsonl and assign sequential doc_id values.

    Each page record gets a ``doc_id`` field (int, starting at 1) because the
    new schema does not include one.  Only pages with HTTP status 200 are kept.
    """
    pages: list[dict[str, Any]] = []
    doc_id = 0
    with open(pages_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # skip non-200 responses — they have no useful content
            if record.get("status") != 200:
                continue
            doc_id += 1
            record["doc_id"] = doc_id
            pages.append(record)
    return pages


def load_edges(edges_path: Path = WEB_GRAPH_CSV_PATH) -> list[dict[str, Any]]:
    """Read all edges from web_graph.csv.

    Returns a list of dicts with keys ``src_url`` and ``dst_url``.
    """
    edges: list[dict[str, Any]] = []
    with open(edges_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edges.append(
                {
                    "src_url": row["src_url"].strip(),
                    "dst_url": row["dst_url"].strip(),
                }
            )
    return edges


def build_url_to_docid(pages: list[dict[str, Any]]) -> dict[str, int]:
    """Map every page URL to its doc_id for graph resolution."""
    mapping: dict[str, int] = {}
    for page in pages:
        mapping[page["url"]] = page["doc_id"]
    return mapping


if __name__ == "__main__":
    pages = load_pages()
    edges = load_edges()
    url_map = build_url_to_docid(pages)
    print(f"Loaded {len(pages)} pages, {len(edges)} edges, {len(url_map)} URL mappings")
