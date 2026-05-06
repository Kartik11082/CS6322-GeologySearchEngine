"""Shared query lists for X5 expansion experiments (Rocchio vs PRF benchmark)."""

from __future__ import annotations

import json
from pathlib import Path

# Project root = parent of expander/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_QUERIES_PATH = _PROJECT_ROOT / "cluster_service" / "benchmarks" / "queries_50.json"

# 20 queries for explicit Rocchio relevance feedback (curated by X5).
ROCCHIO_QUERIES: list[str] = [
    "volcano",
    "earthquake",
    "fossil",
    "minerals",
    "magma composition",
    "tectonic plates",
    "sedimentary rock",
    "carbon dating",
    "erthsquake",
    "volcnos",
    "fosiils",
    "ignous",
    "yellowstone caldera",
    "san andreas fault",
    "hawaiian pahoehoe",
    "mount vesuvius",
    "software engineering",
    "artificial intelligence",
    "stock market trends",
    "football rules",
]


def load_prf_benchmark_rows(path: Path | None = None) -> list[dict]:
    """Load the 50 shared benchmark queries (collaboration with X2 / evaluation with X3)."""
    p = path or BENCHMARK_QUERIES_PATH
    with open(p, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or len(payload) != 50:
        raise ValueError(f"Expected a JSON list of 50 items in {p}")
    return payload


def prf_query_texts(rows: list[dict] | None = None) -> list[tuple[str, str]]:
    """(query_id, query_text) pairs in benchmark order."""
    data = rows if rows is not None else load_prf_benchmark_rows()
    return [(str(r["query_id"]), str(r["query_text"])) for r in data]


def m_neighbors_for_query(query: str) -> int:
    """Match backend/app.py perform_expansion."""
    return 2 if len(query.split()) <= 3 else 6
