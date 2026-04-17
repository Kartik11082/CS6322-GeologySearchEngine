"""Paths and constants for the indexer module."""

from pathlib import Path

# ── directory layout ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # GeologySearchEngine-CS6322/
CRAWLER_NEW_DIR = PROJECT_ROOT / "crawled_data"
PAGES_JSONL_PATH = CRAWLER_NEW_DIR / "pages.jsonl"
WEB_GRAPH_CSV_PATH = CRAWLER_NEW_DIR / "web_graph.csv"

INDEXER_ROOT = Path(__file__).resolve().parents[1]  # indexer/
INDEXER_DATA_DIR = INDEXER_ROOT / "data"

INDEX_PATH = INDEXER_DATA_DIR / "inverted_index.json"
DOC_STORE_PATH = INDEXER_DATA_DIR / "doc_store.json"
GRAPH_PATH = INDEXER_DATA_DIR / "web_graph.json"
GRAPH_STATS_PATH = INDEXER_DATA_DIR / "graph_stats.json"
PAGERANK_PATH = INDEXER_DATA_DIR / "pagerank_scores.json"


# ── BM25 parameters ──────────────────────────────────────────────
BM25_K1 = 1.2
BM25_B = 0.75

# ── PageRank parameters ──────────────────────────────────────────
PAGERANK_DAMPING = 0.85
PAGERANK_MAX_ITER = 100
PAGERANK_TOL = 1e-6

# ── HITS parameters ──────────────────────────────────────────────
HITS_MAX_ITER = 50
HITS_TOL = 1e-6
HITS_BASE_SET_EXPANSION = 50  # max neighbours to add to base set

# ── Search defaults ───────────────────────────────────────────────
DEFAULT_TOP_K = 10
SNIPPET_CHAR_LIMIT = 300


def ensure_directories() -> None:
    INDEXER_DATA_DIR.mkdir(parents=True, exist_ok=True)
