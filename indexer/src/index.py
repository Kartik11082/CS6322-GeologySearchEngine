"""Build and persist an inverted index over the crawled pages."""

import json
import math
import multiprocessing
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config import DOC_STORE_PATH, INDEX_PATH, ensure_directories
from loader import load_pages
from preprocessor import preprocess


# ── data structures ───────────────────────────────────────────────
# inverted_index:  { stem: { doc_id_str: tf, ... }, ... }
# doc_store:       { doc_id_str: { url, title, doc_len, content_type, depth }, ... }
# metadata:        { N, avg_dl }


def _process_page(page: dict[str, Any]) -> tuple[str, dict[str, int], int, dict[str, Any]]:
    doc_id = str(page["doc_id"])
    text = f"{page.get('title', '')} {page.get('text', '')}"
    stems = preprocess(text)

    doc_len = len(stems)
    term_counts = dict(Counter(stems))

    doc_meta = {
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "text_preview": page.get("text", "")[:500],
        "doc_len": doc_len,
        "content_type": page.get("content_type", ""),
        "depth": page.get("depth", 0),
    }

    return doc_id, term_counts, doc_len, doc_meta


def build_index(
    pages: list[dict[str, Any]],
) -> tuple[dict, dict, int, float]:
    """
    Build an inverted index from crawled pages.

    Returns
    -------
    inverted_index : dict[str, dict[str, int]]
        term → { doc_id (str): term_frequency }
    doc_store : dict[str, dict]
        doc_id (str) → { url, title, doc_len, content_type, depth }
    N : int
        total number of documents
    avg_dl : float
        average document length (in stems)
    """
    inverted_index: dict[str, dict[str, int]] = {}
    doc_store: dict[str, dict] = {}
    total_tokens = 0
    N = len(pages)

    num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_process_page, pages, chunksize=1000)
        
        for i, (doc_id, term_counts, doc_len, doc_meta) in enumerate(results):
            total_tokens += doc_len
            doc_store[doc_id] = doc_meta
            
            for term, tf in term_counts.items():
                if term not in inverted_index:
                    inverted_index[term] = {}
                inverted_index[term][doc_id] = tf

    avg_dl = total_tokens / N if N > 0 else 0.0

    # ── pre-calculate TF-IDF document norms ───────────────────────
    # To avoid O(N) calculations at search time, we pre-compute the Euclidean
    # norm of each document's TF-IDF vector here.
    doc_norm_sq: dict[str, float] = {doc_id: 0.0 for doc_id in doc_store}
    
    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log10(N / df) if df > 0 else 0.0
        
        for doc_id, tf in postings.items():
            tf_weight = 1.0 + math.log10(tf) if tf > 0 else 0.0
            weight = tf_weight * idf
            doc_norm_sq[doc_id] += weight * weight
            
    for doc_id in doc_store:
        doc_store[doc_id]["tfidf_norm"] = math.sqrt(doc_norm_sq[doc_id])

    return inverted_index, doc_store, N, avg_dl


def save_index(
    inverted_index: dict,
    doc_store: dict,
    N: int,
    avg_dl: float,
    index_path: Path = INDEX_PATH,
    doc_store_path: Path = DOC_STORE_PATH,
) -> None:
    """Write the index and doc store to disk as JSON."""
    ensure_directories()
    payload = {
        "N": N,
        "avg_dl": avg_dl,
        "index": inverted_index,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    with open(doc_store_path, "w", encoding="utf-8") as f:
        json.dump(doc_store, f)
    print(f"Saved index ({len(inverted_index)} terms, {N} docs) -> {index_path}")
    print(f"Saved doc store -> {doc_store_path}")


def load_index(
    index_path: Path = INDEX_PATH,
    doc_store_path: Path = DOC_STORE_PATH,
) -> tuple[dict, dict, int, float]:
    """Load a previously saved index from disk."""
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(doc_store_path, "r", encoding="utf-8") as f:
        doc_store = json.load(f)
    return data["index"], doc_store, data["N"], data["avg_dl"]


# ── CLI: build and save ──────────────────────────────────────────
if __name__ == "__main__":
    print("Loading pages from crawler_new/pages.jsonl...")
    pages = load_pages()
    print(f"  -> {len(pages)} pages loaded")

    print("Building inverted index...")
    t0 = time.time()
    inv_idx, doc_store, N, avg_dl = build_index(pages)
    elapsed = time.time() - t0

    print(f"  -> {len(inv_idx):,} unique terms")
    print(f"  -> avg doc length = {avg_dl:.1f} stems")
    print(f"  -> built in {elapsed:.2f}s")

    save_index(inv_idx, doc_store, N, avg_dl)
