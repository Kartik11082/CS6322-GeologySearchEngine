"""Relevance models: TF-IDF cosine similarity and Okapi BM25."""

import math
from collections import defaultdict
from typing import Any

from config import BM25_B, BM25_K1
from preprocessor import preprocess


# ═══════════════════════════════════════════════════════════════════
#  Model 1 — TF-IDF with cosine similarity
# ═══════════════════════════════════════════════════════════════════


def _tfidf_weight(tf: int, df: int, N: int) -> float:
    """Log-weighted TF × IDF."""
    if tf <= 0 or df <= 0:
        return 0.0
    return (1.0 + math.log10(tf)) * math.log10(N / df)


def rank_tfidf(
    query: str,
    inverted_index: dict[str, dict[str, int]],
    doc_store: dict[str, dict],
    N: int,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Rank documents by TF-IDF cosine similarity to the query.

    Returns a sorted list of (doc_id, score) descending by score.
    """
    query_terms = preprocess(query)
    if not query_terms:
        return []

    # ── query vector weights ──────────────────────────────────────
    query_weights: dict[str, float] = {}
    term_idfs: dict[str, float] = {}
    
    unique_terms = set(query_terms)
    for term in unique_terms:
        posting = inverted_index.get(term)
        if not posting:
            continue
        df = len(posting)
        idf = math.log10(N / df)
        term_idfs[term] = idf
        query_weights[term] = idf  # TF is 1, so 1 + log10(1) = 1.0. Weight is 1.0 * idf

    if not query_weights:
        return []

    # ── accumulate document scores ────────────────────────────────
    scores: dict[str, float] = defaultdict(float)

    for term, q_w in query_weights.items():
        posting = inverted_index[term]
        idf = term_idfs[term]
        for doc_id, tf in posting.items():
            d_w = (1.0 + math.log10(tf)) * idf
            scores[doc_id] += q_w * d_w

    # ── normalise by document vector magnitude ────────────────────
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
    if query_norm > 0:
        for doc_id in scores:
            d_norm = doc_store.get(doc_id, {}).get("tfidf_norm", 1.0)
            scores[doc_id] /= (query_norm * d_norm) if d_norm > 0 else 1.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════
#  Model 2 — Okapi BM25
# ═══════════════════════════════════════════════════════════════════


def rank_bm25(
    query: str,
    inverted_index: dict[str, dict[str, int]],
    doc_store: dict[str, dict],
    N: int,
    avg_dl: float,
    top_k: int = 10,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[tuple[str, float]]:
    """
    Rank documents using Okapi BM25.

    Returns a sorted list of (doc_id, score) descending by score.
    """
    query_terms = preprocess(query)
    if not query_terms:
        return []

    scores: dict[str, float] = defaultdict(float)
    unique_terms = set(query_terms)

    for term in unique_terms:
        posting = inverted_index.get(term)
        if not posting:
            continue

        df = len(posting)
        # IDF component (with smoothing to avoid negatives)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in posting.items():
            dl = doc_store.get(doc_id, {}).get("doc_len", avg_dl)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
            scores[doc_id] += idf * (numerator / denominator)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
