from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .pipeline import BuildArtifacts


VALID_METHODS = {"flat", "ward", "complete"}


def _normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    max_score = max(float(item.get("score", 0.0)) for item in results) or 1.0
    return {
        item["normalized_url"]: float(item.get("score", 0.0)) / max_score
        for item in results
        if item.get("normalized_url")
    }


def rerank_results(
    query: str,
    baseline_results: list[dict[str, Any]],
    method: str,
    artifacts: BuildArtifacts,
) -> dict[str, Any]:
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported clustering method: {method}")

    normalized_scores = _normalize_scores(baseline_results)
    query_vec = artifacts.projector.transform_query(query)
    centroids = artifacts.centroids[method]
    cluster_affinity_scores = cosine_similarity([query_vec], centroids).reshape(-1)

    per_cluster_support: defaultdict[int, float] = defaultdict(float)
    cluster_counts = Counter()
    enriched = []
    for doc in baseline_results:
        normalized_url = doc.get("normalized_url", "")
        assignment = artifacts.assignments.get(normalized_url)
        cluster_payload = assignment.get(method) if assignment else None
        cluster_id = cluster_payload["id"] if cluster_payload else None
        cluster_name = cluster_payload["name"] if cluster_payload else "unassigned"
        if cluster_id is not None:
            per_cluster_support[int(cluster_id)] += normalized_scores.get(
                normalized_url, 0.0
            )
            cluster_counts[int(cluster_id)] += 1
        enriched.append(
            {
                **doc,
                "cluster_id": str(cluster_id) if cluster_id is not None else None,
                "cluster_name": cluster_name,
            }
        )

    max_support = max(per_cluster_support.values(), default=1.0) or 1.0
    reranked = []
    for doc in enriched:
        normalized_url = doc.get("normalized_url", "")
        cluster_id = doc.get("cluster_id")
        baseline_score = normalized_scores.get(normalized_url, 0.0)
        if cluster_id is None:
            affinity = 0.0
            support = 0.0
        else:
            cid = int(cluster_id)
            affinity = float(cluster_affinity_scores[cid])
            support = per_cluster_support[cid] / max_support

        rerank_score = 0.70 * baseline_score + 0.20 * affinity + 0.10 * support
        reranked.append(
            {
                **doc,
                "baseline_score": round(float(doc.get("score", 0.0)), 6),
                "baseline_score_normalized": round(baseline_score, 6),
                "cluster_affinity": round(affinity, 6),
                "cluster_support": round(support, 6),
                "score": round(rerank_score, 6),
            }
        )

    reranked.sort(key=lambda item: item["score"], reverse=True)
    baseline_positions = {
        item.get("normalized_url", ""): idx
        for idx, item in enumerate(enriched, start=1)
    }
    for idx, item in enumerate(reranked, start=1):
        before = baseline_positions.get(item.get("normalized_url", ""), idx)
        item["rank"] = idx
        item["baseline_rank"] = before
        item["rank_delta"] = before - idx

    clusters = []
    for cluster in artifacts.cluster_catalog["methods"][method]["clusters"]:
        count = cluster_counts.get(int(cluster["id"]), 0)
        clusters.append(
            {
                "id": cluster["id"],
                "name": cluster["name"],
                "size": cluster["size"],
                "result_count": int(count),
                "representatives": cluster["representatives"],
            }
        )

    clusters.sort(key=lambda item: (-item["result_count"], -item["size"], item["name"]))
    return {
        "method": method,
        "query": query,
        "clusters": clusters,
        "baseline": enriched,
        "reranked": reranked,
        "explanations": {
            "weights": {
                "baseline": 0.70,
                "cluster_affinity": 0.20,
                "cluster_support": 0.10,
            }
        },
    }
