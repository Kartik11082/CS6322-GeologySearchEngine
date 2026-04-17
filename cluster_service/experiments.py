from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .corpus import benchmark_queries, judged_query_ids
from .pipeline import BuildArtifacts
from .rerank import VALID_METHODS, rerank_results
from .search_adapter import SearchAdapterConfig, search_documents
from .utils import normalize_url, utc_now_iso, write_json


def _top_k(results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return list(results[:k])


def _cluster_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in results:
        cluster_id = item.get("cluster_id")
        if cluster_id is not None:
            counts[str(cluster_id)] += 1
    return dict(counts)


def automatic_metrics(results: list[dict[str, Any]], top_k: int = 10) -> dict[str, float]:
    top = _top_k(results, top_k)
    if not top:
        return {
            "cluster_coherence_at_10": 0.0,
            "dominant_cluster_share_at_10": 0.0,
            "average_query_cluster_affinity_at_10": 0.0,
            "average_absolute_rank_shift": 0.0,
        }

    top_cluster = top[0].get("cluster_id")
    cluster_coherence = (
        sum(1 for item in top if item.get("cluster_id") == top_cluster) / len(top)
        if top_cluster is not None
        else 0.0
    )
    counts = _cluster_counts(top)
    dominant_share = max(counts.values(), default=0) / len(top)
    avg_affinity = sum(float(item.get("cluster_affinity", 0.0)) for item in top) / len(top)
    avg_rank_shift = (
        sum(abs(int(item.get("rank_delta", 0))) for item in results) / len(results)
        if results
        else 0.0
    )
    return {
        "cluster_coherence_at_10": round(cluster_coherence, 4),
        "dominant_cluster_share_at_10": round(dominant_share, 4),
        "average_query_cluster_affinity_at_10": round(avg_affinity, 4),
        "average_absolute_rank_shift": round(avg_rank_shift, 4),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 4)


def run_experiment(
    run_id: str,
    build: BuildArtifacts,
    benchmark_path: Path,
    adapter: SearchAdapterConfig,
    baseline_method: str,
    top_k: int,
    output_dir: Path,
) -> dict[str, Any]:
    queries = benchmark_queries(benchmark_path)
    per_query = []
    method_rows: dict[str, list[dict[str, float]]] = {method: [] for method in VALID_METHODS}
    baseline_rows: list[dict[str, float]] = []

    for query_meta in queries:
        query_text = str(query_meta["query_text"])
        baseline = search_documents(query_text, baseline_method, top_k, adapter)
        baseline_with_placeholders = [
            {
                **doc,
                "cluster_id": None,
                "cluster_name": "unassigned",
                "cluster_affinity": 0.0,
                "rank_delta": 0,
            }
            for doc in baseline
        ]
        baseline_metrics = automatic_metrics(baseline_with_placeholders, top_k=10)
        baseline_rows.append(baseline_metrics)

        methods_payload = {}
        for method in sorted(VALID_METHODS):
            reranked_payload = rerank_results(query_text, baseline, method, build)
            metrics = automatic_metrics(reranked_payload["reranked"], top_k=10)
            metrics["cluster_affinity_delta"] = round(
                metrics["average_query_cluster_affinity_at_10"]
                - baseline_metrics["average_query_cluster_affinity_at_10"],
                4,
            )
            method_rows[method].append(metrics)
            methods_payload[method] = {
                "metrics": metrics,
                "clusters": reranked_payload["clusters"],
                "results": reranked_payload["reranked"],
            }

        per_query.append(
            {
                **query_meta,
                "baseline": {
                    "metrics": baseline_metrics,
                    "results": baseline_with_placeholders,
                },
                "methods": methods_payload,
            }
        )

    summary = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "build_id": build.build_id,
        "baseline_method": baseline_method,
        "top_k": top_k,
        "queries": len(per_query),
        "automatic_metrics": {
            "baseline": {
                key: _mean(baseline_rows, key)
                for key in baseline_rows[0].keys()
            }
            if baseline_rows
            else {},
            **{
                method: {
                    key: _mean(rows, key)
                    for key in rows[0].keys()
                }
                for method, rows in method_rows.items()
                if rows
            },
        },
    }
    summary["example_queries"] = select_example_queries(per_query, use_judged=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "per_query.json", per_query)
    return {"summary": summary, "per_query": per_query}


def _write_judgment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "topic",
        "url",
        "title",
        "systems",
        "relevance",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_judgment_template(
    run_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    judged_ids = judged_query_ids(run_payload["per_query"])
    pooled: dict[tuple[str, str], dict[str, Any]] = {}

    for query_meta in run_payload["per_query"]:
        query_id = str(query_meta["query_id"])
        if query_id not in judged_ids:
            continue

        systems = {
            "baseline": query_meta["baseline"]["results"],
            **{
                method: query_meta["methods"][method]["results"]
                for method in VALID_METHODS
            },
        }
        for system_name, documents in systems.items():
            for doc in documents[:10]:
                key = (query_id, normalize_url(doc.get("url", "")))
                if not key[1]:
                    continue
                row = pooled.setdefault(
                    key,
                    {
                        "query_id": query_id,
                        "query_text": query_meta["query_text"],
                        "topic": query_meta.get("topic", ""),
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "systems": set(),
                        "relevance": "",
                        "notes": "",
                    },
                )
                row["systems"].add(system_name)

    output_rows = []
    for _, row in sorted(pooled.items()):
        output_rows.append(
            {
                **row,
                "systems": ";".join(sorted(row["systems"])),
            }
        )

    csv_path = output_dir / "judgments_template.csv"
    _write_judgment_csv(csv_path, output_rows)
    return {
        "query_count": len(judged_ids),
        "rows": output_rows,
        "csv_path": str(csv_path),
    }


def precision_at_k(results: list[dict[str, Any]], qrels: dict[str, int], k: int) -> float:
    top = _top_k(results, k)
    if not top:
        return 0.0
    hits = sum(1 for item in top if qrels.get(normalize_url(item.get("url", "")), 0) > 0)
    return hits / k


def success_at_k(results: list[dict[str, Any]], qrels: dict[str, int], k: int) -> float:
    top = _top_k(results, k)
    return (
        1.0
        if any(qrels.get(normalize_url(item.get("url", "")), 0) > 0 for item in top)
        else 0.0
    )


def reciprocal_rank(results: list[dict[str, Any]], qrels: dict[str, int]) -> float:
    for idx, item in enumerate(results, start=1):
        if qrels.get(normalize_url(item.get("url", "")), 0) > 0:
            return 1.0 / idx
    return 0.0


def dcg_at_k(results: list[dict[str, Any]], qrels: dict[str, int], k: int) -> float:
    score = 0.0
    for idx, item in enumerate(_top_k(results, k), start=1):
        rel = qrels.get(normalize_url(item.get("url", "")), 0)
        score += (2**rel - 1) / np.log2(idx + 1)
    return float(score)


def ndcg_at_k(results: list[dict[str, Any]], qrels: dict[str, int], k: int) -> float:
    ideal_rels = sorted(qrels.values(), reverse=True)
    ideal = 0.0
    for idx, rel in enumerate(ideal_rels[:k], start=1):
        ideal += (2**rel - 1) / np.log2(idx + 1)
    if ideal == 0:
        return 0.0
    return dcg_at_k(results, qrels, k) / ideal


def evaluate_with_judgments(
    run_payload: dict[str, Any],
    judgments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in judgments:
        query_id = str(row["query_id"])
        url = normalize_url(str(row["url"]))
        if not query_id or not url:
            continue
        qrels[query_id][url] = int(row["relevance"])

    systems = ["baseline", "flat", "ward", "complete"]
    per_query = []
    summary_accumulator: dict[str, list[dict[str, float]]] = defaultdict(list)

    for query_meta in run_payload["per_query"]:
        query_id = str(query_meta["query_id"])
        if query_id not in qrels:
            continue
        query_row = {
            "query_id": query_id,
            "query_text": query_meta["query_text"],
            "topic": query_meta.get("topic", ""),
            "systems": {},
        }
        for system in systems:
            results = (
                query_meta["baseline"]["results"]
                if system == "baseline"
                else query_meta["methods"][system]["results"]
            )
            metrics = {
                "precision_at_5": round(precision_at_k(results, qrels[query_id], 5), 4),
                "precision_at_10": round(precision_at_k(results, qrels[query_id], 10), 4),
                "success_at_5": round(success_at_k(results, qrels[query_id], 5), 4),
                "success_at_10": round(success_at_k(results, qrels[query_id], 10), 4),
                "mrr": round(reciprocal_rank(results, qrels[query_id]), 4),
                "ndcg_at_5": round(ndcg_at_k(results, qrels[query_id], 5), 4),
                "ndcg_at_10": round(ndcg_at_k(results, qrels[query_id], 10), 4),
            }
            query_row["systems"][system] = metrics
            summary_accumulator[system].append(metrics)
        per_query.append(query_row)

    summary = {
        system: {
            key: round(
                sum(row[key] for row in rows) / len(rows),
                4,
            )
            for key in rows[0].keys()
        }
        for system, rows in summary_accumulator.items()
        if rows
    }
    judged_payload = {
        "evaluated_queries": len(per_query),
        "summary": summary,
        "per_query": per_query,
    }
    write_json(output_dir / "judged_metrics.json", judged_payload)
    return judged_payload


def select_example_queries(
    per_query: list[dict[str, Any]],
    use_judged: bool,
    judged_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    judged_lookup = {
        row["query_id"]: row
        for row in (judged_payload or {}).get("per_query", [])
    }

    for query_meta in per_query:
        if not query_meta.get("demo_candidate", False):
            continue
        topic = query_meta.get("topic", "")
        query_id = str(query_meta["query_id"])
        if use_judged and query_id in judged_lookup:
            judged_row = judged_lookup[query_id]
            for method in VALID_METHODS:
                delta = round(
                    judged_row["systems"][method]["ndcg_at_10"]
                    - judged_row["systems"]["baseline"]["ndcg_at_10"],
                    4,
                )
                candidates.append((delta, query_id, topic, method, query_meta))
        else:
            for method in VALID_METHODS:
                delta = float(query_meta["methods"][method]["metrics"]["cluster_affinity_delta"])
                candidates.append((delta, query_id, topic, method, query_meta))

    picked_topics: set[str] = set()
    examples = []
    for delta, _, topic, method, query_meta in sorted(candidates, key=lambda item: item[0], reverse=True):
        if topic in picked_topics:
            continue
        examples.append(
            {
                "query_id": query_meta["query_id"],
                "query_text": query_meta["query_text"],
                "topic": topic,
                "method": method,
                "delta": round(delta, 4),
                "baseline_results": query_meta["baseline"]["results"][:5],
                "reranked_results": query_meta["methods"][method]["results"][:5],
                "clusters": query_meta["methods"][method]["clusters"][:5],
            }
        )
        picked_topics.add(topic)
        if len(examples) == 3:
            break
    return examples
