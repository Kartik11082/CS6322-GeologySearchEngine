from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize

from .config import ServiceConfig
from .corpus import CorpusRecord, benchmark_queries, load_corpus, select_balanced_sample
from .utils import read_json, utc_now_iso, write_json
from .vectorizer import ProjectionArtifacts, fit_projection


@dataclass(slots=True)
class BuildArtifacts:
    build_id: str
    build_dir: Path
    manifest: dict[str, Any]
    assignments: dict[str, dict[str, Any]]
    cluster_catalog: dict[str, Any]
    centroids: dict[str, np.ndarray]
    projector: ProjectionArtifacts


def _score_clustering(
    vectors: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> tuple[float, float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0, math.inf
    silhouette = float(silhouette_score(vectors, labels, metric=metric))
    dbi = float(davies_bouldin_score(vectors, labels))
    return silhouette, dbi


def _pick_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.get("silhouette", float("-inf"))),
            item.get("davies_bouldin", float("inf")),
            item.get("k", 0),
        ),
    )
    return ranked[0]


def _ctfidf_labels(
    records: list[CorpusRecord],
    labels: np.ndarray,
    vectorizer,
    top_n: int,
) -> dict[int, str]:
    vocab = vectorizer.vocabulary_
    count_vectorizer = CountVectorizer(
        vocabulary=vocab,
        stop_words="english",
        ngram_range=(1, 2),
    )
    ordered_ids = sorted({int(label) for label in labels.tolist()})
    class_docs = []
    for cluster_id in ordered_ids:
        joined = " ".join(
            record.clustering_text
            for record, label in zip(records, labels)
            if int(label) == cluster_id
        )
        class_docs.append(joined or "empty")

    count_matrix = count_vectorizer.transform(class_docs)
    if sparse.issparse(count_matrix):
        counts = count_matrix.toarray().astype(np.float64)
    else:
        counts = np.asarray(count_matrix, dtype=np.float64)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tf = counts / row_sums
    doc_freq = np.count_nonzero(counts > 0, axis=0)
    idf = np.log((1.0 + counts.shape[0]) / (1.0 + doc_freq)) + 1.0
    ctfidf = tf * idf
    feature_names = np.array(count_vectorizer.get_feature_names_out())

    labels_out: dict[int, str] = {}
    for row_idx, cluster_id in enumerate(ordered_ids):
        weights = ctfidf[row_idx]
        top_idx = np.argsort(weights)[-top_n:][::-1]
        top_terms = [str(feature_names[i]) for i in top_idx if weights[i] > 0]
        labels_out[cluster_id] = ", ".join(top_terms[:top_n]) or f"cluster {cluster_id}"
    return labels_out


def _representatives(
    records: list[CorpusRecord],
    labels: np.ndarray,
    embeddings: np.ndarray,
    cluster_centroids: np.ndarray,
    cluster_names: dict[int, str],
    top_n: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ordered_ids = sorted(cluster_names)
    for cluster_id in ordered_ids:
        mask = labels == cluster_id
        cluster_vectors = embeddings[mask]
        cluster_records = [record for record, keep in zip(records, mask) if keep]
        if not cluster_records:
            output.append(
                {
                    "id": str(cluster_id),
                    "name": cluster_names[cluster_id],
                    "size": 0,
                    "representatives": [],
                }
            )
            continue
        center = cluster_centroids[cluster_id : cluster_id + 1]
        sims = cosine_similarity(cluster_vectors, center).reshape(-1)
        top_idx = np.argsort(sims)[-top_n:][::-1]
        reps = [
            {
                "url": cluster_records[i].url,
                "normalized_url": cluster_records[i].normalized_url,
                "title": cluster_records[i].title,
                "domain": cluster_records[i].domain,
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_idx
        ]
        output.append(
            {
                "id": str(cluster_id),
                "name": cluster_names[cluster_id],
                "size": int(mask.sum()),
                "representatives": reps,
            }
        )
    return output


def _method_centroids(labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    cluster_ids = sorted({int(label) for label in labels.tolist()})
    centroids = []
    for cluster_id in cluster_ids:
        centroid = embeddings[labels == cluster_id].mean(axis=0)
        centroids.append(centroid)
    matrix = np.vstack(centroids).astype(np.float32)
    return l2_normalize(matrix)


def _fit_flat(
    sample_vectors: np.ndarray,
    full_vectors: np.ndarray,
    k_candidates: tuple[int, ...],
    cfg: ServiceConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_rows = []
    fitted_models = {}
    for k in k_candidates:
        if k >= len(sample_vectors):
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.random_seed,
            batch_size=cfg.kmeans_batch_size,
            n_init=cfg.kmeans_n_init,
            max_iter=cfg.kmeans_max_iter,
        )
        sample_labels = model.fit_predict(sample_vectors)
        silhouette, dbi = _score_clustering(
            sample_vectors, sample_labels, metric="euclidean"
        )
        row = {
            "k": int(k),
            "silhouette": round(silhouette, 4),
            "davies_bouldin": round(dbi, 4),
            "inertia": round(float(model.inertia_), 4),
        }
        candidate_rows.append(row)
        fitted_models[k] = model

    best = _pick_best(candidate_rows)
    model = fitted_models[int(best["k"])]
    full_labels = model.predict(full_vectors).astype(int)
    centroids = l2_normalize(model.cluster_centers_.astype(np.float32))
    return (
        full_labels,
        centroids,
        {"candidate_scores": candidate_rows, "selected": best},
    )


def _fit_mini_clusters(
    full_vectors: np.ndarray, cfg: ServiceConfig
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = min(cfg.mini_clusters, len(full_vectors))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=cfg.random_seed,
        batch_size=cfg.kmeans_batch_size,
        n_init=cfg.kmeans_n_init,
        max_iter=cfg.kmeans_max_iter,
    )
    labels = model.fit_predict(full_vectors).astype(int)
    centers = model.cluster_centers_.astype(np.float32)
    return labels, centers


def _fit_agglomerative(
    mini_labels: np.ndarray,
    mini_centers: np.ndarray,
    full_vectors: np.ndarray,
    method: str,
    k_candidates: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "ward":
        evaluation_vectors = mini_centers
        eval_metric = "euclidean"
        metric = "euclidean"
    else:
        evaluation_vectors = l2_normalize(mini_centers)
        eval_metric = "cosine"
        metric = "cosine"

    candidate_rows = []
    models = {}
    for k in k_candidates:
        if k >= len(evaluation_vectors):
            continue
        model = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=method,
            metric=metric,
        )
        labels = model.fit_predict(evaluation_vectors).astype(int)
        silhouette, dbi = _score_clustering(
            evaluation_vectors, labels, metric=eval_metric
        )
        row = {
            "k": int(k),
            "silhouette": round(silhouette, 4),
            "davies_bouldin": round(dbi, 4),
        }
        candidate_rows.append(row)
        models[k] = labels

    best = _pick_best(candidate_rows)
    best_labels = models[int(best["k"])]
    doc_labels = np.array([best_labels[label] for label in mini_labels], dtype=int)
    centroids = _method_centroids(doc_labels, full_vectors)
    return doc_labels, centroids, {"candidate_scores": candidate_rows, "selected": best}


def run_build(
    build_id: str,
    cfg: ServiceConfig,
    search_adapter_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg.ensure_directories()
    build_dir = cfg.output_root / "builds" / build_id
    build_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(cfg.crawl_pages_path, cfg.crawl_graph_path, cfg)
    if len(corpus.records) < 10:
        raise RuntimeError("Not enough filtered records to cluster.")

    sample_indices = select_balanced_sample(corpus.records, cfg)
    sample_records = [corpus.records[i] for i in sample_indices]
    sample_texts = [record.clustering_text for record in sample_records]
    full_texts = [record.clustering_text for record in corpus.records]

    projector = fit_projection(sample_texts, cfg)
    projection_path = build_dir / "projection.pkl"
    projector.save(projection_path)

    sample_vectors = projector.transform(sample_texts, cfg.batch_size)
    full_vectors = projector.transform(full_texts, cfg.batch_size)

    flat_labels, flat_centroids, flat_meta = _fit_flat(
        sample_vectors, full_vectors, cfg.flat_k_candidates, cfg
    )

    mini_labels, mini_centers = _fit_mini_clusters(full_vectors, cfg)
    ward_labels, ward_centroids, ward_meta = _fit_agglomerative(
        mini_labels, mini_centers, full_vectors, "ward", cfg.agg_k_candidates
    )
    complete_labels, complete_centroids, complete_meta = _fit_agglomerative(
        mini_labels, mini_centers, full_vectors, "complete", cfg.agg_k_candidates
    )

    flat_names = _ctfidf_labels(
        corpus.records, flat_labels, projector.vectorizer, cfg.cluster_top_terms
    )
    ward_names = _ctfidf_labels(
        corpus.records, ward_labels, projector.vectorizer, cfg.cluster_top_terms
    )
    complete_names = _ctfidf_labels(
        corpus.records, complete_labels, projector.vectorizer, cfg.cluster_top_terms
    )

    flat_catalog = _representatives(
        corpus.records,
        flat_labels,
        full_vectors,
        flat_centroids,
        flat_names,
        cfg.cluster_representatives,
    )
    ward_catalog = _representatives(
        corpus.records,
        ward_labels,
        full_vectors,
        ward_centroids,
        ward_names,
        cfg.cluster_representatives,
    )
    complete_catalog = _representatives(
        corpus.records,
        complete_labels,
        full_vectors,
        complete_centroids,
        complete_names,
        cfg.cluster_representatives,
    )

    assignments = {}
    for idx, record in enumerate(corpus.records):
        assignments[record.normalized_url] = {
            "url": record.url,
            "title": record.title,
            "domain": record.domain,
            "flat": {
                "id": int(flat_labels[idx]),
                "name": flat_names[int(flat_labels[idx])],
            },
            "ward": {
                "id": int(ward_labels[idx]),
                "name": ward_names[int(ward_labels[idx])],
            },
            "complete": {
                "id": int(complete_labels[idx]),
                "name": complete_names[int(complete_labels[idx])],
            },
        }

    cluster_catalog = {
        "build_id": build_id,
        "methods": {
            "flat": {
                "selected_k": int(flat_meta["selected"]["k"]),
                "clusters": flat_catalog,
            },
            "ward": {
                "selected_k": int(ward_meta["selected"]["k"]),
                "clusters": ward_catalog,
            },
            "complete": {
                "selected_k": int(complete_meta["selected"]["k"]),
                "clusters": complete_catalog,
            },
        },
    }

    np.savez_compressed(
        build_dir / "cluster_centroids.npz",
        flat=flat_centroids,
        ward=ward_centroids,
        complete=complete_centroids,
    )
    write_json(build_dir / "url_assignments.json", assignments)
    write_json(build_dir / "cluster_catalog.json", cluster_catalog)

    benchmark = benchmark_queries(cfg.benchmark_path)
    manifest = {
        "build_id": build_id,
        "created_at": utc_now_iso(),
        "status": "completed",
        "corpus": {
            **corpus.stats,
            "training_sample_size": len(sample_indices),
        },
        "search_adapter": search_adapter_payload or {},
        "methods": {
            "flat": {
                **flat_meta,
                "cluster_count": int(flat_meta["selected"]["k"]),
            },
            "ward": {
                **ward_meta,
                "cluster_count": int(ward_meta["selected"]["k"]),
            },
            "complete": {
                **complete_meta,
                "cluster_count": int(complete_meta["selected"]["k"]),
            },
        },
        "benchmark": {
            "path": str(cfg.benchmark_path),
            "queries": len(benchmark),
        },
        "artifacts": {
            "projection": "projection.pkl",
            "cluster_catalog": "cluster_catalog.json",
            "assignments": "url_assignments.json",
            "centroids": "cluster_centroids.npz",
        },
    }
    write_json(build_dir / "manifest.json", manifest)
    return manifest


def load_build(build_dir: Path) -> BuildArtifacts:
    manifest = read_json(build_dir / "manifest.json")
    assignments = read_json(build_dir / "url_assignments.json")
    cluster_catalog = read_json(build_dir / "cluster_catalog.json")
    centroids_npz = np.load(build_dir / "cluster_centroids.npz")
    centroids = {
        "flat": np.array(centroids_npz["flat"], dtype=np.float32),
        "ward": np.array(centroids_npz["ward"], dtype=np.float32),
        "complete": np.array(centroids_npz["complete"], dtype=np.float32),
    }
    projector = ProjectionArtifacts.load(build_dir / "projection.pkl")
    return BuildArtifacts(
        build_id=str(manifest["build_id"]),
        build_dir=build_dir,
        manifest=manifest,
        assignments=assignments,
        cluster_catalog=cluster_catalog,
        centroids=centroids,
        projector=projector,
    )
