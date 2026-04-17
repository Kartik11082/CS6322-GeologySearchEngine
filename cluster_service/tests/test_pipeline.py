from __future__ import annotations

import csv
import json
from pathlib import Path

from cluster_service.config import ServiceConfig
from cluster_service.pipeline import load_build, run_build
from cluster_service.rerank import rerank_results


def _write_fixture_corpus(base: Path) -> tuple[Path, Path]:
    pages_path = base / "pages.jsonl"
    graph_path = base / "web_graph.csv"

    pages = []
    for idx in range(6):
        pages.append(
            {
                "url": f"https://volcano.example.com/doc-{idx}",
                "title": f"Volcano lava article {idx}",
                "text": (
                    "The earth volcano lava magma eruption and geology guide explains lava flows "
                    f"and volcanic rocks in clear detail for sample {idx}. "
                )
                * 4,
                "content_type": "text/html; charset=utf-8",
                "status": 200,
                "depth": 1,
            }
        )
        pages.append(
            {
                "url": f"https://quake.example.com/doc-{idx}",
                "title": f"Earthquake fault article {idx}",
                "text": (
                    "The earth earthquake fault seismic wave and geology lesson explains tectonic "
                    f"plates and aftershock patterns in detail for sample {idx}. "
                )
                * 4,
                "content_type": "text/html; charset=utf-8",
                "status": 200,
                "depth": 1,
            }
        )
        pages.append(
            {
                "url": f"https://fossil.example.com/doc-{idx}",
                "title": f"Fossil record article {idx}",
                "text": (
                    "The earth fossil record and rock layer guide explains trilobite history and "
                    f"paleozoic geology with detailed examples for sample {idx}. "
                )
                * 4,
                "content_type": "text/html; charset=utf-8",
                "status": 200,
                "depth": 1,
            }
        )

    with open(pages_path, "w", encoding="utf-8") as handle:
        for row in pages:
            handle.write(json.dumps(row) + "\n")

    with open(graph_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src_url", "dst_url"])
        writer.writeheader()
        for index, row in enumerate(pages[:-1]):
            writer.writerow({"src_url": row["url"], "dst_url": pages[index + 1]["url"]})

    return pages_path, graph_path


def _test_config(base: Path, pages_path: Path, graph_path: Path) -> ServiceConfig:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = ServiceConfig(
        crawl_pages_path=pages_path,
        crawl_graph_path=graph_path,
        output_root=base / "output",
        benchmark_path=repo_root / "cluster_service" / "benchmarks" / "queries_50.json",
        min_chars=40,
        training_sample_cap=18,
        domain_cap_per_sample=6,
        tfidf_min_df=1,
        tfidf_max_df=0.95,
        tfidf_max_features=256,
        svd_components=8,
        flat_k_candidates=(3, 4, 5),
        agg_k_candidates=(3, 4),
        mini_clusters=6,
        cluster_top_terms=4,
        cluster_representatives=2,
        batch_size=6,
    )
    return cfg


def test_build_pipeline_creates_artifacts(tmp_path):
    pages_path, graph_path = _write_fixture_corpus(tmp_path)
    cfg = _test_config(tmp_path, pages_path, graph_path)

    manifest = run_build("fixturebuild", cfg, {"url": "http://search.test/api"})
    assert manifest["build_id"] == "fixturebuild"
    assert manifest["corpus"]["records_kept"] == 18
    assert manifest["methods"]["flat"]["cluster_count"] in {3, 4, 5}
    assert manifest["methods"]["ward"]["cluster_count"] in {3, 4}

    build = load_build(cfg.output_root / "builds" / "fixturebuild")
    assert len(build.assignments) == 18
    assert "flat" in build.cluster_catalog["methods"]
    assert build.centroids["flat"].shape[0] == manifest["methods"]["flat"]["cluster_count"]


def test_rerank_results_produces_cluster_annotations(tmp_path):
    pages_path, graph_path = _write_fixture_corpus(tmp_path)
    cfg = _test_config(tmp_path, pages_path, graph_path)
    run_build("fixturebuild", cfg, {"url": "http://search.test/api"})
    build = load_build(cfg.output_root / "builds" / "fixturebuild")

    baseline = []
    for rank, info in enumerate(list(build.assignments.values())[:6], start=1):
        baseline.append(
            {
                "rank": rank,
                "title": info["title"],
                "url": info["url"],
                "normalized_url": next(
                    key for key, value in build.assignments.items() if value["url"] == info["url"]
                ),
                "snippet": "",
                "score": 1.0 / rank,
            }
        )

    payload = rerank_results("volcano lava geology", baseline, "flat", build)
    assert len(payload["reranked"]) == 6
    assert payload["reranked"][0]["cluster_name"]
    assert all("cluster_affinity" in row for row in payload["reranked"])
