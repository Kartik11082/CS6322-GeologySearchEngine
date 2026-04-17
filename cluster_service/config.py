from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ServiceConfig:
    project_root: Path = field(default_factory=_project_root)
    crawl_pages_path: Path = field(
        default_factory=lambda: _project_root() / "crawled_data" / "pages.jsonl"
    )
    crawl_graph_path: Path = field(
        default_factory=lambda: _project_root() / "crawled_data" / "web_graph.csv"
    )
    output_root: Path = field(
        default_factory=lambda: _project_root() / "cluster_service" / "output"
    )
    benchmark_path: Path = field(
        default_factory=lambda: _project_root()
        / "cluster_service"
        / "benchmarks"
        / "queries_50.json"
    )

    min_chars: int = 150
    max_text_chars: int = 8_000
    training_sample_cap: int = 30_000
    domain_cap_per_sample: int = 1_000
    cluster_top_terms: int = 6
    cluster_representatives: int = 3
    batch_size: int = 2_000
    random_seed: int = 42

    tfidf_max_features: int = 25_000
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.40
    svd_components: int = 200

    flat_k_candidates: tuple[int, ...] = (8, 12, 16, 20, 24)
    agg_k_candidates: tuple[int, ...] = (8, 12, 16, 20, 24)
    mini_clusters: int = 512
    kmeans_batch_size: int = 1_024
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 200

    build_executor_workers: int = 1
    experiment_executor_workers: int = 2

    default_search_api_url: str = field(
        default_factory=lambda: os.getenv(
            "CLUSTER_SERVICE_SEARCH_API_URL", "http://127.0.0.1:8000/api/search"
        )
    )
    default_search_api_method: str = field(
        default_factory=lambda: os.getenv("CLUSTER_SERVICE_SEARCH_API_METHOD", "POST")
    )

    blocked_url_substrings: tuple[str, ...] = (
        "telegram.me/share",
        "pinterest.com",
        "apps.apple.com",
        "itunes.apple.com",
        "wordpress.com/abuse",
        "localization?currency=",
        "return_to=%2F",
        "replytocom=",
        "/share?",
        "facebook.com/share",
    )
    blocked_title_substrings: tuple[str, ...] = (
        "Attention Required! | Cloudflare",
        "App Store",
        "Enable cookies",
        "Blocked",
        "Just a moment",
    )
    blocked_text_substrings: tuple[str, ...] = (
        "cloudflare ray id",
        "please enable cookies",
        "performance & security by cloudflare",
        "share with others",
        "click to reveal",
    )

    def ensure_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "builds").mkdir(parents=True, exist_ok=True)
        (self.output_root / "experiments").mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.output_root / "service_state.json"
