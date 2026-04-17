from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from .config import ServiceConfig
from .utils import nested_get, normalize_url


@dataclass(slots=True)
class SearchAdapterConfig:
    url: str
    http_method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 20.0
    query_field: str = "q"
    method_field: str = "method"
    top_k_field: str = "top_k"
    results_path: list[str] = field(default_factory=lambda: ["results"])
    title_field: str = "title"
    url_field: str = "url"
    snippet_field: str = "snippet"
    score_field: str = "score"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_payload(
        payload: dict[str, Any] | None,
        cfg: ServiceConfig,
    ) -> "SearchAdapterConfig":
        base = SearchAdapterConfig(
            url=cfg.default_search_api_url,
            http_method=cfg.default_search_api_method.upper(),
        )
        if not payload:
            return base
        merged = {**base.to_dict(), **payload}
        merged["http_method"] = str(merged.get("http_method", "GET")).upper()
        results_path = merged.get("results_path") or ["results"]
        if isinstance(results_path, str):
            results_path = [part for part in results_path.split(".") if part]
        merged["results_path"] = results_path
        return SearchAdapterConfig(**merged)


def search_documents(
    query: str,
    baseline_method: str,
    top_k: int,
    adapter: SearchAdapterConfig,
) -> list[dict[str, Any]]:
    http_method = adapter.http_method.upper()
    payload = {
        adapter.query_field: query,
        adapter.method_field: baseline_method,
        adapter.top_k_field: top_k,
    }

    if http_method == "GET":
        response = requests.get(
            adapter.url,
            params=payload,
            headers=adapter.headers,
            timeout=adapter.timeout_sec,
        )
    else:
        response = requests.post(
            adapter.url,
            json=payload,
            headers=adapter.headers,
            timeout=adapter.timeout_sec,
        )

    response.raise_for_status()
    data = response.json()
    items = nested_get(data, adapter.results_path)
    if not isinstance(items, list):
        raise ValueError(
            f"Expected a list at results path {adapter.results_path!r}, got {type(items)!r}"
        )

    documents = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get(adapter.url_field, "") or "")
        normalized_url = normalize_url(url)
        documents.append(
            {
                "rank": int(item.get("rank") or rank),
                "title": str(item.get(adapter.title_field, "") or ""),
                "url": url,
                "normalized_url": normalized_url,
                "snippet": str(item.get(adapter.snippet_field, "") or ""),
                "score": float(item.get(adapter.score_field) or 0.0),
            }
        )
    return documents
