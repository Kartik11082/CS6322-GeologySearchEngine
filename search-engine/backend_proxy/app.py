from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field


SEARCH_ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEARCH_ENGINE_ROOT.parent
DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "cluster_service" / "benchmarks" / "queries_50.json"
)
SERPAPI_URL = "https://serpapi.com/search"
SEARCH_METHOD_ALIASES = {"combined": "pagerank"}
VALID_SEARCH_METHODS = {"tfidf", "bm25", "pagerank", "hits", "combined"}
VALID_EXTERNAL_ENGINES = {"google", "bing"}
FALLBACK_DEMO_QUERIES = [
    {
        "query_id": "Q001",
        "query_text": "volcanic eruption hawaii",
        "topic": "volcanology",
    },
    {
        "query_id": "Q009",
        "query_text": "earthquake fault lines california",
        "topic": "seismology",
    },
    {
        "query_id": "Q017",
        "query_text": "quartz crystal formation",
        "topic": "mineralogy",
    },
    {
        "query_id": "Q024",
        "query_text": "aquifer contamination sources",
        "topic": "hydrogeology",
    },
    {
        "query_id": "Q033",
        "query_text": "metamorphic rock foliation",
        "topic": "petrology",
    },
]


def load_local_env() -> None:
    env_path = SEARCH_ENGINE_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


def config(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def normalize_search_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in VALID_SEARCH_METHODS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid method '{method}'. Must be one of: "
                f"{', '.join(sorted(VALID_SEARCH_METHODS))}."
            ),
        )
    return SEARCH_METHOD_ALIASES.get(normalized, normalized)


def parse_upstream_error(
    exc: requests.RequestException, fallback_status: int = 502
) -> HTTPException:
    response = getattr(exc, "response", None)
    if response is None:
        return HTTPException(status_code=fallback_status, detail=str(exc))

    try:
        payload = response.json()
    except ValueError:
        payload = response.text or str(exc)

    detail = payload.get("detail") if isinstance(payload, dict) else payload
    return HTTPException(status_code=response.status_code, detail=detail)


def proxy_public_base() -> str:
    return config("PROXY_PUBLIC_BASE_URL", "http://127.0.0.1:8020").rstrip("/")


def root_search_url() -> str:
    return config("ROOT_SEARCH_API_URL", "http://127.0.0.1:8000/api/search")


def root_expand_url() -> str:
    return config("ROOT_EXPAND_API_URL", "http://127.0.0.1:8000/api/expand")


def root_cluster_api_base() -> str:
    return config("ROOT_CLUSTER_API_BASE", "http://127.0.0.1:8010").rstrip("/")


def read_demo_queries() -> list[dict]:
    if not DEFAULT_BENCHMARK_PATH.exists():
        return FALLBACK_DEMO_QUERIES

    with DEFAULT_BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    demo_queries = [query for query in queries if query.get("demo_candidate")]
    return demo_queries or FALLBACK_DEMO_QUERIES


def serpapi_params(engine_name: str, query: str) -> dict[str, str]:
    api_key = config("SERPAPI_API_KEY", "")
    params = {"engine": engine_name, "q": query, "api_key": api_key}
    if engine_name == "google":
        params.update({"google_domain": "google.com", "gl": "us", "hl": "en"})
    else:
        params.update({"cc": "us", "mkt": "en-US"})
    return params


def normalize_external_results(items: list[dict], top_k: int) -> list[dict]:
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or item.get("url") or "").strip()
        if not title or not url:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "position": item.get("position") or len(results) + 1,
                "title": title,
                "url": url,
                "display_url": str(item.get("displayed_link") or "").strip(),
                "snippet": str(item.get("snippet") or item.get("source") or "").strip(),
                "source": str(
                    item.get("source") or item.get("displayed_link") or ""
                ).strip(),
            }
        )
        if len(results) >= top_k:
            break
    return results


class SearchRequest(BaseModel):
    query: str = Field(..., validation_alias=AliasChoices("query", "q"))
    method: str = Field(default="bm25")
    top_k: int = Field(default=10, ge=1, le=50)


class ExpandRequest(BaseModel):
    query: str
    method: Literal["rocchio", "association", "scalar", "metric"] = "association"
    top_k: int = Field(default=10, ge=1, le=20)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    irrelevant_doc_ids: list[str] = Field(default_factory=list)


class ClusteredSearchRequest(BaseModel):
    query: str
    cluster_method: Literal["flat", "ward", "complete"] = "flat"
    baseline_method: str = "combined"
    top_k: int = Field(default=10, ge=1, le=20)
    build_id: str | None = None


class ExternalSearchRequest(BaseModel):
    engine: Literal["google", "bing"]
    query: str
    top_k: int = Field(default=10, ge=1, le=20)


SearchRequest.model_rebuild()
ExpandRequest.model_rebuild()
ClusteredSearchRequest.model_rebuild()
ExternalSearchRequest.model_rebuild()


app = FastAPI(
    title="Search Engine MVP Proxy",
    version="1.0.0",
    description="Isolated proxy layer for the React search engine MVP.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "proxy_public_base_url": proxy_public_base(),
        "root_search_api_url": root_search_url(),
        "root_expand_api_url": root_expand_url(),
        "root_cluster_api_base": root_cluster_api_base(),
        "serpapi_configured": bool(config("SERPAPI_API_KEY", "")),
    }


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    resolved_method = normalize_search_method(req.method)
    t0 = time.time()
    try:
        response = requests.post(
            root_search_url(),
            json={"query": req.query, "method": resolved_method, "top_k": req.top_k},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise parse_upstream_error(exc) from exc

    payload["query"] = req.query
    payload["method"] = resolved_method
    payload.setdefault("metadata", {})
    payload["metadata"]["proxy_execution_time_ms"] = round((time.time() - t0) * 1000, 2)
    return payload


@app.post("/api/expand")
def expand(req: ExpandRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        response = requests.post(
            root_expand_url(),
            json=req.model_dump(),
            timeout=45,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise parse_upstream_error(exc) from exc


@app.post("/api/clustered-search")
def clustered_search(req: ClusteredSearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    payload = {
        "query": req.query,
        "cluster_method": req.cluster_method,
        "baseline_method": req.baseline_method,
        "top_k": req.top_k,
        "build_id": req.build_id,
        "search_adapter": {
            "url": f"{proxy_public_base()}/api/search",
            "http_method": "POST",
            "query_field": "q",
            "method_field": "method",
            "top_k_field": "top_k",
            "results_path": ["results"],
        },
    }
    if req.build_id is None:
        payload.pop("build_id")

    try:
        response = requests.post(
            f"{root_cluster_api_base()}/v1/rerank",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise parse_upstream_error(exc) from exc


@app.post("/api/external-search")
def external_search(req: ExternalSearchRequest) -> dict:
    engine_name = req.engine.strip().lower()
    if engine_name not in VALID_EXTERNAL_ENGINES:
        raise HTTPException(
            status_code=422, detail="Engine must be one of: google, bing."
        )

    if not config("SERPAPI_API_KEY", ""):
        raise HTTPException(
            status_code=503, detail="SERPAPI_API_KEY is not configured."
        )

    t0 = time.time()
    try:
        response = requests.get(
            SERPAPI_URL,
            params=serpapi_params(engine_name, req.query),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise parse_upstream_error(exc) from exc

    results = normalize_external_results(
        payload.get("organic_results") or [], req.top_k
    )
    return {
        "status": "success",
        "engine": engine_name,
        "query": req.query,
        "metadata": {
            "total_results": len(results),
            "execution_time_ms": round((time.time() - t0) * 1000, 2),
        },
        "results": results,
    }


@app.get("/api/demo-queries")
def demo_queries() -> dict:
    queries = read_demo_queries()
    return {"status": "success", "count": len(queries), "queries": queries}
