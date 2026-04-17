from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .manager import ServiceManager
from .rerank import rerank_results
from .schemas import (
    BuildRequest,
    EvaluateExperimentRequest,
    ExperimentRunRequest,
    JudgmentTemplateRequest,
    RerankRequest,
)
from .search_adapter import SearchAdapterConfig, search_documents


app = FastAPI(
    title="Cluster Service",
    version="1.0.0",
    description="Standalone clustering service for crawl-based reranking and experiments.",
)
manager = ServiceManager()


@app.on_event("startup")
def startup_build() -> None:
    manager.ensure_startup_build()


@app.get("/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "current_build_id": manager.current_build_id(),
        "startup_build": manager.startup_status(),
    }


@app.post("/v1/build")
def start_build(request: BuildRequest) -> dict:
    return manager.start_build(
        search_adapter_payload=request.search_adapter.model_dump(exclude_none=True)
        if request.search_adapter
        else None,
        make_current=request.make_current,
    )


@app.get("/v1/build/{build_id}")
def build_status(build_id: str) -> dict:
    try:
        return manager.get_build_status(build_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/build/current")
def current_build() -> dict:
    build_id = manager.current_build_id()
    if not build_id:
        raise HTTPException(status_code=404, detail="No current build is set.")
    return manager.get_build_status(build_id)


@app.post("/v1/rerank")
def rerank(request: RerankRequest) -> dict:
    try:
        build = manager.load_build(request.build_id)
        adapter = SearchAdapterConfig.from_payload(
            request.search_adapter.model_dump(exclude_none=True)
            if request.search_adapter
            else build.manifest.get("search_adapter"),
            manager.cfg,
        )
        baseline = search_documents(
            query=request.query,
            baseline_method=request.baseline_method,
            top_k=request.top_k,
            adapter=adapter,
        )
        payload = rerank_results(request.query, baseline, request.cluster_method, build)
        return {
            "build_id": build.build_id,
            "baseline_method": request.baseline_method,
            "cluster_method": request.cluster_method,
            "query": request.query,
            **payload,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/clusters/{method}")
def cluster_catalog(method: str, build_id: str | None = Query(default=None)) -> dict:
    try:
        build = manager.load_build(build_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    methods = build.cluster_catalog["methods"]
    if method not in methods:
        raise HTTPException(status_code=404, detail=f"Unknown method: {method}")
    return {
        "build_id": build.build_id,
        "method": method,
        **methods[method],
    }


@app.post("/v1/experiments/run")
def start_experiment(request: ExperimentRunRequest) -> dict:
    return manager.start_experiment(
        build_id=request.build_id,
        baseline_method=request.baseline_method,
        top_k=request.top_k,
        search_adapter_payload=request.search_adapter.model_dump(exclude_none=True)
        if request.search_adapter
        else None,
    )


@app.get("/v1/experiments/{run_id}")
def experiment_status(run_id: str) -> dict:
    try:
        return manager.get_experiment(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/experiments/judgment-template")
def judgment_template(request: JudgmentTemplateRequest) -> dict:
    try:
        return manager.build_judgment_template(request.run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/experiments/evaluate")
def evaluate(request: EvaluateExperimentRequest) -> dict:
    try:
        return manager.evaluate_experiment(
            request.run_id,
            [row.model_dump() for row in request.judgments],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
