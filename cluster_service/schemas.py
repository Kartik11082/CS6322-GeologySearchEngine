from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchAdapterConfigModel(BaseModel):
    url: str | None = None
    http_method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float = 20.0
    query_field: str = "q"
    method_field: str = "method"
    top_k_field: str = "top_k"
    results_path: list[str] = Field(default_factory=lambda: ["results"])
    title_field: str = "title"
    url_field: str = "url"
    snippet_field: str = "snippet"
    score_field: str = "score"


class BuildRequest(BaseModel):
    search_adapter: SearchAdapterConfigModel | None = None
    make_current: bool = True


class RerankRequest(BaseModel):
    query: str
    cluster_method: Literal["flat", "ward", "complete"] = "flat"
    baseline_method: str = "combined"
    top_k: int = Field(default=10, ge=1, le=50)
    build_id: str | None = None
    search_adapter: SearchAdapterConfigModel | None = None


class ExperimentRunRequest(BaseModel):
    build_id: str | None = None
    baseline_method: str = "combined"
    top_k: int = Field(default=10, ge=1, le=20)
    search_adapter: SearchAdapterConfigModel | None = None


class JudgmentTemplateRequest(BaseModel):
    run_id: str


class JudgmentInput(BaseModel):
    query_id: str
    url: str
    relevance: int = Field(ge=0, le=2)
    notes: str = ""


class EvaluateExperimentRequest(BaseModel):
    run_id: str
    judgments: list[JudgmentInput]


class JobResponse(BaseModel):
    id: str
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)
