import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

# Add indexer/src to python path so we can import SearchEngine
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

sys.path.insert(0, str(PROJECT_ROOT / "expander"))
from core import QueryExpander


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search import SearchEngine

# Global singleton for the search engine
engine = SearchEngine()
query_expander: QueryExpander | None = None
DEBUG_LOG_PATH = None


def _engine_ready() -> bool:
    return bool(getattr(engine, "inverted_index", None)) and getattr(engine, "N", 0) > 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_expander
    print("Loading search engine (this may take ~5-15 seconds)...")
    engine.load()
    print("Engine loaded successfully! Ready for blazing fast searches.")
    query_expander = QueryExpander(engine)
    yield
    print("Shutting down engine...")


app = FastAPI(
    title="Geology Search Engine API",
    description="Backend API wrapper for the IR assignment search engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(
        ..., description="The query string to search for. Must not be empty."
    )
    method: str = Field(
        "hits",
        description="The ranking algorithm to use (tfidf, pagerank, hits, tfidf_pagerank, tfidf_hits).",
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")


@app.post("/api/search")
async def perform_search(req: SearchRequest):
    if not _engine_ready():
        raise HTTPException(
            status_code=503,
            detail="Search engine index is not loaded. Build and load the index first.",
        )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    valid_methods = {
        "tfidf",
        "pagerank",
        "hits",
        "tfidf_pagerank",
        "tfidf_hits",
    }
    if req.method not in valid_methods:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid method '{req.method}'. Must be one of: {', '.join(valid_methods)}.",
        )

    t0 = time.time()
    try:
        results = engine.search(query=req.query, method=req.method, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    execution_time_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "metadata": {
            "total_results": len(results),
            "execution_time_ms": execution_time_ms,
        },
        "results": results,
    }


ExpansionMethod = Literal["rocchio", "association", "scalar", "metric"]


class ExpandRequest(BaseModel):
    query: str = Field(..., description="The query string to expand.")
    method: ExpansionMethod = Field(
        "association",
        description="Expansion method: rocchio, association, scalar, metric.",
    )
    top_k: int = Field(
        10, ge=1, le=100, description="Number of results for the final search."
    )
    relevant_doc_ids: list[str] = Field(
        default_factory=list, description="For Rocchio: IDs of relevant docs."
    )
    irrelevant_doc_ids: list[str] = Field(
        default_factory=list, description="For Rocchio: IDs of non-relevant docs."
    )
    search_method: str = Field(
        default="hits", description="Ranking method for the final search after expansion."
    )


@app.post("/api/expand")
async def perform_expansion(req: ExpandRequest):
    if not _engine_ready() or query_expander is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine index is not loaded. Build and load the index first.",
        )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    expander = query_expander
    t0 = time.time()

    try:
        query_terms = len(req.query.split())
        m_neighbors = 2 if query_terms <= 3 else 6
        top_k_docs = 25

        if req.method == "rocchio":
            expanded_query = expander.expand_rocchio(
                query=req.query,
                relevant_doc_ids=req.relevant_doc_ids,
                irrelevant_doc_ids=req.irrelevant_doc_ids,
                alpha=1.0,
                beta=0.75,
                gamma=0.25,
                num_new_terms=10,
            )
        elif req.method == "association":
            expanded_query = expander.expand_association(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                normalized=True,
                max_new_terms=5,
            )
        elif req.method == "scalar":
            expanded_query = expander.expand_scalar(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                max_new_terms=5,
            )
        elif req.method == "metric":
            expanded_query = expander.expand_metric(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                max_new_terms=5,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid expansion method: {req.method}",
            )

        results = engine.search(query=expanded_query, method=req.search_method, top_k=req.top_k)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    execution_time_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "original_query": req.query,
        "expanded_query": expanded_query,
        "metadata": {
            "total_results": len(results),
            "execution_time_ms": execution_time_ms,
        },
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
