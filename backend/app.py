import sys
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path

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
DEBUG_LOG_PATH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading search engine (this may take ~5-15 seconds)...")
    try:
        engine.load()
        print("Engine loaded successfully! Ready for blazing fast searches.")
    except Exception as e:
        print(f"Error loading index: {e}")
        print(
            "Please ensure you have built the index first using: python indexer/src/search.py --build"
        )
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
        "bm25",
        description="The ranking algorithm to use (tfidf, bm25, pagerank, hits).",
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")


@app.post("/api/search")
async def perform_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    valid_methods = {
        "tfidf",
        "bm25",
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


class ExpandRequest(BaseModel):
    query: str = Field(..., description="The query string to expand.")
    method: str = Field(
        "rocchio", description="Expansion method: rocchio, association, scalar, metric"
    )
    top_k: int = Field(10, description="Number of results for the final search.")
    relevant_doc_ids: list[str] = Field(
        default_factory=list, description="For Rocchio: IDs of relevant docs."
    )
    irrelevant_doc_ids: list[str] = Field(
        default_factory=list, description="For Rocchio: IDs of non-relevant docs."
    )


@app.post("/api/expand")
async def perform_expansion(req: ExpandRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    # debug logging removed

    expander = QueryExpander(engine)
    t0 = time.time()

    try:
        # Tune expansion based on query length
        query_terms = len(req.query.split())
        # Short queries (1-2 terms) get fewer additions; longer queries can get more
        m_neighbors = 5 if query_terms <= 3 else 10
        top_k_docs = 50  # Local document set size for term correlation

        if req.method == "rocchio":
            # Explicit Relevance Feedback: user-marked relevant/irrelevant docs
            # debug logging removed
            expanded_query = expander.expand_rocchio(
                query=req.query,
                relevant_doc_ids=req.relevant_doc_ids,
                irrelevant_doc_ids=req.irrelevant_doc_ids,
                alpha=1.0,  # Weight of original query
                beta=0.75,  # Weight of relevant docs
                gamma=0.25,  # Weight of non-relevant docs (negative)
                num_new_terms=5,  # Max new terms to add
            )
        elif req.method == "association":
            # Pseudo-Relevance Feedback: co-occurrence in local docs
            expanded_query = expander.expand_association(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                normalized=True,  # Use normalized correlation (helps with frequency bias)
                max_new_terms=5,  # Max new terms to add
            )
        elif req.method == "scalar":
            # Pseudo-Relevance Feedback: cosine similarity of association vectors
            expanded_query = expander.expand_scalar(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                max_new_terms=5,  # Max new terms to add
            )
        elif req.method == "metric":
            # Pseudo-Relevance Feedback: physical word proximity in local docs
            expanded_query = expander.expand_metric(
                query=req.query,
                top_k_docs=top_k_docs,
                m_neighbors=m_neighbors,
                max_new_terms=5,  # Metric is stricter; fewer terms
            )
        else:
            raise HTTPException(
                status_code=422, detail=f"Invalid expansion method: {req.method}"
            )

        # Run the final search with the newly expanded query
        results = engine.search(query=expanded_query, method="bm25", top_k=req.top_k)
        # debug logging removed

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
