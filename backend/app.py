import sys
import time
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading search engine (this may take ~5-15 seconds)...")
    try:
        engine.load()
        print("Engine loaded successfully! Ready for blazing fast searches.")
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Please ensure you have built the index first using: python indexer/src/search.py --build")
    yield
    print("Shutting down engine...")

app = FastAPI(
    title="Geology Search Engine API",
    description="Backend API wrapper for the IR assignment search engine.",
    version="1.0.0",
    lifespan=lifespan
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
    query: str = Field(..., description="The query string to search for. Must not be empty.")
    method: str = Field("bm25", description="The ranking algorithm to use (tfidf, bm25, pagerank, hits).")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")


@app.post("/api/search")
async def perform_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    valid_methods = {"tfidf", "bm25", "pagerank", "hits"}
    if req.method not in valid_methods:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid method '{req.method}'. Must be one of: {', '.join(valid_methods)}."
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
            "execution_time_ms": execution_time_ms
        },
        "results": results
    }

class ExpandRequest(BaseModel):
    query: str = Field(..., description="The query string to expand.")
    method: str = Field("rocchio", description="Expansion method: rocchio, association, scalar, metric")
    top_k: int = Field(10, description="Number of results for the final search.")
    relevant_doc_ids: list[str] = Field(default_factory=list, description="For Rocchio: IDs of relevant docs.")
    irrelevant_doc_ids: list[str] = Field(default_factory=list, description="For Rocchio: IDs of non-relevant docs.")

@app.post("/api/expand")
async def perform_expansion(req: ExpandRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    expander = QueryExpander(engine)
    t0 = time.time()
    
    try:
        if req.method == "rocchio":
            # For Rocchio, we pass the explicit manual judgments
            expanded_query = expander.expand_rocchio(
                query=req.query, 
                relevant_doc_ids=req.relevant_doc_ids, 
                irrelevant_doc_ids=req.irrelevant_doc_ids
            )
        elif req.method == "association":
            expanded_query = expander.expand_association(req.query, normalized=False)
        elif req.method == "scalar":
            expanded_query = expander.expand_scalar(req.query)
        elif req.method == "metric":
            expanded_query = expander.expand_metric(req.query)
        else:
            raise HTTPException(status_code=422, detail=f"Invalid expansion method: {req.method}")
            
        # Run the final search with the newly expanded query
        results = engine.search(query=expanded_query, method="bm25", top_k=req.top_k)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    execution_time_ms = round((time.time() - t0) * 1000, 2)
    
    return {
        "status": "success",
        "original_query": req.query,
        "expanded_query": expanded_query,
        "metadata": {
            "total_results": len(results),
            "execution_time_ms": execution_time_ms
        },
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
