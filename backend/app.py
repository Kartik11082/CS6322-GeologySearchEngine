import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Add indexer/src to python path so we can import SearchEngine
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
