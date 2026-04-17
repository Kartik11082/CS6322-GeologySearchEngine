# Cluster Service

Standalone FastAPI service for:

- building flat and agglomerative clusters from `crawler_v2/output`
- reranking search results with cluster-aware signals
- running the 50-query benchmark
- generating judgment templates and judged IR metrics

The service now builds once during server startup if no completed build exists yet.
If a completed build already exists, startup loads that build and serves requests
immediately.

## Run

```powershell
pip install -r cluster_service\requirements.txt
python -m cluster_service
```

The server starts on `http://localhost:8010`.

On first startup, the process can take a while because clustering is built before
the API begins serving requests. After that, restarts reuse the saved build under
`cluster_service/output/builds/`.

## Main Endpoints

- `POST /v1/build`
- `GET /v1/build/{build_id}`
- `POST /v1/rerank`
- `GET /v1/clusters/{method}`
- `POST /v1/experiments/run`
- `GET /v1/experiments/{run_id}`
- `POST /v1/experiments/judgment-template`
- `POST /v1/experiments/evaluate`

## Notes

- The canonical 50-query benchmark lives in [queries_50.json](/C:/Users/karke/OneDrive/Desktop/UTD/IR/project/GeologySearchEngine-CS6322/cluster_service/benchmarks/queries_50.json).
- Builds and experiment artifacts are written under `cluster_service/output/`.
- `POST /v1/build` is still available for manual rebuilds, but normal usage does
  not require calling it from the frontend.
- The external search engine is not imported directly. The service calls a configurable HTTP search endpoint and reranks the returned documents.
