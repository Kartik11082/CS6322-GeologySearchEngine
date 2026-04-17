# API Contract

Base URL: `http://localhost:8010`

The service builds clustering artifacts on server startup. The frontend should
wait until `/v1/health` reports a ready `current_build_id`, then send search
requests to `/v1/rerank`.

## 1. Health

`GET /v1/health`

Response:

```json
{
  "status": "ok",
  "current_build_id": "ae290aa71916",
  "startup_build": {
    "status": "loaded_existing"
  }
}
```

`startup_build.status` values:

- `not_started`
- `running`
- `loaded_existing`
- `built_on_startup`
- `failed`

## 2. Rerank Search Results

`POST /v1/rerank`

Request body:

```json
{
  "query": "groundwater aquifer types",
  "cluster_method": "ward",
  "baseline_method": "combined",
  "top_k": 10,
  "search_adapter": {
    "url": "http://127.0.0.1:8000/api/search",
    "http_method": "GET",
    "results_path": ["results"]
  }
}
```

Fields:

- `query`: user search text
- `cluster_method`: `flat`, `ward`, or `complete`
- `baseline_method`: forwarded to the external search API
- `top_k`: number of baseline docs to fetch and rerank
- `search_adapter`: optional override for the upstream search API

Response:

```json
{
  "build_id": "ae290aa71916",
  "baseline_method": "combined",
  "cluster_method": "ward",
  "query": "groundwater aquifer types",
  "method": "ward",
  "clusters": [
    {
      "id": "3",
      "name": "groundwater, aquifer, hydrogeology",
      "size": 412,
      "result_count": 4,
      "representatives": [
        {
          "url": "https://example.com/doc",
          "normalized_url": "https://example.com/doc",
          "title": "Aquifer Types",
          "domain": "example.com",
          "similarity": 0.9123
        }
      ]
    }
  ],
  "baseline": [
    {
      "rank": 1,
      "title": "Aquifer Types",
      "url": "https://example.com/doc",
      "normalized_url": "https://example.com/doc",
      "snippet": "...",
      "score": 0.91,
      "cluster_id": "3",
      "cluster_name": "groundwater, aquifer, hydrogeology"
    }
  ],
  "reranked": [
    {
      "rank": 1,
      "baseline_rank": 3,
      "rank_delta": 2,
      "title": "Aquifer Types",
      "url": "https://example.com/doc",
      "normalized_url": "https://example.com/doc",
      "snippet": "...",
      "baseline_score": 0.72,
      "baseline_score_normalized": 0.81,
      "cluster_affinity": 0.88,
      "cluster_support": 0.64,
      "score": 0.801,
      "cluster_id": "3",
      "cluster_name": "groundwater, aquifer, hydrogeology"
    }
  ],
  "explanations": {
    "weights": {
      "baseline": 0.7,
      "cluster_affinity": 0.2,
      "cluster_support": 0.1
    }
  }
}
```

Frontend use:

- render `reranked` as the clustered result list
- use `clusters` for sidebar/group metadata
- use `rank_delta`, `cluster_name`, and `explanations.weights` for debug UI if needed

## 3. Cluster Catalog

`GET /v1/clusters/{method}`

Example:

`GET /v1/clusters/flat`

Optional query param:

- `build_id`

Response:

```json
{
  "build_id": "ae290aa71916",
  "method": "flat",
  "selected_k": 20,
  "clusters": [
    {
      "id": "0",
      "name": "volcano, lava, eruption",
      "size": 820,
      "representatives": [
        {
          "url": "https://example.com/volcano",
          "normalized_url": "https://example.com/volcano",
          "title": "Volcano Basics",
          "domain": "example.com",
          "similarity": 0.9011
        }
      ]
    }
  ]
}
```

## 4. Run 50-Query Experiment

`POST /v1/experiments/run`

Request body:

```json
{
  "baseline_method": "combined",
  "top_k": 10,
  "search_adapter": {
    "url": "http://127.0.0.1:8000/api/search",
    "http_method": "GET",
    "results_path": ["results"]
  }
}
```

Response:

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "queued",
  "created_at": "2026-04-17T22:22:45.955130+00:00",
  "updated_at": "2026-04-17T22:22:45.955130+00:00"
}
```

Then poll:

`GET /v1/experiments/{run_id}`

Completed response contains:

- `summary`
- `per_query`
- optional `judged`

## 5. Judgment Template

`POST /v1/experiments/judgment-template`

Request body:

```json
{
  "run_id": "a1b2c3d4e5f6"
}
```

Response contains:

- `query_count`
- `rows`
- `csv_path`

## 6. Evaluate with Manual Judgments

`POST /v1/experiments/evaluate`

Request body:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "judgments": [
    {
      "query_id": "Q044",
      "url": "https://example.com/aquifer-types",
      "relevance": 2,
      "notes": ""
    }
  ]
}
```

Response contains:

- `evaluated_queries`
- `summary`
- `per_query`

## 7. Manual Rebuild

`POST /v1/build`

Normally the frontend should not call this. This is only for manual rebuilds.

Request body:

```json
{
  "make_current": true,
  "search_adapter": {
    "url": "http://127.0.0.1:8000/api/search",
    "http_method": "GET",
    "results_path": ["results"]
  }
}
```

Poll status with:

- `GET /v1/build/{build_id}`
- `GET /v1/build/current`

## Search Adapter Rules

The default upstream search endpoint is `http://localhost:8000/api/search`.

If the upstream search API is the current `search_engine/main.py`, use:

```json
{
  "url": "http://127.0.0.1:8000/api/search",
  "http_method": "GET",
  "results_path": ["results"]
}
```

Do not use `POST` unless the upstream search API actually supports it.
