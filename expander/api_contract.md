# Query Expander — API Contract

This document outlines the API contract for the Query Expansion and Relevance Feedback service (Student 5). It details the endpoint used to expand user queries and fetch newly reranked results based on those expansions.

## Endpoint

**`POST /api/expand`**

*(Note: This endpoint is hosted on the same FastAPI backend server as the search engine, typically `http://127.0.0.1:8000`)*

## Request Schema (Input)

**Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | **Yes** | - | The initial query string to expand (e.g., "volcanic eruption"). Must not be empty. |
| `method` | `string` | No | `"association"` | The query expansion algorithm to use. <br>Enum: `["rocchio", "association", "scalar", "metric"]`. |
| `top_k` | `integer` | No | `10` | The number of top results to return for the *final* expanded search. |
| `search_method` | `string` | No | `"hits"` | Ranker for the final search on `expanded_query` (same enum as `/api/search`, e.g. `hits`, `tfidf`). |
| `relevant_doc_ids` | `array of strings` | No | `[]` | Used **only** when `method` is `"rocchio"`. A list of document IDs the user explicitly marked as relevant. |
| `irrelevant_doc_ids` | `array of strings` | No | `[]` | Used **only** when `method` is `"rocchio"`. A list of document IDs the user explicitly marked as non-relevant. |

### Example Request Body (Rocchio):
```json
{
  "query": "magma composition",
  "method": "rocchio",
  "top_k": 5,
  "relevant_doc_ids": ["14", "45"],
  "irrelevant_doc_ids": ["218"]
}
```

### Example Request Body (Local Clustering):
```json
{
  "query": "tectonic plate boundaries",
  "method": "metric",
  "top_k": 10
}
```

---

## Response Schema (Output)

**Content-Type:** `application/json`
**HTTP Status:** `200 OK`

The response includes the newly expanded query string along with the search results executed against that new query.

| Field | Type | Description |
|---|---|---|
| `status` | `string` | Represents the result status, e.g., `"success"`. |
| `original_query` | `string` | The query string originally provided in the request. |
| `expanded_query` | `string` | The new, automatically expanded query string. |
| `metadata` | `object` | Information about the query execution. |
| `metadata.total_results` | `integer` | Total number of documents retrieved by the expanded query. |
| `metadata.execution_time_ms` | `float` | Server-time taken to expand the query and run the search in ms. |
| `results` | `array of objects` | The search results (identical schema to standard search endpoint) retrieved using the `expanded_query`. |

### Example Response Body:
```json
{
  "status": "success",
  "original_query": "magma composition",
  "expanded_query": "magma composition silica basalt viscosity",
  "metadata": {
    "total_results": 87,
    "execution_time_ms": 45.2
  },
  "results": [
    {
      "rank": 1,
      "doc_id": 45,
      "score": 5.1204,
      "url": "[https://www.usgs.gov/volcanoes/magma](https://www.usgs.gov/volcanoes/magma)",
      "title": "Magma Types | U.S. Geological Survey",
      "snippet": "Magma is a complex mixture of molten rock... Silica content strongly influences magma viscosity..."
    }
  ]
}
```

---

## Error Schema

**Content-Type:** `application/json`
**HTTP Status:** `422 Unprocessable Entity` (for validation errors) or `500 Internal Server Error`

| Field | Type | Description |
|---|---|---|
| `detail` | `string/array` | A human-readable description of the error. |

### Example Error Response:
```json
{
  "detail": "Invalid expansion method: neural_net"
}
```