# Geology Search Engine — API Contract

This document outlines the API contract for the backend search service. It details the endpoint, request schema, response schema, and includes sample requests and responses. This contract can be directly used to implement the FastAPI wrapper.

## Endpoint

**`POST /api/search`**

_(Note: While `GET /search?q=...` is typical for simple search endpoints, `POST` is highly recommended when using FastAPI as it cleanly allows Pydantic to validate complex schemas or extended body parameters without messing up the URI length.)_

## Request Schema (Input)

**Content-Type:** `application/json`

| Field    | Type      | Required | Default   | Description                                                                                              |
| -------- | --------- | -------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `query`  | `string`  | **Yes**  | -         | The query string to search for (e.g., "earthquake fault lines"). Must not be empty.                      |
| `method` | `string`  | No       | `"tfidf"` | The ranking algorithm to use. <br>Enum: `["tfidf", "pagerank", "hits", "tfidf_pagerank", "tfidf_hits"]`. |
| `top_k`  | `integer` | No       | `10`      | The number of top results to return. <br>Range: `1` to `100`.                                            |

### Example Request Body:

```json
{
  "query": "volcanic activity in the pacific ring of fire",
  "method": "pagerank",
  "top_k": 5
}
```

---

## Response Schema (Output)

**Content-Type:** `application/json`
**HTTP Status:** `200 OK`

The response will contain metadata about the search (time taken, total results found, etc.) and an array of the ranked result objects matching the inner output of `search.py`.

| Field                        | Type               | Description                                                          |
| ---------------------------- | ------------------ | -------------------------------------------------------------------- |
| `status`                     | `string`           | Represents the result status, e.g., `"success"`.                     |
| `metadata`                   | `object`           | Information about the query execution.                               |
| `metadata.total_results`     | `integer`          | Total number of relevant documents found containing the query terms. |
| `metadata.execution_time_ms` | `float`            | Server-time taken to process the query in milliseconds.              |
| `results`                    | `array of objects` | The search results sorted by rank (descending score/relevance).      |

### Result Object Schema (`results` array)

| Field     | Type      | Description                                                                                          |
| --------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `rank`    | `integer` | The 1-based index/rank of the result in the retrieved list (1 is the best).                          |
| `doc_id`  | `integer` | The internal unique identifier of the document.                                                      |
| `score`   | `float`   | The actual relevance score assigned by the selected `method`.                                        |
| `url`     | `string`  | The fully qualified URL of the page.                                                                 |
| `title`   | `string`  | The extracted HTML title of the page.                                                                |
| `snippet` | `string`  | A dynamically generated relevant excerpt (up to 300 characters, per `config.py`) from the page text. |

### Example Response Body:

```json
{
  "status": "success",
  "metadata": {
    "total_results": 142,
    "execution_time_ms": 15.4
  },
  "results": [
    {
      "rank": 1,
      "doc_id": 14,
      "score": 4.2931,
      "url": "https://www.usgs.gov/mission-areas/natural-hazards",
      "title": "Natural Hazards | U.S. Geological Survey",
      "snippet": "Natural Hazards | U.S. Geological Survey Skip to main content. Earthquakes, tsunamis, and volcanic eruptions are primary examples of natural hazards studied by our teams..."
    },
    {
      "rank": 2,
      "doc_id": 218,
      "score": 3.9812,
      "url": "https://www.usgs.gov/observatories/hvo",
      "title": "Hawaiian Volcano Observatory | U.S. Geological Survey",
      "snippet": "Hawaiian Volcano Observatory monitors active volcanoes in Hawaii. We assess hazards, issue warnings, and advance scientific understanding to reduce impacts of volcanic activity..."
    }
  ]
}
```

---

## Error Schema

If an invalid parameter is passed or an internal failure occurs, an error payload is returned.

**Content-Type:** `application/json`
**HTTP Status:** `422 Unprocessable Entity` (for validation errors) or `500 Internal Server Error`

| Field    | Type           | Description                                                                          |
| -------- | -------------- | ------------------------------------------------------------------------------------ |
| `detail` | `string/array` | A human-readable description of what went wrong, or Pydantic validation error lists. |

### Example Error Response:

```json
{
  "detail": "Invalid method 'neural'. Must be one of: tfidf, pagerank, hits, tfidf_pagerank, tfidf_hits, bm25."
}
```
