# Indexer — Geology Search Engine (Student 2)

This module takes the raw crawled pages from Student 1 and makes them **searchable** using an inverted index, two relevance models, topic-specific PageRank, and HITS.

---

## Table of Contents

- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Module Reference](#module-reference)
  - [config.py](#configpy)
  - [loader.py](#loaderpy)
  - [preprocessor.py](#preprocessorpy)
  - [index.py](#indexpy)
  - [relevance.py](#relevancepy)
  - [graph.py](#graphpy)
  - [search.py](#searchpy--main-entry-point)
- [Data Formats](#data-formats)
- [How to Run](#how-to-run)
- [Python API for Integration](#python-api-for-integration)
- [Configuration & Tuning](#configuration--tuning)

---

## Architecture

```
indexer/
├── src/
│   ├── config.py          # Paths, constants (BM25 k1/b, PageRank damping, etc.)
│   ├── loader.py          # Reads gzipped JSONL batch files from the crawler
│   ├── preprocessor.py    # Tokenize → stopword removal → Porter stemming
│   ├── index.py           # Builds inverted index + doc store
│   ├── relevance.py       # TF-IDF cosine similarity + Okapi BM25
│   ├── graph.py           # Web graph + Topic-specific PageRank + HITS
│   └── search.py          # Unified SearchEngine class (CLI + importable API)
├── data/                  # Generated at build time (git-ignored)
│   ├── inverted_index.json
│   ├── doc_store.json
│   ├── web_graph.json
│   ├── graph_stats.json
│   └── pagerank_scores.json
├── requirements.txt       # nltk, numpy
└── README.md              # This file
```

---

## Data Flow

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Crawler Output  │────▶│   loader.py   │────▶│ preprocessor.py │────▶│   index.py    │
│ (pages/edges     │     │ reads .jsonl.gz│    │ tokenize, stem  │     │ inverted index│
│  batch files)    │     └──────────────┘     └────────────────┘     │ + doc store   │
└─────────────────┘              │                                    └──────┬───────┘
                                 │                                           │
                                 ▼                                           ▼
                          ┌──────────────┐                            ┌──────────────┐
                          │   graph.py    │                            │ relevance.py  │
                          │ web graph +   │                            │ TF-IDF + BM25 │
                          │ PageRank +    │                            └──────┬───────┘
                          │ HITS          │                                   │
                          └──────┬───────┘                                   │
                                 │                                           │
                                 └─────────────┬─────────────────────────────┘
                                               ▼
                                        ┌──────────────┐
                                        │  search.py    │
                                        │ unified API   │
                                        │ (4 methods)   │
                                        └──────────────┘
```

---

## Module Reference

### `config.py`

Defines all paths and algorithm hyperparameters. **No functions to call — import constants directly.**

| Constant | Value | Description |
|----------|-------|-------------|
| `CRAWLER_BATCHES_DIR` | `../crawler/data/batches/` | Where to find crawler output |
| `INDEXER_DATA_DIR` | `./data/` | Where generated files are saved |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `PAGERANK_DAMPING` | `0.85` | Probability of following a link (vs. teleporting) |
| `PAGERANK_MAX_ITER` | `100` | Maximum PageRank iterations |
| `PAGERANK_TOL` | `1e-6` | Convergence threshold for PageRank |
| `HITS_MAX_ITER` | `50` | Maximum HITS iterations |
| `HITS_TOL` | `1e-6` | Convergence threshold for HITS |
| `HITS_BASE_SET_EXPANSION` | `50` | Max neighbors added per root-set node |
| `DEFAULT_TOP_K` | `10` | Default number of results |
| `SNIPPET_CHAR_LIMIT` | `300` | Max characters in result snippet |

---

### `loader.py`

Reads the crawler's gzipped JSONL batch files into memory.

| Function | Input | Output |
|----------|-------|--------|
| `load_pages()` | reads `pages_batch_*.jsonl.gz` | `list[dict]` — each dict has: `doc_id`, `title`, `clean_text`, `url`, `final_url`, `geology_score` |
| `load_edges()` | reads `edges_batch_*.jsonl.gz` | `list[dict]` — each dict has: `source_doc_id`, `source_url`, `target_url`, `anchor_text` |
| `build_url_to_docid(pages)` | list of page dicts | `dict[str, int]` — maps every URL (original + final/redirect) to its `doc_id` |

**Example page record:**
```json
{
  "doc_id": 1,
  "url": "https://www.usgs.gov/science/science-explorer/geology",
  "final_url": "https://www.usgs.gov/science/science-explorer/geology",
  "title": "Geology | U.S. Geological Survey",
  "clean_text": "Geology | U.S. Geological Survey Skip to main content...",
  "geology_score": 10
}
```

**Example edge record:**
```json
{
  "source_doc_id": 1,
  "source_url": "https://www.usgs.gov/science/science-explorer/geology",
  "target_url": "https://www.usgs.gov/natural-hazards",
  "anchor_text": "Natural Hazards"
}
```

---

### `preprocessor.py`

Three-step NLP pipeline applied to both documents (at index time) and queries (at search time).

| Step | Function | Example |
|------|----------|---------|
| 1. Tokenize | `tokenize(text)` | `"Geological formations"` → `["geological", "formations"]` |
| 2. Stopwords | `remove_stopwords(tokens)` | `["the", "geological", "formations"]` → `["geological", "formations"]` |
| 3. Stem | `stem(tokens)` | `["geological", "formations"]` → `["geolog", "format"]` |
| **Full pipeline** | `preprocess(text)` | `"Geological formations include rocks"` → `["geolog", "format", "includ", "rock"]` |

- Uses **NLTK Porter Stemmer** and English stopword list
- Tokenization regex: `[a-z0-9]+` (lowercase alphanumeric only)
- Downloads NLTK data lazily on first import (silent)

---

### `index.py`

Builds and persists the inverted index.

#### `build_index(pages) → (inverted_index, doc_store, N, avg_dl)`

For each page:
1. Combines `title` + `clean_text` into a single string
2. Runs `preprocess()` to get a list of stems
3. Counts term frequencies with `collections.Counter`
4. Stores postings in the inverted index

**Inverted index structure:**
```
{
  "earthquak": { "1": 2, "2": 15, "3": 2, ... },   // term → {doc_id: tf}
  "fault":     { "1": 5, "2": 8,  "3": 1, ... },
  ...
}
```

**Doc store structure:**
```
{
  "1": {
    "url": "https://...",
    "final_url": "https://...",
    "title": "Geology | U.S. Geological Survey",
    "clean_text_preview": "first 500 chars...",
    "doc_len": 518,          // number of stems in the document
    "geology_score": 10      // from crawler's quality filter
  }
}
```

**Key values:**
- `N` = total number of documents
- `avg_dl` = average document length (in stems), needed by BM25

#### Persistence
| Function | What it does |
|----------|-------------|
| `save_index(...)` | Writes `inverted_index.json` and `doc_store.json` to disk |
| `load_index()` | Reads them back; returns `(inverted_index, doc_store, N, avg_dl)` |

**On-disk format** (`inverted_index.json`):
```json
{
  "N": 237,
  "avg_dl": 2592.2,
  "index": { "earthquak": {"1": 2, "2": 15}, ... }
}
```

---

### `relevance.py`

Two ranking models. Both accept a raw query string, preprocess it, then score documents from the inverted index.

#### Model 1: TF-IDF Cosine Similarity

```
rank_tfidf(query, inverted_index, doc_store, N, top_k=10)
  → list[(doc_id_str, score)]
```

**Algorithm:**
```
TF(t, d)  = 1 + log₁₀(tf)          if tf > 0, else 0
IDF(t)    = log₁₀(N / df)           where df = number of docs containing term t
Score     = cosine_similarity(query_tfidf_vector, doc_tfidf_vector)
```

Cosine normalization divides by `||query|| × ||doc||` to handle varying document lengths.

#### Model 2: Okapi BM25

```
rank_bm25(query, inverted_index, doc_store, N, avg_dl, top_k=10, k1=1.2, b=0.75)
  → list[(doc_id_str, score)]
```

**Algorithm:**
```
IDF(t)    = log((N - df + 0.5) / (df + 0.5) + 1)
Score(q,d) = Σ  IDF(t) × [tf × (k1 + 1)] / [tf + k1 × (1 - b + b × dl/avg_dl)]
```

- **k1 = 1.2**: Controls term frequency saturation. Higher → more weight to repeated terms.
- **b = 0.75**: Controls length normalization. Higher → more penalty for long documents.
- BM25 is generally more effective than TF-IDF because it normalizes for document length.

**Both models return:** `list[tuple[str, float]]` — list of `(doc_id_string, score)` sorted descending.

---

### `graph.py`

Three components: web graph construction, topic-specific PageRank, and HITS.

#### `WebGraph` class

```python
graph = WebGraph.build_from_data(pages, edges, url_to_docid)
```

- Builds a **directed graph** where nodes = `doc_id`s and edges = hyperlinks
- Only keeps edges where **both** source and target are in the crawled set
- Resolves `target_url` to `doc_id` using the URL mapping from `loader.py`
- Skips self-links
- Stores adjacency as `out_links: dict[int, set[int]]` and `in_links: dict[int, set[int]]`

| Method | Returns |
|--------|---------|
| `graph.stats()` | `dict` with `num_nodes`, `num_edges`, `max_in_degree`, `max_out_degree`, `avg_out_degree` |
| `graph.save()` | Writes `web_graph.json` and `graph_stats.json` |
| `WebGraph.load()` | Reads graph back from JSON |

#### Topic-specific PageRank

```python
scores = topic_pagerank(graph, doc_store, damping=0.85, max_iter=100, tol=1e-6)
  → dict[int, float]   # doc_id → PageRank score
```

**How it differs from standard PageRank:**
- Standard PageRank: random surfer teleports to any page with equal probability
- **Topic-specific**: teleport probability is **proportional to `geology_score`**
- Pages with higher geology relevance get more teleport probability
- This biases the ranking toward geology-relevant authoritative pages

**Algorithm:**
```
PR(new) = d × (link_contributions) + (1-d) × teleport_vector
teleport[i] = geology_score[i] / sum(all_scores)
```

Dangling nodes (no outlinks) distribute their rank evenly across all nodes.

#### HITS (Hyperlink-Induced Topic Search)

```python
authority_ranking, hub_ranking = hits(query, graph, inverted_index, top_k=10)
  → (list[(doc_id, auth_score)], list[(doc_id, hub_score)])
```

**HITS is query-dependent** (unlike PageRank which is precomputed once).

**Algorithm:**
1. **Root set**: Find all documents containing any query term (via inverted index)
2. **Base set**: Expand by adding up to 50 pages that link to/from each root-set page
3. **Iterate**:
   - `authority(p) = Σ hub(q)` for all pages q that link TO p
   - `hub(p) = Σ authority(q)` for all pages q that p links TO
   - Normalize both vectors by L2 norm
4. Stop when converged (delta < 1e-6) or after 50 iterations

Returns authority scores (good content pages) and hub scores (good navigation pages).

---

### `search.py` — Main entry point

The `SearchEngine` class wraps everything into a single interface.

#### Lifecycle

```python
engine = SearchEngine()

# Option A: Build from scratch (reads crawler batches, takes ~5s for 237 pages)
engine.build()

# Option B: Load pre-built data from disk (instant)
engine.load()
```

#### `engine.search(query, method, top_k) → list[dict]`

| Method | How it works |
|--------|-------------|
| `"tfidf"` | Calls `rank_tfidf()` directly |
| `"bm25"` | Calls `rank_bm25()` directly |
| `"pagerank"` | Runs BM25 to get 5× candidates, then re-ranks with `0.7 × norm_bm25 + 0.3 × pagerank × 1000` |
| `"hits"` | Runs HITS on the query's subgraph, returns by authority score |

**Return format:**
```python
[
    {
        "rank": 1,
        "doc_id": 14,
        "score": 4.2931,
        "url": "https://www.usgs.gov/mission-areas/natural-hazards",
        "title": "Natural Hazards | U.S. Geological Survey",
        "snippet": "Natural Hazards | U.S. Geological Survey Skip to main content..."
    },
    ...
]
```

---

## Data Formats

### Input (from Crawler — Student 1)

| File pattern | Format | Contents |
|-------------|--------|----------|
| `pages_batch_NNNN.jsonl.gz` | Gzipped JSON Lines | One JSON object per line: `{doc_id, url, final_url, title, clean_text, geology_score}` |
| `edges_batch_NNNN.jsonl.gz` | Gzipped JSON Lines | One JSON object per line: `{source_doc_id, source_url, target_url, anchor_text}` |

### Output (generated by Indexer)

| File | Format | Size (237 pages) |
|------|--------|-----------------|
| `inverted_index.json` | JSON | ~1.3 MB |
| `doc_store.json` | JSON | ~186 KB |
| `web_graph.json` | JSON | ~837 KB |
| `graph_stats.json` | JSON | ~185 B |
| `pagerank_scores.json` | JSON | ~6.9 KB |

All output files are in `indexer/data/` and are **git-ignored**. They are regenerated by running `python src/search.py --build`.

---

## How to Run

### Prerequisites

```bash
# From the project root, with the venv activated:
uv pip install -r indexer/requirements.txt   # installs nltk, numpy
```

### Build the index (required once, or when new batches arrive)

```bash
cd indexer
python src/search.py --build
```

**What it does:**
1. Reads all `pages_batch_*.jsonl.gz` and `edges_batch_*.jsonl.gz` from `crawler/data/batches/`
2. Preprocesses all page text (tokenize, stopwords, stem)
3. Builds the inverted index and doc store
4. Constructs the web graph from edge data
5. Computes topic-specific PageRank
6. Saves everything to `indexer/data/`

### Search from the command line

```bash
python src/search.py -q "earthquake fault" -m bm25 -k 10
```

| Flag | Description | Default |
|------|-------------|---------|
| `--build` | Build index from scratch before searching | off |
| `-q` / `--query` | The search query | *(required for search)* |
| `-m` / `--method` | `tfidf`, `bm25`, `pagerank`, or `hits` | `bm25` |
| `-k` / `--top` | Number of results to return | `10` |

### Examples

```bash
python src/search.py -q "volcanic eruption" -m tfidf -k 5
python src/search.py -q "sedimentary rock" -m bm25
python src/search.py -q "mineral deposits" -m pagerank
python src/search.py -q "plate tectonics" -m hits
```

---

## Python API for Integration

This is how Student 3 (frontend) or any other module connects to the search engine:

```python
import sys
sys.path.insert(0, "indexer/src")

from search import SearchEngine

engine = SearchEngine()
engine.load()  # loads pre-built index, graph, PageRank from disk

# Search with any method
results = engine.search("earthquake fault", method="bm25", top_k=10)

for r in results:
    print(f"#{r['rank']} [{r['score']:.4f}] {r['title']}")
    print(f"  {r['url']}")
    print(f"  {r['snippet'][:100]}...")
```

The FastAPI wrapper in `frontend/app.py` does exactly this to serve the Next.js frontend.

---

## Configuration & Tuning

All hyperparameters are in `config.py`. To tune:

| Parameter | Effect of increasing |
|-----------|---------------------|
| `BM25_K1` (1.2) | More weight to repeated terms |
| `BM25_B` (0.75) | More penalty for long documents |
| `PAGERANK_DAMPING` (0.85) | Less teleport, more link-following |
| `HITS_BASE_SET_EXPANSION` (50) | Larger subgraph for HITS (slower but more thorough) |
| `SNIPPET_CHAR_LIMIT` (300) | Longer/shorter snippets in results |

After changing parameters, re-run `python src/search.py --build` to regenerate the data.
