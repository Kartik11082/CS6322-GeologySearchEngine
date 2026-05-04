# CS 6322 — Geology Search Engine

A full information retrieval pipeline specialized for geosciences content. Built as a 5-student class project for CS 6322 (Information Retrieval) at UTD, Spring 2026.

> **Project Title:** Search engine for **Geology / Earth Sciences**
> **Course:** CS 6322 — Information Retrieval · UTD · Spring 2026
> **Modules:** Crawler · Indexer + Relevance · User Interface · Clustering · Query Expansion

---

## Team

| Role | Name | NetID | Directory |
|------|------|-------|-----------|
| Crawler | Zafeer Rangoonwala | zxr240004 | `crawler/` |
| Indexer & Relevance Models | Rahul Patil | rxp240025 | `indexer/src/` |
| User Interface | Kartik Karkera | KXK230091 | `search-engine/frontend/`, `search-engine/backend_proxy/` |
| Clustering | Preeti Vasaikar | pxv230036 | `cluster_service/` |
| Query Expansion | Uddesh Singh | uxs230004 | `expander/` |

---

## Table of Contents

1. [Part 1 — Report Answers](#part-1--report-answers)
   - [1.1 The Problem & Architecture](#11-the-problem--architecture)
   - [1.2 Crawling](#12-crawling)
   - [1.3 Indexing & Relevance](#13-indexing--relevance)
   - [1.4 User Interface & Comparisons](#14-user-interface--comparisons-with-google--bing)
   - [1.5 Clustering](#15-clustering)
   - [1.6 Query Expansion & Relevance Feedback](#16-query-expansion--relevance-feedback)
2. [Part 2 — Technical Flow](#part-2--technical-flow)
   - [2.1 The Big Picture](#21-the-big-picture)
   - [2.2 End-to-End Query Lifecycle](#22-end-to-end-lifecycle-of-a-query)
   - [2.3 Service Topology](#23-service-topology--ports--data-files)
   - [2.4 Data Pipeline](#24-data-pipeline-offline)
   - [2.5 How Modules Are Linked](#25-how-modules-are-linked)
   - [2.6 Running the Project](#26-running-the-project-end-to-end)

---

# Part 1 — Report Answers

This section maps directly to the rubric questions. Each subsection explains **what we did** and **why**.

---

## 1.1 The Problem & Architecture

**Goal.** Build a domain-focused search engine for geology / earth-science web content that supports keyword search, multiple ranking models, clustering, and query expansion — and lets us compare results against Google and Bing.

**Why a vertical (domain-specific) engine?** General-purpose engines optimize for popularity. A geology-focused crawler + index lets us:

- Aggressively filter non-geology content so the corpus is dense with relevant material.
- Run topic-specific PageRank that boosts pages with high geology relevance, not pages with the most links overall.
- Apply each IR concept (BM25, HITS, Rocchio, scalar/metric/associative expansion, Ward/flat/complete clustering) on a coherent, interpretable corpus.

**Architecture.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite, :5173)                                       │
│  search-engine/frontend/src/App.jsx                                   │
└──────────────────────────────┬────────────────────────────────────────┘
                               │  REST (/api/*)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Backend Proxy (FastAPI, :8020)                                       │
│  search-engine/backend_proxy/app.py                                   │
│  Orchestrates search → cluster → external comparison                  │
└──────┬─────────────────┬───────────────────────────┬──────────────────┘
       │                 │                           │
       ▼                 ▼                           ▼
┌────────────┐    ┌─────────────────┐        ┌──────────────┐
│ Search API │    │ Cluster Service │        │  SerpAPI     │
│ FastAPI    │    │ FastAPI :8010   │        │ (Google/Bing)│
│ :8000      │    │ cluster_service/│        └──────────────┘
│ backend/   │    └─────────────────┘
│  app.py    │
│            │
│ uses:      │
│  • indexer │
│  • expander│
└─────┬──────┘
      │
      ▼
┌──────────────────────────────────────────────────┐
│  Static index data (built offline)               │
│  indexer/data/  ← inverted_index.json            │
│                   doc_store.json                 │
│                   web_graph.json                 │
│                   pagerank_scores.json           │
└──────────────────────────────────────────────────┘
                ▲
                │ build step
┌───────────────┴──────────────────────────────────┐
│  Crawler (async BFS, geology-focused)            │
│  crawler/  →  crawler/output/pages.jsonl         │
│               crawler/output/web_graph.csv       │
└──────────────────────────────────────────────────┘
```

---

## 1.2 Crawling

> **Owner:** Zafeer Rangoonwala (zxr240004) · `crawler/`

### How pages were gathered

Async BFS crawler (`crawler/crawler/`) using `aiohttp` with a domain-aware frontier. Key parameters (`crawler/crawler/config.py`):

| Parameter | Value |
|-----------|-------|
| `CONCURRENCY` | 60 parallel workers |
| `DOMAIN_DELAY` | 1.0 s per host (politeness) |
| `MAX_DEPTH` | 10 hops from seed |
| `TARGET_PAGES` | 100,000 |
| `BLOOM_CAPACITY` | 5,000,000 |
| `BLOOM_ERROR_RATE` | 0.001 |

**Result: 105,730 pages crawled** (saved in `crawler/output/pages.jsonl`).

### Seed list

186 seed URLs across authoritative geoscience sources (`crawler/main.py`):

- **Government surveys:** USGS, British Geological Survey (bgs.ac.uk), NOAA, Natural Resources Canada, Geoscience Australia, GNS New Zealand, and regional surveys (Ireland, Norway, Finland, Sweden, Denmark, Maryland, Illinois, Kansas, California, Mississippi, Queensland)
- **Research institutions:** IRIS Seismology, LDEO, WHOI, MBARI, UNAVCO, GFZ Potsdam, Nagoya University, IGS China
- **Universities:** UT Austin, UC Berkeley, Stanford, Arizona, Harvard, Columbia, Utah, Wisconsin, Hawaii, Indiana, LSU, UMass, Cornell, Dartmouth, Colorado State, Montana, Kansas, New Mexico, Ohio State, Rutgers (19 total)
- **Specialty domains:** mindat.org (mineralogy), RRUFF (Raman/mineralogy), Smithsonian Global Volcanism Program, Paleobiology Database, SERC Carleton (education), GeoSciWorld (journals)
- **Topics covered:** volcanology, seismology, mineralogy, paleontology, sedimentary geology, hydrogeology, geophysics, geomorphology, geohazards, climate/earth systems

### Geology filter

Two-stage filter keeps the corpus domain-focused:

1. **URL keyword filter** — URL must contain at least one geology keyword (`geolog*`, `mineral`, `volcano`, `seismic`, `earthquake`, `tectonic`, `fossil`, `sediment`, `stratigraph`, `petrolog*`, `geophysic*`, `geochem*`, `igneous`, `metamorphic`, `magma`, `lava`, `fault`, etc.).
2. **Content keyword filter** (`CONTENT_KEYWORDS`) — on-page text must contain geology terms. This prevents navigation pages and blank landing pages from slipping through on URL alone.

### Deduplication (`crawler/crawler/dedup.py`)

Three layers:

1. **URL canonicalization** — lowercase scheme/host, strip trailing slashes, sort query parameters, drop tracking params (`utm_*`, `share=`, `ref=`).
2. **Bloom filter** — O(1) membership check for visited URLs. Memory-efficient at 5 M capacity with 0.001 false-positive rate.
3. **Content hash** (SHA-256 of normalized body text) — catches mirror sites and redirect pairs that canonicalization misses.

### Output files

| File | Format | Contents |
|------|--------|----------|
| `crawler/output/pages.jsonl` | JSONL | `{url, title, text, content_type, crawled_at, status, depth}` — 105,730 rows |
| `crawler/output/web_graph.csv` | CSV | `src_url, dst_url` edges |
| `crawler/output/crawl_stats.json` | JSON | Run summary |

### What we learned / difficulties

- **Politeness vs. throughput:** Per-domain delays are essential — without them, USGS and IRIS rate-limit the crawler immediately. The 1 s delay cuts throughput but keeps us from being blocked.
- **Bloom filter at scale:** A plain Python `set` works at 10 k URLs; at 100 k+ the memory and lookup overhead matters. Bloom keeps membership checks O(1) and the footprint constant.
- **Content hashing necessity:** Several geology education sites serve the same article under multiple URLs (e.g., course mirrors). Content hashing prevents duplicate documents from inflating TF scores.

---

## 1.3 Indexing & Relevance

> **Owner:** Rahul Patil (rxp240025) · `indexer/src/`

### Index construction

`indexer/src/index.py` builds the index in a parallel pipeline using `multiprocessing.Pool` over all CPU cores:

```
pages.jsonl ──► tokenize ──► remove stopwords ──► Porter stem
                                                        │
                                                        ▼
                                          inverted_index + doc_store
```

- **Tokenizer:** regex `[a-z0-9]+` (lowercase alphanumeric only).
- **Stopwords:** NLTK English list.
- **Stemmer:** PyStemmer (C-based Porter stemmer — ~10× faster than NLTK's pure-Python version).
- **Inverted index shape:** `{ stem: { doc_id: tf } }`.
- **Doc store:** stores URL, title, text length, and `geology_score` per document.

### Relevance models (`indexer/src/relevance.py`)

**TF-IDF cosine similarity**

```
TF(t, d)   = 1 + log₁₀(raw_tf)
IDF(t)     = log₁₀(N / df)
score(q,d) = cosine(query_vector, doc_vector)
```

Length normalization via cosine handles variance in document length. Used as a baseline to compare against BM25.

**Okapi BM25** (default ranking model)

```
IDF(t) = log((N − df + 0.5) / (df + 0.5) + 1)
BM25(q,d) = Σ IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 − b + b × |d|/avgdl))
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k1` | 1.2 | Term-frequency saturation — a single word repeated 100 times is not 100× more relevant |
| `b` | 0.75 | Length normalization — penalizes long documents that happen to mention a term many times |

BM25 outperforms TF-IDF on our benchmark because it saturates term frequency and handles the wide document-length variance in the crawl (short USGS fact-sheets vs. long GeoSciWorld articles).

### Topic-specific PageRank (`indexer/src/graph.py`)

Standard PageRank teleports uniformly to any node. We bias the teleport vector by `geology_score` so authority flows preferentially toward geology-focused pages:

```
teleport[i] = geology_score[i] / Σ geology_scores
PR_new      = d × (link contributions) + (1 − d) × teleport
```

| Parameter | Value |
|-----------|-------|
| `damping` | 0.85 |
| `max_iter` | 100 |
| `tol` | 1e-6 |

Top-PageRank pages in our corpus are USGS hub pages (Natural Hazards, Volcano Hazards Program, Earthquake Hazards) — exactly the authorities a geology search engine should surface.

### HITS (`indexer/src/graph.py`)

Query-dependent HITS: build root set from inverted-index hits → expand by up to 50 neighbours → iterate hub/authority scores to convergence.

| Parameter | Value |
|-----------|-------|
| `max_iter` | 50 |
| `tol` | 1e-6 |
| `base_set_expansion` | 50 neighbours |

For a query like _"plate tectonics"_, top authorities are usgs.gov tectonics pages and IRIS educational pages. HITS wins on broad authority-style queries where hub structure matters.

### Web graph statistics (`indexer/data/graph_stats.json`)

| Metric | Value |
|--------|-------|
| Nodes | 101,956 |
| Edges | 3,188,271 |
| Max in-degree | 21,343 (doc_id 1256) |
| Max out-degree | 149 (doc_id 184) |
| Avg out-degree | 31.27 |

The graph has more nodes than crawled pages because edges reference target URLs that were discovered but not necessarily crawled — every target gets a node so PageRank flows correctly toward un-crawled authorities.

### Output files (`indexer/data/`)

| File | Size | Contents |
|------|------|----------|
| `inverted_index.json` | 452 MB | `{ stem: { doc_id: tf } }` |
| `doc_store.json` | 79 MB | Per-doc metadata (URL, title, length, geology_score) |
| `web_graph.json` | 110 MB | Serialized directed graph |
| `pagerank_scores.json` | 3.2 MB | `{ doc_id: pr_score }` |
| `graph_stats.json` | ~200 B | Summary statistics |

### Collaboration

We agreed on a JSON contract (`indexer/api_contract.md`) so each component maps to a stable method string (`tfidf | bm25 | pagerank | hits`). The UI student generated 50 queries; we used those to A/B the rankings. The cluster service consumes our `(doc_id, score, snippet)` payload and produces a reordered list — we did not modify the relevance models themselves.

---

## 1.4 User Interface & Comparisons with Google & Bing

> **Owner:** Kartik Karkera (KXK230091) · `search-engine/frontend/`, `search-engine/backend_proxy/`

### 1.4.1 Interface design

**One-line summary.** A minimal, light-mode React SPA whose job is to make four ranking models, a clustered view, a query-expansion view, and live Google/Bing results understandable — at a glance, side by side.

**Two app states:**

1. **Landing** (`Landing` component) — serif wordmark, one-line tagline, pill search input, "About this project" link. Nothing competes with the query box.
2. **Results page** — three layers stacked:
   - Sticky **TopBar** — keeps the query editable at all times; three controls: `Top K` stepper (3–20), `Expand` method (`association | scalar | metric`), `Cluster` method (`flat | ward | complete`).
   - **TabBar** — four tabs with result-count badges: **Relevance Models · Clustered · Query Expansion · Compare Engines**.
   - **Pane content** swapping on the active tab.

**Result card anatomy** (`ResultCard` component):

```
┌─────┬───────────────────────────────────────────────┬─────────┐
│  #1 │ [Strong]  [cluster: Seismology]  ↑ +4         │ Score   │
│     │ usgs.gov                                      │ 4.293   │
│     │ Earthquake Hazards Program | USGS             │         │
│     │ "The U.S. Geological Survey monitors..."      │         │
└─────┴───────────────────────────────────────────────┴─────────┘
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔  (full-width score bar)
```

- **Match-strength badge** (`Strong / Moderate / Weak`) — non-numeric relevance read. Score ≥ 0.7 = Strong, ≥ 0.3 = Moderate, else Weak.
- **Cluster tag** — visible only on the Clustered tab.
- **Rank delta** (`↑ +N` / `↓ -N`) — colored pill showing movement vs. baseline rank after cluster reranking.
- **Score bar** — full-width, 2 px bottom border normalized to the **max score in the same panel** (honest within one ranker; not misleadingly comparable across rankers).
- **Per-pane attribution** (`PaneAttrib` component) — every pane shows the owner's name + NetID directly under the pane title.

**Why these choices:**

- **Tabs over combined view** — each module (relevance / cluster / expansion / external) has a different explanation story; mixing them obscures each component's contribution.
- **Skeleton cards** — panels load independently; the user sees the fastest model first rather than waiting on the slowest.
- **URL state sync** (`syncUrl`) — query, top-K, methods, and active tab are reflected in the URL for shareability and demo reproducibility.
- **Typography:** Inter for UI chrome, Fraunces (serif) for content text. Near-black on white, single dark accent. Color appears only on information-bearing elements (cluster dots, badges, overlap tags).

### 1.4.2 Working with the indexer student

**Decoupling principle.** The frontend never imports Python. The indexer student owns `SearchEngine`; the UI student owns the SPA + FastAPI proxy. We froze a JSON contract early so both could iterate independently.

**The contract** (`indexer/api_contract.md`):

```http
POST :8000/api/search
{ "query": "earthquake fault", "method": "bm25", "top_k": 10 }

→ { "query": "...", "method": "bm25",
    "results": [
      { "rank": 1, "doc_id": 14, "score": 4.2931,
        "url": "...", "display_url": "usgs.gov",
        "title": "...", "snippet": "..." }, ...
    ]
}
```

`method` is one of `bm25 | tfidf | pagerank | hits`. Result shape is identical across methods — the UI reuses the same `ResultCard` for all four.

**Request fan-out** — on every search submit, 8 requests fire in parallel:

| Endpoint | Purpose | Tab |
|----------|---------|-----|
| `POST /api/search` × 4 | One per method (BM25, TF-IDF, PageRank, HITS) | Relevance Models |
| `POST /api/expand` | Pseudo-relevance feedback expansion | Query Expansion |
| `POST /api/clustered-search` | Cluster reranking | Clustered |
| `POST /api/external-search` × 2 | Google + Bing via SerpAPI | Compare Engines |

A `requestIdRef` guard ensures stale responses from a previous query never overwrite a newer one.

**Why a proxy at :8020:**

1. **CORS** — keeps all requests same-origin (Vite proxies `/api/*` → `:8020`), works identically in dev and production.
2. **Orchestration** — `/api/clustered-search` must call `:8000` then `:8010`; that logic belongs server-side.
3. **Secret hiding** — `SERPAPI_KEY` lives only on the server, never in the JS bundle.

### 1.4.3 Number of queries used for testing

**Total: 50 benchmark queries** drawn from `cluster_service/benchmarks/queries_50.json`, split into:

| Source | Count | Purpose |
|--------|-------|---------|
| Co-authored with relevance-model student (Rahul) | 30 | Probe specific ranker behaviours: rare-term recall, BM25 vs TF-IDF divergence, link-heavy topics for PageRank, authoritative-hub queries for HITS |
| Generated independently (UI student) | 20 | Representative user-facing queries: single-term ("fault"), multi-word phrase ("metamorphic rock classification"), natural-language ("how do volcanoes form") |

**Why this split.** The 30 relevance-focused queries served the indexer team for tuning BM25/PageRank weights. The 20 UI queries cover edge cases the indexer queries don't: very short queries, ambiguous terms ("fault" = rock fault or generic fault), and conversational full-sentence queries.

**Scoring criteria (UI testing):**

1. Does BM25 return ≥ 1 obviously-relevant result in top-3?
2. Does Clustered produce ≥ 2 distinct, sensibly-named clusters?
3. Does Query Expansion add domain terms (not noise)?

### 1.4.4 Collaborating with the clustering student

**Service boundary.** Preeti's clustering service runs independently on `:8010`. We agreed it would **never replace** baseline ranking — it only **reorders within the top-k** that BM25 already produced. This protects retrieval quality.

**The contract** (proxy-side, forwarded to `:8010/v1/rerank`):

```json
{
  "reranked": [
    {
      "rank": 1, "baseline_rank": 3, "rank_delta": 2,
      "score": 0.84, "baseline_score": 0.71,
      "cluster_id": "c3", "cluster_name": "Seismology",
      "url": "...", "title": "...", "snippet": "..."
    }
  ],
  "clusters": [
    {
      "id": "c3", "name": "Seismology", "result_count": 4,
      "representatives": [{ "url": "...", "title": "...", "domain": "...", "similarity": 0.92 }]
    }
  ],
  "explanations": { "weights": { "baseline": 0.7, "cluster_affinity": 0.2, "cluster_support": 0.1 } }
}
```

Three fields I specifically requested to make the UI work:

1. **`baseline_rank` + `rank_delta`** — makes the effect of reranking visible (`↑ +N`). Without this, clustering is invisible.
2. **`representatives` per cluster** — lets the active cluster panel show "what's in this topic" without a second round-trip.
3. **`explanations.weights`** — enables the score formula in the hover tooltip: `baseline × 0.7 + cluster_affinity × 0.2 + cluster_support × 0.1`.

### 1.4.5 GeoSearch vs. Google and Bing

The Compare Engines tab (`ComparePane`) shows three columns: GeoSearch (BM25) · Google · Bing. Overlapping domains are tagged with `G` / `B` pills on the GeoSearch column; "In GeoSearch" appears on the Google/Bing columns when a domain overlaps.

| Query type | Winner | Why |
|------------|--------|-----|
| **Domain-scoped, technical** ("metamorphic facies", "garnet almandine zoning") | GeoSearch competitive/better | Corpus restricted to geoscience hosts → no commercial noise dilutes top-10. PageRank + HITS surface USGS, IRIS, BGS within the subgraph. |
| **Broad informational** ("volcanic eruption", "earthquake today") | Google / Bing | Unlimited index, freshness, click-trained ranking. GeoSearch doesn't recrawl so news-style queries are stale. |
| **Ambiguous terms** ("fault", "magma") | GeoSearch (with Clustered tab) | Google returns a flat list mixing senses; our cluster sidebar splits _fault_ into structural-geology vs hazard-mapping clusters. |
| **Pedagogical / definition** ("what is sedimentary rock") | Tie | Wikipedia wins position 1 on Google; GeoSearch returns textbook chapters and SERC modules — arguably better for a student. |

**Honest limits.** GeoSearch does not try to beat Google on coverage or freshness. It competes on **interpretability inside the geology domain**: visible clusters, visible expansion, visible rank deltas, visible authority signals.

### 1.4.6 Clustering in the UI

Clustering is treated as a **navigable dimension**, not a different ranking. Three UI surfaces:

1. **Left sidebar** — every cluster as a colored row with name and result count. Clicking filters the list to that cluster. Empty clusters appear below a divider so the user sees what _didn't_ match — useful for query reformulation.
2. **Per-card cluster tag** — each result card carries a `[cluster: name]` badge so users scanning the unfiltered list know each result's topic.
3. **Cluster representatives panel** — when a cluster is active, its top representative URLs appear below the result list for a "what's in this topic" summary.

Rank deltas (`↑ +4`) make the _effect_ of cluster reranking observable.

### 1.4.7 Demo query selection

Three constraints applied in order:

1. **Each query must exercise a distinct system component** — relevance/authority, clustering, expansion. All-same would feel redundant.
2. **Each query must return non-empty results on Google AND Bing** — otherwise the comparison column is empty.
3. **Each query must be intelligible to a non-geology audience** — so highly technical queries ("Wopmay orogen metamorphic facies") were rejected in favour of accessible ones with clear domain depth.

### 1.4.8 Three demo queries

#### Query 1 — `volcanic eruption hawaii` _(showcases PageRank / HITS authority)_

| Engine | Top-3 |
|--------|-------|
| **GeoSearch (BM25 + PageRank)** | USGS Hawaiian Volcano Observatory · Smithsonian Global Volcanism Program (Kīlauea) · USGS Volcano Hazards Program — all government/academic authorities. |
| **Google** | News articles (latest eruption coverage), Wikipedia, USGS, travel sites. Freshness signals dominate. |
| **Bing** | Similar to Google with more video/image carousels interleaved. |

**Take-away.** GeoSearch trades freshness for authority concentration. The Compare tab's domain-overlap badges show USGS appears in all three, but our top-3 is purely authoritative.

#### Query 2 — `metamorphic rock classification` _(showcases clustering)_

| Engine | Top-3 |
|--------|-------|
| **GeoSearch (Clustered tab)** | Cluster _protolith-based_: USGS metamorphic rocks, GeoSciWorld chapter. Cluster _grade-based_: SERC Carleton metamorphic grade module, university facies notes. Two clusters split cleanly. |
| **Google** | Wikipedia, Britannica, tutoring sites (study.com, byjus.com). Flat list, no topical decomposition. |
| **Bing** | Wikipedia, encyclopedia.com, educational video transcripts. |

**Take-away.** "Classification" has two distinct schemes in the literature. GeoSearch reveals that split in the sidebar; Google and Bing return a flat list.

#### Query 3 — `earthquake fault san andreas` _(showcases query expansion)_

| Engine | Behaviour |
|--------|-----------|
| **GeoSearch — BM25** | USGS earthquake hazard pages, SCEC research summaries, peer-reviewed seismology overviews. |
| **GeoSearch — Query Expansion (association)** | Original terms + `strike-slip`, `transform`, `tremor`, `creeping`. Expanded results add SCEC transform-fault explainer pages and USGS creeping-section pages — material the original query missed. |
| **Google** | USGS, Wikipedia, LA Times, National Geographic. Strong on news/feature; weaker on technical depth. |
| **Bing** | USGS, Wikipedia, history.com, news features. |

**Take-away.** Query expansion materially changes recall without leaving the same ranking model, and you can _see_ what was added in the Query Evolution banner. Neither Google nor Bing exposes this.

---

## 1.5 Clustering

> **Owner:** Preeti Vasaikar (pxv230036) · `cluster_service/`

### Algorithms

Three clustering algorithms available, selectable via the UI `Cluster` dropdown:

**Flat clustering (`flat`)** — MiniBatchKMeans on TF-IDF document vectors (`cluster_service/pipeline.py`):

| Parameter | Value |
|-----------|-------|
| K candidates | 8, 12, 16, 20, 24 |
| Batch size | 1,024 |
| `n_init` | 10 |
| `max_iter` | 200 |

Best K is selected by silhouette score over the candidates.

**Hierarchical Ward (`ward`)** — `AgglomerativeClustering(linkage='ward', metric='euclidean')` on mini-cluster centres:

| Parameter | Value |
|-----------|-------|
| K candidates | 8, 12, 16, 20, 24 |
| Linkage | Ward |
| Metric | Euclidean |

Ward minimizes total within-cluster variance at each merge step, producing compact, evenly-sized clusters well suited to navigable sidebar display.

**Hierarchical Complete (`complete`)** — `AgglomerativeClustering(linkage='complete', metric='cosine')` on L2-normalized mini-cluster centres:

| Parameter | Value |
|-----------|-------|
| K candidates | 8, 12, 16, 20, 24 |
| Linkage | Complete |
| Metric | Cosine |

Complete linkage minimizes the maximum inter-cluster distance, producing tight clusters that are good at isolating distinct sub-topics.

### Reranking formula

```
score = 0.70 × baseline_score + 0.20 × cluster_affinity + 0.10 × cluster_support
```

The 0.70 baseline weight is deliberate: even a perfect cluster match can only nudge a result, not invent a new top-1. Weights were tuned against the 50-query benchmark; nDCG@10 was best at 0.7/0.2/0.1 vs 0.5/0.3/0.2.

### Evaluation

50 benchmark queries from `cluster_service/benchmarks/queries_50.json`. nDCG@10 measured with and without cluster reranking. Reranking helped on ambiguous queries ("fault", "mineral") and was neutral on already-precise queries.

### What clustering adds

Clustering is a **post-processing layer** on top of relevance — it never replaces it. The UI exposes this honestly: the Clustered tab labels its scores with the formula, shows rank deltas vs. baseline, and lists empty clusters separately so the user sees when a topic had no matching results.

---

## 1.6 Query Expansion & Relevance Feedback

> **Owner:** Uddesh Singh (uxs230004) · `expander/`

### Rocchio relevance feedback (`expander/core.py`)

```
q_m = α · q₀  +  (β / |Dᵣ|) · Σ Dⱼ∈Dᵣ  −  (γ / |D̄ᵣ|) · Σ Dⱼ∈D̄ᵣ
```

| Parameter | Value | Role |
|-----------|-------|------|
| `alpha` | 1.0 | Weight of original query |
| `beta` | 0.75 | Weight of relevant documents |
| `gamma` | 0.25 | Weight of irrelevant documents |
| `num_new_terms` | 3 | Terms added per expansion |

20 queries were selected for Rocchio (spanning single-term to multi-term, easy to ambiguous). Relevant/irrelevant documents were hand-marked from the top-10 of each query. Modified queries gain 3–6 high-IDF domain terms (e.g., _"earthquake"_ → `+ seismograph + epicentre + magnitude`).

### Cluster-based expansion (`expander/core.py`)

Three methods for pseudo-relevance feedback on all 50 benchmark queries:

| Method | Idea | Function |
|--------|------|----------|
| **Association** | Co-occurrence of stems across the local top-k document set | `expand_association()` |
| **Scalar** | Cosine similarity between term-context vectors (association vectors normalized) | `expand_scalar()` |
| **Metric** | Distance-weighted co-occurrence (physically closer terms weighted more) | `expand_metric()` |

All three:

1. Retrieve top-k documents from BM25 as the local set.
2. Compute a term–term correlation matrix from that local vocabulary.
3. Cluster terms around the query terms.
4. Select the top-N new terms from each cluster.

### Evaluation (`expander/evaluate.py`)

Loads the 50-query benchmark and runs:

1. **Rocchio experiment** — 20 queries with simulated user judgments; records query vectors and modified queries.
2. **Local clustering experiment** — all 50 queries through Association, Scalar, and Metric; records expanded queries and new terms.

Output written to `expander/expansion_report.txt`.

### UI integration

The UI's Query Expansion tab calls `POST /api/expand` with `{ query, method, top_k }`. The response carries:

- `original_query` — original terms
- `expanded_query` — full expanded query string
- `results` — re-searched results using the expanded query

The **Query Evolution banner** renders original terms as pill chips, then `→`, then original + new terms (italicized). This makes what the system inferred visible to the user — a transparency affordance neither Google nor Bing provides.

---

# Part 2 — Technical Flow

## 2.1 The Big Picture

1. User opens `http://localhost:5173` — a React SPA built with Vite.
2. They type a query. On submit, 8 parallel HTTP requests fire through the backend proxy.
3. Four tabs receive results as they arrive, one panel at a time. The user can start reading the fastest model without waiting for the slowest.

Each tab corresponds to exactly one HTTP route. Each route corresponds to exactly one team member's module.

## 2.2 End-to-End Lifecycle of a Query

Take the query **"sedimentary rock"** on the **Clustered** tab:

```
Browser (App.jsx)
   │  POST /api/clustered-search
   │  { query: "sedimentary rock", cluster_method: "flat", baseline_method: "combined", top_k: 10 }
   ▼
Backend Proxy :8020  (search-engine/backend_proxy/app.py)
   │
   │  step 1 — search backend
   ├──► POST :8000/api/search
   │       • preprocess: tokenize → stopwords → stem → ["sediment", "rock"]
   │       • BM25 over inverted_index
   │       • return top_k results [{doc_id, score, url, title, snippet}, ...]
   │
   │  step 2 — cluster service
   ├──► POST :8010/v1/rerank
   │       • look up each doc's cluster assignment
   │       • compute cluster_affinity + cluster_support per doc
   │       • rerank: 0.70 × baseline_score + 0.20 × affinity + 0.10 × support
   │       • attach cluster_id, cluster_name, rank_delta to each result
   │
   │  step 3 — return enriched payload to browser
   ▼
Browser renders cluster sidebar + tagged result cards
```

For the **Query Expansion** tab: proxy hits `:8000/api/expand` → `QueryExpander` expands the query string → re-runs BM25 with expanded terms → returns `original_query`, `expanded_query`, `results`.

For the **Compare Engines** tab: proxy calls SerpAPI with `engine=google` and `engine=bing`, normalizes the response shape, and returns both alongside BM25 results.

## 2.3 Service Topology — Ports & Data Files

| Port | Service | File | Owns |
|------|---------|------|------|
| 5173 | Vite + React | `search-engine/frontend/` | UI |
| 8020 | Backend Proxy (FastAPI) | `search-engine/backend_proxy/app.py` | Orchestration, SerpAPI |
| 8000 | Search Backend (FastAPI) | `backend/app.py` | SearchEngine + QueryExpander |
| 8010 | Cluster Service (FastAPI) | `cluster_service/app.py` | Flat/Ward/Complete clustering, rerank |

Data files loaded at startup:

| Service | Reads |
|---------|-------|
| Search Backend | `indexer/data/inverted_index.json`, `doc_store.json`, `web_graph.json`, `pagerank_scores.json` |
| Cluster Service | `cluster_service/output/builds/<id>/cluster_catalog.json`, `url_assignments.json` |
| Backend Proxy | `cluster_service/benchmarks/queries_50.json` (for `/api/demo-queries`) |

Startup time: 5–15 s for the Search Backend (JSON deserialization of the 452 MB inverted index).

## 2.4 Data Pipeline (offline)

Run once before starting any service:

```
Step 1 — Crawl
  python -m crawler
    ├── frontier.py     async BFS, per-domain politeness
    ├── fetcher.py      aiohttp + retries + exponential backoff
    ├── parser.py       extract title, clean text, outbound links
    ├── dedup.py        bloom filter + URL canonicalization + content hash
    └── storage.py      writes pages.jsonl + web_graph.csv
  → crawler/output/pages.jsonl    (105,730 pages)
  → crawler/output/web_graph.csv  (outbound edges)

        ▼

Step 2 — Build the index
  cd indexer && python src/search.py --build
    ├── loader.py       reads pages.jsonl + web_graph.csv
    ├── preprocessor.py tokenize → stopwords → Porter stem
    ├── index.py        parallel inverted index build
    └── graph.py        web graph + topic-specific PageRank
  → indexer/data/inverted_index.json   (452 MB)
  → indexer/data/doc_store.json        (79 MB)
  → indexer/data/web_graph.json        (110 MB)
  → indexer/data/pagerank_scores.json  (3.2 MB)
  → indexer/data/graph_stats.json

        ▼

Step 3 — Build clusters (auto on first cluster_service start)
  python -m cluster_service
    vectorizer.py → corpus.py → pipeline.py
  → cluster_service/output/builds/<id>/cluster_catalog.json
  → cluster_service/output/builds/<id>/url_assignments.json
```

## 2.5 How Modules Are Linked

Every module boundary is a **plain JSON contract** — no shared Python state, no shared memory.

| Boundary | Contract |
|----------|----------|
| Crawler → Indexer | `pages.jsonl` (JSONL) + `web_graph.csv` (CSV) on disk |
| Indexer → Search Backend | JSON files in `indexer/data/` loaded by `SearchEngine.load()` |
| Indexer → Expander | `QueryExpander` constructed with a `SearchEngine` instance; reuses the same in-memory index |
| Search Backend → Frontend | HTTP JSON: `POST /api/search`, `POST /api/expand` |
| Cluster Service → Frontend | HTTP JSON via proxy: `POST /api/clustered-search` → `:8010/v1/rerank` |
| Frontend → External | HTTP JSON via proxy: `POST /api/external-search` → SerpAPI |

This decoupling is why the team could work in parallel: as long as the JSON shape is stable, each student owns their internals.

## 2.6 Running the Project End-to-End

```bash
# 0. one-time setup
pip install -r requirements.txt

# 1. crawl (skip if crawler/output/ already populated)
python -m crawler

# 2. build the index (~5 min)
cd indexer && python src/search.py --build && cd ..

# 3. start all services (each in its own terminal)
python backend/app.py                                                      # :8000
python -m cluster_service                                                   # :8010
uvicorn search-engine.backend_proxy.app:app --host 127.0.0.1 --port 8020  # :8020
cd search-engine/frontend && npm install && npm run dev                    # :5173

# 4. open http://localhost:5173
```

**Environment variables** (set in `search-engine/.env`):

```
SERPAPI_KEY=<your_key>          # required for Compare Engines tab
```

**CLI search** (no servers needed):

```bash
cd indexer
python src/search.py -q "earthquake fault" -m bm25 -k 10
python src/search.py -q "volcanic eruption" -m pagerank
# methods: tfidf | bm25 | pagerank | hits
```

**Query expansion evaluation:**

```bash
python expander/evaluate.py   # writes expander/expansion_report.txt
```
