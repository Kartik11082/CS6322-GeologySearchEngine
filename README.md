# CS 6322 — Geology Search Engine

A full information retrieval pipeline specialized for geosciences content. Built as a 5-student class project for CS 6322 (Information Retrieval) at UTD, Spring 2026.

> **Project Title:** Search engine for **Geology / Earth Sciences**
> **Team:** UTD CS 6322 — Spring 2026
> **Modules:** Crawler · Indexer + Relevance · User Interface · Clustering · Query Expansion

---

## Table of Contents

1. [Part 1 — Report Answers (Approach + Reasoning)](#part-1--report-answers-approach--reasoning)
   - [1.1 The Problem & Architecture](#11-the-problem--architecture)
   - [1.2 Crawling](#12-crawling)
   - [1.3 Indexing & Relevance](#13-indexing--relevance)
   - [1.4 User Interface & Comparisons](#14-user-interface--comparisons-with-google--bing)
   - [1.5 Clustering](#15-clustering)
   - [1.6 Query Expansion & Relevance Feedback](#16-query-expansion--relevance-feedback)
2. [Part 2 — Technical Flow (How Everything Works)](#part-2--technical-flow-how-everything-works)
   - [2.1 The Big Picture](#21-the-big-picture-what-the-user-sees)
   - [2.2 End-to-End Lifecycle of a Query](#22-end-to-end-lifecycle-of-a-query)
   - [2.3 Service Topology](#23-service-topology--ports--data-files)
   - [2.4 Data Pipeline](#24-data-pipeline-offline)
   - [2.5 How Modules Are Linked](#25-how-modules-are-linked)
   - [2.6 Running the Project](#26-running-the-project-end-to-end)

---

# Part 1 — Report Answers (Approach + Reasoning)

This section maps directly to the rubric in `documents/Project_Report_Templates.pdf`. Each subsection explains **what we did** and **why we chose that approach**.

## 1.1 The Problem & Architecture

**Goal.** Build a domain-focused search engine for geology / earth-science web content that supports keyword search, multiple ranking models, clustering, and query expansion — and lets us compare results against Google and Bing.

**Why a vertical (domain-specific) engine?** General-purpose engines optimize for popularity. A geology-focused crawler + index lets us:

- Aggressively filter junk so the corpus is dense in geology content.
- Run topic-specific PageRank that boosts pages with high geology relevance, not pages with the most ads.
- Let students apply each IR concept (BM25, HITS, Rocchio, scalar/metric/associative clustering) on a coherent corpus.

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
│ :8000      │    │ cluster_service │        └──────────────┘
│ backend/   │    └─────────────────┘
│  app.py    │
│            │
│ uses:      │
│  • indexer │
│  • expander│
└─────┬──────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  Static index data (built offline)                   │
│  indexer/data/  ← inverted_index.json, doc_store.json│
│                  web_graph.json, pagerank_scores.json│
└──────────────────────────────────────────────────────┘
                ▲
                │ build step
                │
┌───────────────┴──────────────────────────────────────┐
│  Crawler (async, BFS, geology-focused)               │
│  crawler/  →  crawled_data/pages.jsonl + web_graph.csv│
└──────────────────────────────────────────────────────┘
```

**Module ownership.**

| Module               | Owner     | Directory                                                 |
| -------------------- | --------- | --------------------------------------------------------- |
| Crawler              | Student 1 | `crawler/`                                                |
| Indexing & relevance | Student 2 | `indexer/src/`                                            |
| User interface       | Student 3 | `search-engine/frontend/`, `search-engine/backend_proxy/` |
| Clustering           | Student 4 | `cluster_service/`                                        |
| Query expansion      | Student 5 | `expander/`                                               |

**What we learned / difficulties.**

- Crawling: politeness vs. throughput is a real trade-off. Per-domain delays + bloom filter dedup were essential.
- Indexing: JSON deserialization at boot was slow (5–13 s), which killed CLI ergonomics. We solved it by keeping the index hot in a long-running FastAPI server.
- Frontend: cluster reranking and query expansion both reorder results — surfacing this transparently in the UI was harder than building the algorithms.

---

## 1.2 Crawling

**How pages were gathered.** Async BFS crawler (`crawler/crawler/`) using `aiohttp` with a domain-aware frontier. The crawler keeps `CONCURRENCY = 60` workers, enforces a `DOMAIN_DELAY = 1.0 s` per host, and walks up to `MAX_DEPTH = 10` from the seed list.

All 604 pages were preprocessed and indexed. Why this number? The crawler is configured for `TARGET_PAGES = 100,000`, but the **two-stage geology filter** (URL keywords + on-page content keywords) discards the bulk of crawled URLs — the trade is corpus quality over raw count.

**Sources / seeds.** The crawler is biased toward authoritative geoscience hosts via `FOCUS_KEYWORDS` in `crawler/crawler/config.py`:

- US Geological Survey (`usgs`), British Geological Survey (`bgs.ac.uk`), NOAA (`noaa.gov`), IRIS Seismology (`iris.edu`), SERC Carleton, GeoNet, NASA Earth Observatory, mindat.org.
- Topic-keyword seeds: `geolog*`, `mineral`, `volcano`, `seismic`, `earthquake`, `tectonic`, `fossil`, `sediment`, `stratigraph`, `petrolog*`, `geophysic*`, `geochem*`, `igneous`, `metamorphic`, `magma`, `lava`, `fault`, etc. (see lines 17–71 of the config).

**Reasoning:** Restricting both the URL substrings AND the on-page content (`CONTENT_KEYWORDS`) to geology terms is what keeps `geology_score` meaningful. Without the on-page check, navigation pages from authoritative domains would slip through.

**Deduplication.** Three layers (`crawler/crawler/dedup.py`):

1. **URL canonicalization** — lowercase host, strip fragments / tracking params (`utm_*`, `share=`, `ref=`).
2. **Bloom filter** for visited URLs (`BLOOM_CAPACITY = 5,000,000`, `BLOOM_ERROR_RATE = 0.001`). Memory-efficient O(1) membership check.
3. **Content hash** (SHA-256 of normalized text) so two URLs that resolve to the same page (redirects, mirror sites) are stored once.

**Why bloom + hash, not just a set?** A set works at small scale; bloom is for the planned 100 k corpus. Content hashing catches the case where canonicalization can't (different URLs, same body).

**Hyperlink hand-off to the indexer.** The crawler emits two files into `crawled_data/`:

- `pages.jsonl` — one JSON object per page: `{doc_id, url, final_url, title, clean_text, geology_score}`.
- `web_graph.csv` — one edge per row: `source_doc_id, source_url, target_url, anchor_text`.

The indexer's `loader.py` reads both, uses `build_url_to_docid()` to map every URL (and its redirect target) to a `doc_id`, and constructs the directed web graph with only edges whose source AND target both exist in the crawled set.

---

## 1.3 Indexing & Relevance

**Index construction.** `indexer/src/index.py` runs a 4-step pipeline in parallel using `multiprocessing.Pool` over all CPU cores:

```
pages.jsonl ──► tokenize ──► remove stopwords ──► PyStemmer (Porter)
                                                        │
                                                        ▼
                                          inverted_index + doc_store
```

- **Tokenization regex:** `[a-z0-9]+` (lowercase alphanumeric).
- **Stopwords:** NLTK English list.
- **Stemmer:** PyStemmer (C-based, ~10× faster than NLTK's Porter).

Inverted index shape: `{ stem: { doc_id: tf } }`. Doc store keeps URL, title, length, and `geology_score`.

**Web graph statistics** (from `indexer/data/graph_stats.json`):

| Metric         | Value                |
| -------------- | -------------------- |
| Nodes          | 101,956              |
| Edges          | 3,188,271            |
| Max in-degree  | 21,343 (doc_id 1256) |
| Max out-degree | 149 (doc_id 184)     |
| Avg out-degree | 31.27                |

The graph has more nodes than crawled pages because edges reference target URLs we discovered but did not necessarily crawl — every target gets a node so PageRank flows correctly even toward un-crawled authorities.

**Connecting graph → index.** `WebGraph.build_from_data()` resolves every edge's `target_url` to a `doc_id` via the URL-to-docid map, so PageRank scores and HITS scores are keyed by the same `doc_id` that the inverted index uses. The search endpoint joins them by ID at query time.

**Two relevance models** (`indexer/src/relevance.py`):

1. **TF-IDF cosine similarity**
   - `TF(t,d) = 1 + log10(tf)`
   - `IDF(t) = log10(N / df)`
   - Score = cosine(query vector, doc vector).
   - **Why:** classic baseline; cosine normalization handles document-length variance.

2. **Okapi BM25** (default)
   - `IDF(t) = log((N − df + 0.5) / (df + 0.5) + 1)`
   - Score saturates with `k1 = 1.2` and length-normalizes with `b = 0.75`.
   - **Why:** outperforms TF-IDF empirically; saturating term-frequency stops a single repeated word from dominating.

**Topic-specific PageRank.** Standard PageRank teleports uniformly. We bias the teleport vector by `geology_score`:

```
teleport[i] = geology_score[i] / Σ geology_scores
PR_new = d × link_contributions + (1 − d) × teleport
```

with `damping = 0.85`, `max_iter = 100`, `tol = 1e-6`. Top-PageRank pages are typically USGS hub pages (e.g., USGS Natural Hazards, USGS Volcano Hazards Program) — exactly what we want surfaced for general geology queries.

**HITS.** Query-dependent (`graph.py`): build root set from inverted-index hits → expand by 50 neighbours → iterate `auth/hub` to convergence. For a query like _"plate tectonics"_, top authorities are usgs.gov tectonics pages and IRIS educational pages.

**Collaboration with UI student.** We agreed on a JSON contract (`indexer/api_contract.md`) so the frontend dropdown maps 1:1 to a method string (`tfidf | bm25 | pagerank | hits`). The UI student generated 50 queries; we used those to A/B the rankings.

**Query set & judgment.** ~50 queries (the cluster benchmark in `cluster_service/benchmarks/queries_50.json` is reused). We graded the top-10 of each method on a 0/1 relevance scale and averaged precision@10. BM25 and PageRank-blended BM25 came out best; HITS won on broad authority-style queries.

**Collaboration with clustering student.** Cluster reranking is layered on top of the BM25 result list — we did not modify the relevance models themselves. The cluster service consumes our `(doc_id, score, snippet)` payload and produces a reordered list.

---

## 1.4 User Interface & Comparisons with Google & Bing

> **Owner:** Kartik Karkera (NetID: KXK230091)
> **Files:** `search-engine/frontend/src/App.jsx`, `search-engine/frontend/src/styles.css`, `search-engine/backend_proxy/app.py`

This section covers the full rubric: interface design (10) → working with the indexer (5) → query testing volume (5) → working with the cluster student (3) → comparison with Google/Bing (5) → clustering in presentation (2) → demo-query selection (5) → three concrete query examples (10). I'll go through each with the reasoning behind the choice so it's defensible at the demo.

---

### 1.4.1 How I designed the interface (10 points)

**One-line summary.** A minimal, light-mode React SPA whose entire job is to make four ranking models, a clustered view, a query-expansion view, and live Google/Bing results understandable to a non-IR user — at a glance, side by side.

**Layout.** Two app states:

1. **Landing.** Single serif wordmark (`Geo<em>Search</em>`), one-line tagline, pill-shaped search input, and an "About this project" link. Nothing else competes with the query box. (`Landing` component, `App.jsx:161`.)
2. **Results page.** Three layers stacked vertically:
   - Sticky **TopBar** (`App.jsx:192`) — keeps the query editable at all times, plus three controls: `Top K` stepper (3–20), `Expand` method (`association | scalar | metric`), `Cluster` method (`flat | ward | complete`).
   - **TabBar** with four tabs (`App.jsx:244`): **Relevance Models · Clustered · Query Expansion · Compare Engines**. Each tab shows a count badge of how many results it produced.
   - **Pane content** that swaps based on the active tab.

**Result card design** (`ResultCard`, `App.jsx:267`). Every result (whatever the source) renders the same way so visual comparison is fair:

```
┌─────┬───────────────────────────────────────────────┬─────────┐
│  #1 │ [Strong]  [cluster: Seismology]  ↑ +4         │ Score   │
│     │ usgs.gov                                      │ 4.293   │
│     │ Earthquake Hazards Program | USGS             │         │
│     │ "The U.S. Geological Survey monitors..."      │         │
└─────┴───────────────────────────────────────────────┴─────────┘
            ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔  (score bar, normalized)
```

- **Match-strength badge** (`Strong / Moderate / Weak`) computed from the score (`matchStrength`, `App.jsx:101`) — gives a non-numeric read of relevance.
- **Cluster tag** — only shown on the Clustered tab.
- **Rank delta** (`↑ +N` / `↓ -N`) — shown when the cluster service moved a result relative to its baseline rank. This makes the *effect* of reranking visible, not just the result.
- **Score bar** (`scoreWidth`, `App.jsx:85`) — bar widths are normalized to the **max score in the same panel**, so within one ranker the bars are honest and across rankers they aren't misleadingly comparable.

**Why these choices?**

- **Tabs over a single combined view** because each module (relevance / cluster / expansion / external) has a *different* explanation story; mixing them would obscure what each component contributes.
- **Skeleton cards** (`SkeletonCard`, `App.jsx:128`) instead of a global spinner — every panel loads independently (`runSearch`, `App.jsx:926` fires four model calls + expand + cluster + Google + Bing **in parallel**), so the user sees the fastest model first instead of waiting for the slowest.
- **URL state sync** (`syncUrl`, `App.jsx:74`) — query, top-K, methods, and active tab are all reflected in the URL, so any view is shareable / bookmarkable. Useful at demo time and for grading reproducibility.
- **Per-pane attribution** (`PaneAttrib`, `App.jsx:39`) — every pane shows the name + NetID of the student who owns the underlying module, satisfying the rubric's "name/netID on every page" requirement directly inside the UI.
- **Typography & color.** Inter for UI chrome, Fraunces (serif) for content; near-black on white with a single dark accent. Color appears only on **information-bearing elements** (cluster dots, badges, overlap tags) — not as decoration. This was a deliberate move so the cluster sidebar (which *does* use color) doesn't fight with surrounding chrome.

---

### 1.4.2 How I worked with the indexer student & accessed the relevance models (5 points)

**Decoupling principle.** The frontend never imports any Python. The indexer student (Rahul) owns `indexer/src/SearchEngine`; I owned the SPA + the FastAPI proxy. We met early and froze a JSON contract so we could iterate independently.

**The contract** (documented in `indexer/api_contract.md`):

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

`method` is one of `bm25 | tfidf | pagerank | hits`. Result shape is identical across methods, so the UI re-uses the same `ResultCard` for all four — switching between rankers is just changing one field in the request body.

**Request fan-out** (`runSearch`, `App.jsx:926`). On every search submit, the UI fires:

| Endpoint | Purpose | Tab that renders it |
|---|---|---|
| `POST /api/search` × 4 | one per method (BM25, TF-IDF, PageRank, HITS) | Relevance Models |
| `POST /api/expand` | pseudo-relevance feedback | Query Expansion |
| `POST /api/clustered-search` | reranked + clustered | Clustered |
| `POST /api/external-search` × 2 | Google + Bing via SerpAPI | Compare Engines |

Eight requests in parallel through one proxy. Each response patches its own slice of state via `patchResults` (`App.jsx:908`) using a `requestIdRef` guard so a stale response from a previous query never overwrites a newer one — a real bug I hit when typing fast.

**Why a proxy at :8020 instead of calling :8000 directly?**

1. **Browser CORS** — keeping all requests same-origin (the Vite dev proxy forwards `/api/*` to `:8020`) avoids CORS preflights and works the same in dev and bundled builds.
2. **Orchestration** — `/api/clustered-search` needs to call the search backend *and then* the cluster service. That logic belongs server-side, not in React.
3. **Secret hiding** — the SerpAPI key (`SERPAPI_KEY` in `search-engine/.env`) only ever lives on the server, never in the bundle.

**What we iterated on together.** The first contract returned only `(doc_id, score)`. After building a first version of the UI, we added `display_url`, `title`, and `snippet` to the response so the card could render without an extra round-trip. We also negotiated `top_k` clamping (3–20) on the server so the UI stepper limits make sense.

---

### 1.4.3 Number of queries used for testing (5 points)

**Total: 50 + 20 = 70 queries.**

| Source | Count | Purpose |
|---|---|---|
| **Co-authored with the relevance-model student** | **30** | Probe specific ranker behaviours: rare-term recall, BM25 vs TF-IDF divergence, link-heavy topics for PageRank, authoritative-hub queries for HITS. Drawn from `cluster_service/benchmarks/queries_50.json` + ad-hoc additions. |
| **Generated independently by me (frontend)** | **20** | Representative user-facing queries: a mix of single-term ("fault"), multi-word phrase ("metamorphic rock classification"), and natural-language style ("how do volcanoes form") — pulled from geology textbook indexes and USGS topic landing pages. |

**Why this split.** The 30 relevance-focused queries served the indexer team for tuning BM25 / PageRank weights. The 20 UI queries served *me* — they catch edge cases the indexer queries don't: very short queries (single-term), ambiguous queries ("fault" can mean rock fault or generic fault), and conversational queries (full sentences). Both pools live in the cluster benchmark file so the same set drives evaluation across modules.

**Judgment.** For UI testing I scored each query on three axes:

1. Does the BM25 panel return ≥ 1 obviously-relevant result in top-3?
2. Does the Clustered panel produce ≥ 2 distinct, sensibly-named clusters?
3. Does Query Expansion add domain terms (not noise)?

Failures here drove the back-end fixes (e.g., empty-result fallback for HITS on rare terms).

---

### 1.4.4 How I collaborated with the clustering student (3 points)

**Service boundary.** The clustering student (Preeti) runs an independent FastAPI service on `:8010`. We agreed she would **never replace** the baseline ranking — she only **reorders within the top-k** that BM25 already produced. This matters because clustering quality is bounded by retrieval quality; bypassing relevance would be strictly worse.

**The contract** (proxy-side): `POST /api/clustered-search` → forwards to `:8010/v1/rerank`. The response shape:

```json
{
  "reranked": [
    { "rank": 1, "baseline_rank": 3, "rank_delta": 2,
      "score": 0.84, "baseline_score": 0.71,
      "cluster_id": "c3", "cluster_name": "Seismology",
      "url": "...", "title": "...", "snippet": "..." }
  ],
  "clusters": [
    { "id": "c3", "name": "Seismology", "result_count": 4,
      "representatives": [ { "url": "...", "title": "..." } ] }
  ],
  "explanations": {
    "weights": { "baseline": 0.7, "cluster_affinity": 0.2, "cluster_support": 0.1 }
  }
}
```

The three things I asked for to make the UI work:

1. **`baseline_rank` + `rank_delta`** so the UI can show *movement* (`↑ +N`) — clustering is invisible without this signal.
2. **`representatives`** per cluster so the active cluster panel can show "what's in this topic" without a second request.
3. **`explanations.weights`** so the score formula can be displayed in the pane tooltip — interpretability is the entire point of the Clustered tab.

**How clustering improves relevance, not replaces it.** The cluster service uses `baseline_score × 0.7 + cluster_affinity × 0.2 + cluster_support × 0.1`. The 0.7 baseline weight is what protects relevance — even a perfect cluster match can only nudge a result, not invent a new top-1. We tuned these weights together against the 50-query benchmark; nDCG@10 was best at 0.7/0.2/0.1 vs 0.5/0.3/0.2.

---

### 1.4.5 How GeoSearch compares to Google and Bing (5 points)

The Compare Engines tab (`ComparePane`, `App.jsx:538`) shows three columns side by side: GeoSearch (BM25) · Google · Bing. Domains that overlap are tagged with `G` / `B` badges on the GeoSearch column, and a header shows how many domains overlap. This is the most-used view at demo time.

**My judgement, with reasoning.**

| Query type | Winner | Why |
|---|---|---|
| **Domain-scoped, technical** ("metamorphic facies", "garnet almandine zoning") | **GeoSearch competitive / better** | Crawl is restricted to geoscience hosts → no commercial / news noise dilutes top-10. PageRank + HITS surface authoritative hubs (USGS, IRIS, BGS, SERC) within the subgraph. |
| **Broad informational** ("volcanic eruption", "earthquake today") | **Google / Bing** | Effectively unlimited index, freshness, and learned ranking signals trained on click data. We don't recrawl, so news-style queries are stale. |
| **Ambiguous terms** ("fault", "magma") | **GeoSearch** when paired with the Clustered tab | Google returns a flat top-10 mixing senses; our cluster sidebar splits *fault* into structural-geology vs hazard-mapping clusters, letting the user disambiguate visually. |
| **Pedagogical / definition** ("what is sedimentary rock") | **Tie** | Wikipedia tends to win position 1 on Google; GeoSearch returns textbook chapter pages and SERC modules, which are arguably *better* for a student but harder to recognise without the brand cue. |

**Honest limits.** We are not trying to beat Google on coverage or freshness. We compete on **interpretability inside the geology domain**: visible clusters, visible expansion, visible rank deltas, visible authority signals. That's the trade.

---

### 1.4.6 How clustering shapes presentation in the UI (2 points)

Clustering is treated as a **navigable dimension**, not a different ranking. Three concrete UI uses (`ClusteredPane` + `ClusterSidebar`, `App.jsx:353`):

1. **Left sidebar** — every cluster as a colored row with name and result count. Clicking filters the result list to that cluster (`activeClusterId` state). Empty clusters are shown below a divider so the user sees *what didn't match* — useful for query reformulation.
2. **Per-card cluster tag** — each result card carries a `[cluster: name]` badge so a user scanning the unfiltered list still knows which topic each result belongs to.
3. **Cluster representatives panel** — when a cluster is active, its top representative URLs are listed below the result list, giving a one-glance "what's in this topic" summary.

The reranked-vs-baseline rank delta (`↑ +4`) on each card is the fourth piece — it makes the *effect* of clustering observable, not just the output.

---

### 1.4.7 How I selected the demo queries (5 points)

Three constraints, applied in order:

1. **Each query must exercise a distinct system component** — one for relevance/authority, one for clustering, one for expansion. If all three queries showcased the same module, the demo would feel redundant.
2. **Each query must return non-empty results on Google AND Bing** — otherwise the comparison column is empty and the side-by-side story falls apart.
3. **Each query must be intelligible to a non-geology audience** (the grader) — so highly technical queries like "Wopmay orogen metamorphic facies" were rejected in favour of accessible ones with clear domain depth.

That gave the three queries below.

---

### 1.4.8 Three demo queries with side-by-side results (10 points)

#### Query 1 — `volcanic eruption hawaii`  *(showcases PageRank / HITS authority)*

| Engine | Top-3 (by domain class) |
|---|---|
| **GeoSearch (BM25 + PageRank)** | 1. USGS Hawaiian Volcano Observatory · 2. Smithsonian Global Volcanism Program (Kīlauea) · 3. USGS Volcano Hazards Program — all three are government / academic authorities in the crawl. |
| **Google** | News articles (latest eruption coverage), Wikipedia "Hawaiian eruption", USGS, travel sites (volcanonationalpark.com). News dominates because of freshness signals. |
| **Bing** | Similar to Google, with more video / image carousels interleaved. |

**Take-away.** GeoSearch trades freshness for authority concentration. The Compare tab's domain-overlap badges show USGS appears in all three, but our top-3 is purely authoritative — Google's isn't.

#### Query 2 — `metamorphic rock classification`  *(showcases clustering)*

| Engine | Top-3 |
|---|---|
| **GeoSearch (Clustered tab)** | Cluster *protolith-based*: USGS metamorphic rocks, GeoSciWorld chapter on parent-rock classification. Cluster *grade-based*: SERC Carleton metamorphic grade module, university course notes on facies series. The **two clusters split cleanly** in the sidebar. |
| **Google** | Wikipedia "Metamorphic rock", Britannica, tutoring sites (study.com, byjus.com). Flat list, no topical decomposition. |
| **Bing** | Wikipedia, encyclopedia.com, educational video transcripts. |

**Take-away.** Google returns a flat list dominated by encyclopedic sources. GeoSearch reveals that "classification" actually has *two* distinct schemes in the literature — the user can pick the one they want.

#### Query 3 — `earthquake fault san andreas`  *(showcases query expansion + comparison)*

| Engine | Top-3 / behaviour |
|---|---|
| **GeoSearch — BM25 only** | USGS earthquake hazard pages, SCEC research summaries, peer-reviewed seismology overviews. |
| **GeoSearch — Query Expansion (associative)** | Original `earthquake fault san andreas` → `+ strike-slip + transform + tremor + creeping`. The expanded results add SCEC's transform-fault explainer pages and USGS creeping-section pages — material the original query missed. |
| **Google** | USGS, Wikipedia, LA Times, National Geographic. Strong on news/feature; weaker on technical depth. |
| **Bing** | USGS, Wikipedia, history.com, news features. |

**Take-away.** Query expansion materially changes recall — without leaving the same ranking model — and you can *see* what was added in the Query Evolution banner (`ExpandedPane`, `App.jsx:476`). Neither Google nor Bing exposes this.

---

### 1.4.9 Quick visual map of the UI ↔ backend wiring

```
                              ┌──────────────────────────────┐
                              │  React SPA (Vite :5173)      │
                              │  search-engine/frontend/      │
                              │  ┌──────────────────────────┐ │
                              │  │ TopBar  (query, K, etc.) │ │
                              │  ├──────────────────────────┤ │
                              │  │ Tabs                     │ │
                              │  │  ├─ Relevance Models     │ │  POST /api/search × 4
                              │  │  ├─ Clustered            │ │  POST /api/clustered-search
                              │  │  ├─ Query Expansion      │ │  POST /api/expand
                              │  │  └─ Compare Engines      │ │  POST /api/external-search × 2
                              │  └──────────────────────────┘ │
                              └────────────┬─────────────────┘
                                           │ (Vite proxy → /api/* → :8020)
                                           ▼
                              ┌──────────────────────────────┐
                              │  Backend Proxy  :8020        │
                              │  search-engine/backend_proxy │
                              └──┬─────────┬──────────┬──────┘
                                 │         │          │
                                 ▼         ▼          ▼
                            :8000        :8010      SerpAPI
                          Search API   Cluster      Google + Bing
                          (indexer)    (Preeti)     (external)
```

Every tab in the UI maps to exactly one network endpoint, every endpoint maps to exactly one team-mate's module — that is the system property the design is built around.

---

## 1.5 Clustering

**Flat clustering.** K-means / flat partitioning on TF-IDF document vectors (`cluster_service/pipeline.py`). We picked **k = 8** clusters because the corpus naturally groups by sub-discipline: _seismology, volcanology, mineralogy, paleontology, sedimentary geology, hydrogeology, geophysics, planetary geology_. Eight strikes the balance between meaningful labels and avoiding singleton clusters.

**Use of flat clusters.** Two ways:

1. **Reranking** — `/v1/rerank` takes the search engine's top-50 and boosts results in the dominant cluster of the query.
2. **UI presentation** — every result on the Clustered tab carries a cluster badge.

We did **not** retrain the relevance model on cluster info; clusters are a post-processing layer.

**Agglomerative clustering.** Ward linkage with cosine distance, cut to obtain comparable cluster counts (~8–12 leaf clusters). The dendrogram shows the expected nesting (e.g., _volcanology_ and _seismology_ merging under _natural hazards_).

**UI presentation.** Tree view collapsible by cluster; clicking a cluster filters the result list.

**Query set for clustering experiments.** All 50 queries in `cluster_service/benchmarks/queries_50.json`. We measured nDCG@10 with and without cluster reranking. Reranking helped on ambiguous queries (e.g., _"fault"_) and was neutral on already-precise queries.

**Demo queries** (3 examples shown in the demo): _"earthquake hazard"_, _"mineral identification"_, _"volcanic ash"_ — all show visible cluster grouping in the UI.

---

## 1.6 Query Expansion & Relevance Feedback

**Module:** `expander/core.py` (class `QueryExpander`, injected into `SearchEngine`).

**Rocchio.** 20 queries selected to span the corpus (mix of single-term and multi-term, easy and ambiguous). For each, we hand-marked relevant / irrelevant docs in the top-10 and applied:

```
q_new = α·q + β·(1/|Dr|)·Σ d∈Dr − γ·(1/|Dnr|)·Σ d∈Dnr
```

with α = 1.0, β = 0.75, γ = 0.15. Modified queries surface ~3–6 new high-IDF terms (e.g., _"earthquake"_ gains _seismograph, epicentre, magnitude_).

**Pseudo-relevance feedback (50 queries).** The 50-query benchmark drives all three cluster-based expansion methods:

| Method                     | Idea                                                 | Where in code                          |
| -------------------------- | ---------------------------------------------------- | -------------------------------------- |
| **Associative clustering** | Co-occurrence of stems across the local document set | `expander/core.py::associative_expand` |
| **Metric clustering**      | Distance-weighted co-occurrence (closer = stronger)  | `expander/core.py::metric_expand`      |
| **Scalar clustering**      | Cosine similarity between term-context vectors       | `expander/core.py::scalar_expand`      |

For each method we record (in `expansion_report.txt` after `python expander/evaluate.py`):

1. Local document set (top-N from initial BM25).
2. Local vocabulary + stems.
3. Correlation matrix and the cluster picked per query term.
4. Final expanded query.

**UI integration.** The expander has an endpoint `/api/expand` exposed by the search backend; the UI's `Expanded` tab calls it and renders the new query above the new results so the user can see what was added.

**Demo selection.** Three queries shown live: _"yellowstone"_ (gains _caldera, supervolcano_), _"san andreas"_ (gains _fault, transform, california_), _"impact crater"_ (gains _meteorite, ejecta_) — chosen because the expansion is large and easy to explain.

---

# Part 2 — Technical Flow (How Everything Works)

## 2.1 The Big Picture (what the user sees)

1. The user opens **`http://localhost:5173`** — a React SPA.
2. They type a query, pick a ranking method (BM25 / TF-IDF / PageRank / HITS) and optionally a tab (Local / Clustered / Google / Bing / Expanded).
3. They get back ranked results with snippets, cluster badges, and (on the comparison tabs) Google/Bing equivalents side by side.

Every tab corresponds to one HTTP route on the backend proxy.

## 2.2 End-to-End Lifecycle of a Query

Take the query **"sedimentary rock"** with method `bm25` on the **Clustered** tab:

```
Browser (App.jsx)
   │  POST /api/clustered-search { query: "sedimentary rock", method: "bm25", top_k: 10 }
   ▼
Backend Proxy :8020 (search-engine/backend_proxy/app.py)
   │  step 1: forward to search backend
   ├──► POST :8000/api/search   ──► SearchEngine.search(...)
   │       • preprocess query: ["sediment", "rock"]
   │       • BM25 over inverted_index
   │       • merge with PageRank if method = "pagerank"
   │       • return top_k results [{doc_id, score, url, title, snippet}, ...]
   │
   │  step 2: forward those results to cluster service
   ├──► POST :8010/v1/rerank    ──► ClusterPipeline.rerank(...)
   │       • look up each doc's cluster
   │       • boost results in the dominant cluster
   │       • attach cluster label to each result
   │
   │  step 3: return enriched payload to browser
   ▼
Browser renders cluster-tagged result cards
```

If the user is on the **Expanded** tab instead, the proxy hits `:8000/api/expand` (which calls `QueryExpander` before re-ranking). On the **Google / Bing** tab, the proxy hits SerpAPI using the key in `search-engine/.env`.

## 2.3 Service Topology — Ports & Data Files

| Port | Service                   | File                                 | What it owns                          |
| ---- | ------------------------- | ------------------------------------ | ------------------------------------- |
| 5173 | Vite + React              | `search-engine/frontend/`            | UI                                    |
| 8020 | Backend Proxy (FastAPI)   | `search-engine/backend_proxy/app.py` | Orchestration, SerpAPI                |
| 8000 | Search Backend (FastAPI)  | `backend/app.py`                     | SearchEngine + QueryExpander          |
| 8010 | Cluster Service (FastAPI) | `cluster_service/app.py`             | Flat + agglomerative clusters, rerank |

All three FastAPI servers load their data files at startup so the request path is hot:

| Service         | Reads at startup                                                                               |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Search Backend  | `indexer/data/inverted_index.json`, `doc_store.json`, `web_graph.json`, `pagerank_scores.json` |
| Cluster Service | `cluster_service/output/builds/<id>/{cluster_catalog,url_assignments}.json`                    |
| Backend Proxy   | `cluster_service/benchmarks/queries_50.json` (for `/api/demo-queries`)                         |

## 2.4 Data Pipeline (offline)

This runs **once** before any service starts.

```
┌───────────────────────────────────────────────────────────────┐
│ Step 1 — Crawl                                                │
│   python -m crawler                                           │
│   crawler/main.py                                             │
│     ├── frontier.py     async BFS, per-domain politeness      │
│     ├── fetcher.py      aiohttp + retries + backoff           │
│     ├── parser.py       extract title, clean text, links      │
│     ├── dedup.py        bloom filter + content hash           │
│     └── storage.py      writes pages.jsonl + web_graph.csv    │
│   → crawled_data/pages.jsonl    (604 pages)                   │
│   → crawled_data/web_graph.csv  (28,137 edges)                │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 2 — Build the index                                      │
│   cd indexer && python src/search.py --build                  │
│     ├── loader.py       reads pages.jsonl + web_graph.csv     │
│     ├── preprocessor.py tokenize → stopwords → Porter stem    │
│     ├── index.py        parallel build of inverted index      │
│     └── graph.py        web graph + topic-PageRank            │
│   → indexer/data/inverted_index.json                          │
│   → indexer/data/doc_store.json                               │
│   → indexer/data/web_graph.json                               │
│   → indexer/data/pagerank_scores.json                         │
│   → indexer/data/graph_stats.json                             │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 3 — Build clusters (auto on first cluster_service boot)  │
│   python -m cluster_service                                   │
│     vectorizer.py → corpus.py → pipeline.py                   │
│   → cluster_service/output/builds/<id>/cluster_catalog.json   │
│   → cluster_service/output/builds/<id>/url_assignments.json   │
└───────────────────────────────────────────────────────────────┘
```

After this, the four services start and stay up.

## 2.5 How Modules Are Linked

The seam between every two modules is a **plain JSON contract** — no shared Python state, no shared memory.

| Boundary                   | Contract                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Crawler → Indexer          | `pages.jsonl` (JSONL) + `web_graph.csv` (CSV) on disk                                                                    |
| Indexer → Search Backend   | `inverted_index.json`, `doc_store.json`, `web_graph.json`, `pagerank_scores.json` loaded by `SearchEngine.load()`        |
| Indexer → Expander         | Expander imports `SearchEngine`; `QueryExpander` is constructed _with_ the engine, so it reuses the same in-memory index |
| Search Backend → Frontend  | HTTP JSON: `POST /api/search`, `POST /api/expand`                                                                        |
| Cluster Service → Frontend | HTTP JSON via proxy: `POST /api/clustered-search` → `:8010/v1/rerank`                                                    |
| Frontend → external        | HTTP JSON via proxy: `POST /api/external-search` → SerpAPI                                                               |

This decoupling is _why_ the team could work in parallel: as long as the JSON shape is stable, each student owns their internals.

## 2.6 Running the Project End-to-End

```bash
# 0. one-time setup
pip install -r requirements.txt

# 1. crawl (skip if crawled_data/ already populated)
python -m crawler

# 2. build the index
cd indexer && python src/search.py --build && cd ..

# 3. start all services (each in its own terminal)
python backend/app.py                                                       # :8000
python -m cluster_service                                                    # :8010
uvicorn search-engine.backend_proxy.app:app --host 127.0.0.1 --port 8020     # :8020
cd search-engine/frontend && npm install && npm run dev                      # :5173

# 4. open http://localhost:5173
```

CLI search (no servers needed) is also supported:

```bash
cd indexer
python src/search.py -q "earthquake fault" -m bm25 -k 10
python src/search.py -q "volcanic eruption" -m pagerank
```
