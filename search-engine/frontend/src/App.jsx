import { startTransition, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_PROXY_API_BASE || "").replace(/\/$/, "");

const MODEL_CONFIG = [
  { key: "bm25",     label: "BM25",     description: "Okapi BM25 — primary lexical relevance model." },
  { key: "tfidf",    label: "TF-IDF",   description: "Vector-space cosine similarity baseline." },
  { key: "pagerank", label: "PageRank", description: "BM25 blended with link-authority from the crawl graph." },
  { key: "hits",     label: "HITS",     description: "Query-dependent authority scoring (Kleinberg)." },
];

const TAB_CONFIG = [
  { key: "models",    label: "Relevance Models" },
  { key: "clustered", label: "Clustered" },
  { key: "expanded",  label: "Query Expansion" },
  { key: "compare",   label: "Compare Engines" },
];

const EXPANSION_OPTIONS = [
  { value: "association", label: "Association" },
  { value: "scalar",      label: "Scalar" },
  { value: "metric",      label: "Metric" },
];

const CLUSTER_OPTIONS = [
  { value: "flat",     label: "Flat" },
  { value: "ward",     label: "Ward" },
  { value: "complete", label: "Complete" },
];

const emptyPanel = () => ({ status: "idle", data: null, error: "" });

const createInitialResults = () => ({
  models:    Object.fromEntries(MODEL_CONFIG.map((m) => [m.key, emptyPanel()])),
  expanded:  emptyPanel(),
  clustered: emptyPanel(),
  google:    emptyPanel(),
  bing:      emptyPanel(),
});

function readInitialState() {
  const p = new URLSearchParams(window.location.search);
  const topKParam = Number(p.get("topK"));
  return {
    query:           p.get("q") || "",
    topK:            Number.isFinite(topKParam) && topKParam > 0 ? topKParam : 10,
    expansionMethod: p.get("expand")  || "association",
    clusterMethod:   p.get("cluster") || "flat",
    tab:             p.get("tab")     || "models",
    view:            p.get("view")    || "search",
  };
}

function syncUrl({ query, topK, expansionMethod, clusterMethod, tab, view }) {
  const p = new URLSearchParams();
  if (query) p.set("q", query);
  p.set("topK",    String(topK));
  p.set("expand",  expansionMethod);
  p.set("cluster", clusterMethod);
  p.set("tab",     tab);
  if (view && view !== "search") p.set("view", view);
  window.history.replaceState({}, "", `${window.location.pathname}?${p.toString()}`);
}

function scoreWidth(value, items, field = "score") {
  if (!items?.length || typeof value !== "number" || Number.isNaN(value)) return 0;
  const vals = items.map((it) => Number(it[field] ?? 0)).filter(Number.isFinite);
  const max  = Math.max(...vals, 1);
  return Math.max(6, Math.min(100, (value / max) * 100));
}

function termsFromQuery(text) {
  return String(text || "").split(/\s+/).map((t) => t.trim()).filter(Boolean);
}

function domainFromUrl(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return url; }
}

function matchStrength(item) {
  const v = Number(item.score ?? item.baseline_score ?? 0);
  if (v >= 0.7) return { label: "Strong",   cls: "str" };
  if (v >= 0.3) return { label: "Moderate", cls: "mod" };
  return           { label: "Weak",     cls: "wek" };
}

async function requestJson(path, { method = "GET", body } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body:    body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail || data || "Request failed");
    throw new Error(detail);
  }
  return data;
}

/* ================================================================
   SKELETON LOADING
   ================================================================ */
function SkeletonCard() {
  return (
    <div className="skel">
      <div className="skel-rnk skel-line" />
      <div className="skel-main">
        <div className="skel-line" />
        <div className="skel-line" />
        <div className="skel-line" />
        <div className="skel-line" />
      </div>
      <div className="skel-score">
        <div className="skel-line" />
        <div className="skel-line" />
        <div className="skel-line" />
      </div>
    </div>
  );
}

function SkeletonList({ count = 5 }) {
  return <>{Array.from({ length: count }, (_, i) => <SkeletonCard key={i} />)}</>;
}

function getStatus(panel, emptyMessage, skeletonCount = 5) {
  if (panel.status === "loading") return <SkeletonList count={skeletonCount} />;
  if (panel.status === "error")   return <div className="state err">{panel.error}</div>;
  if (!panel.data)                return <div className="state muted">{emptyMessage}</div>;
  return null;
}

/* ================================================================
   LANDING
   ================================================================ */
function Landing({ queryInput, setQueryInput, onSubmit, onAbout }) {
  return (
    <div className="landing">
      <div className="brand">
        <span className="brand-mark">Geo<em>Search</em></span>
      </div>
      <p className="landing-lede">A domain-tuned search engine for the geological sciences.</p>

      <form className="landing-form" onSubmit={onSubmit}>
        <div className="hero-input">
          <input
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Search earthquakes, minerals, volcanoes…"
            autoFocus
          />
          <button className="hero-submit" type="submit">Search</button>
        </div>
      </form>

      <div className="landing-footer">
        CS 6322 · UTD · Geology corpus ·{" "}
        <button type="button" onClick={onAbout}>About this project</button>
      </div>
    </div>
  );
}

/* ================================================================
   TOP BAR (post-search)
   ================================================================ */
function TopBar({
  queryInput, setQueryInput, onSubmit,
  topK, setTopK,
  expansionMethod, setExpansionMethod,
  clusterMethod, setClusterMethod,
  onHome,
}) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <button className="logo" onClick={onHome} type="button" aria-label="Home">
          Geo<em>Search</em>
        </button>
        <form className="searchbar" onSubmit={onSubmit}>
          <input
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Search geology…"
          />
          <button type="submit">Search</button>
        </form>
        <div className="controls">
          <div className="ctl">
            <span className="ctl-lbl">Top K</span>
            <div className="topk-stepper">
              <button type="button" onClick={() => setTopK((v) => Math.max(3, v - 1))}>−</button>
              <span>{topK}</span>
              <button type="button" onClick={() => setTopK((v) => Math.min(20, v + 1))}>+</button>
            </div>
          </div>
          <label className="ctl">
            <span className="ctl-lbl">Expand</span>
            <select value={expansionMethod} onChange={(e) => setExpansionMethod(e.target.value)}>
              {EXPANSION_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <label className="ctl">
            <span className="ctl-lbl">Cluster</span>
            <select value={clusterMethod} onChange={(e) => setClusterMethod(e.target.value)}>
              {CLUSTER_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
        </div>
      </div>
    </header>
  );
}


/* ================================================================
   TABS
   ================================================================ */
function TabBar({ activeTab, setActiveTab, counts }) {
  return (
    <div className="tabs-wrap">
      <nav className="tabs">
        {TAB_CONFIG.map((t) => (
          <button
            key={t.key}
            type="button"
            className={`tab${activeTab === t.key ? " on" : ""}`}
            onClick={() => setActiveTab(t.key)}
          >
            {t.label}
            {counts?.[t.key] != null && <span className="badge">{counts[t.key]}</span>}
          </button>
        ))}
      </nav>
    </div>
  );
}

/* ================================================================
   RESULT CARD
   ================================================================ */
function ResultCard({ item, items, scoreField = "score", showClusterTag = false }) {
  const width     = scoreWidth(Number(item[scoreField] ?? 0), items, scoreField);
  const strength  = matchStrength(item);
  const rankDelta = Number(item.rank_delta ?? 0);

  return (
    <article className="rcard">
      <div className="rnk">{item.rank || item.position}</div>
      <div className="rc">
        <div className="rmeta">
          <span className={`badge ${strength.cls}`}>{strength.label}</span>
          {showClusterTag && item.cluster_name && (
            <span className="ctag">{item.cluster_name}</span>
          )}
          {"baseline_rank" in item && (
            <span className={`delta ${rankDelta > 0 ? "up" : rankDelta < 0 ? "down" : ""}`}>
              {rankDelta > 0
                ? `↑ +${rankDelta}`
                : rankDelta < 0
                  ? `↓ ${rankDelta}`
                  : "rank unchanged"}
            </span>
          )}
        </div>
        <div className="ru">{item.display_url || domainFromUrl(item.url)}</div>
        <a className="rt" href={item.url} target="_blank" rel="noreferrer">
          {item.title || item.url}
        </a>
        <p className="rs">{item.snippet || "No snippet available."}</p>
      </div>
      <aside className="rsc">
        <span className="sclbl">Score</span>
        <strong>{Number(item[scoreField] ?? 0).toFixed(3)}</strong>
        <div className="rsbar"><div className="rsfill" style={{ width: `${width}%` }} /></div>
      </aside>
    </article>
  );
}

/* ================================================================
   MODELS PANE
   ================================================================ */
function ModelsPane({ results, activeModel, setActiveModel }) {
  const panel  = results.models[activeModel];
  const active = MODEL_CONFIG.find((m) => m.key === activeModel);
  const status = getStatus(panel, `No ${active.label} results yet.`);

  return (
    <section>
      <div className="model-pills">
        {MODEL_CONFIG.map((m) => (
          <button
            key={m.key}
            type="button"
            className={`mpill${activeModel === m.key ? " on" : ""}`}
            onClick={() => setActiveModel(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="pane-intro">
        <h2>
          {active.label}
          <span className="info-tip" aria-label={active.description}>?</span>
        </h2>
      </div>

      <div className="results">
        {status || panel.data.results.map((item) => (
          <ResultCard
            key={`${activeModel}-${item.url}-${item.rank}`}
            item={item}
            items={panel.data.results}
          />
        ))}
      </div>
    </section>
  );
}

/* ================================================================
   CLUSTER SIDEBAR
   ================================================================ */
function ClusterSidebar({ clusters, activeClusterId, setActiveClusterId }) {
  const populated = clusters.filter((c) => (c.result_count ?? 0) > 0);
  const empty     = clusters.filter((c) => (c.result_count ?? 0) === 0);

  return (
    <aside className="cside">
      <div className="cside-head">Clusters</div>
      {populated.map((cluster, i) => (
        <button
          key={cluster.id}
          type="button"
          className={`crow${activeClusterId === cluster.id ? " on" : ""}`}
          onClick={() =>
            setActiveClusterId((cur) => (cur === cluster.id ? null : cluster.id))
          }
        >
          <span className={`cdot c${i % 8}`} />
          <span className="clbl">{cluster.name}</span>
          <span className="ccnt">{cluster.result_count}</span>
        </button>
      ))}
      {empty.length > 0 && (
        <>
          <div className="cside-divider">No results</div>
          {empty.map((cluster, i) => (
            <div key={cluster.id} className="crow crow-empty">
              <span className={`cdot c${(populated.length + i) % 8}`} />
              <span className="clbl">{cluster.name}</span>
              <span className="ccnt">0</span>
            </div>
          ))}
        </>
      )}
    </aside>
  );
}

/* ================================================================
   CLUSTERED PANE
   ================================================================ */
function ClusteredPane({ panel, activeClusterId, setActiveClusterId, clusterMethod }) {
  const status = getStatus(panel, "Run a query to inspect clustered reranking.");
  if (status) {
    return (
      <>
        <aside className="cside" />
        <section>
          <div className="pane-intro">
            <h2>Clustered Results</h2>
          </div>
          {status}
        </section>
      </>
    );
  }

  const clusters      = panel.data.clusters  || [];
  const reranked      = panel.data.reranked  || [];
  const filtered      = activeClusterId
    ? reranked.filter((it) => it.cluster_id === activeClusterId)
    : reranked;
  const activeCluster = clusters.find((c) => c.id === activeClusterId);
  const weights       = panel.data.explanations?.weights || {};

  return (
    <>
      <ClusterSidebar
        clusters={clusters}
        activeClusterId={activeClusterId}
        setActiveClusterId={setActiveClusterId}
      />
      <section>
        <div className="pane-intro">
          <h2>
            Clustered Results
            <span className="method-tag">{clusterMethod}</span>
            <span className="info-tip" aria-label="Results from the relevance backbone grouped by topical cluster, then reranked using cluster affinity and support. Score = baseline × 0.7 + cluster_affinity × 0.2 + cluster_support × 0.1">?</span>
          </h2>
        </div>

        <div className="hint">
          <span className="pulse" />
          <span>
            {activeCluster ? `Showing cluster: ${activeCluster.name}` : `${clusters.length} clusters · all results`}
          </span>
        </div>

        <div className="results">
          {filtered.map((item) => (
            <ResultCard
              key={`clustered-${item.url}-${item.rank}`}
              item={item}
              items={filtered}
              showClusterTag
            />
          ))}
        </div>

        {activeCluster && (activeCluster.representatives || []).length > 0 && (
          <div className="cluster-reps">
            <h4>Cluster Representatives</h4>
            <ul>
              {activeCluster.representatives.map((it) => (
                <li key={it.url}>
                  <a href={it.url} target="_blank" rel="noreferrer">
                    {it.title || it.url}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}

      </section>
    </>
  );
}

/* ================================================================
   EXPANDED PANE
   ================================================================ */
function ExpandedPane({ panel, searchQuery, expansionMethod }) {
  const status = getStatus(panel, "Run a query to inspect expansion output.");
  if (status) {
    return (
      <section>
        <div className="pane-intro">
          <h2>Query Expansion
            <span className="info-tip" aria-label="Pseudo-relevance feedback adds domain terms to the original query using co-occurrence statistics over top-ranked documents.">?</span>
          </h2>
        </div>
        {status}
      </section>
    );
  }

  const originalTerms = termsFromQuery(panel.data.original_query || searchQuery);
  const expandedTerms = termsFromQuery(panel.data.expanded_query).filter(
    (t) => !originalTerms.includes(t),
  );

  return (
    <section>
      <div className="pane-intro">
        <h2>
          Query Expansion
          <span className="method-tag">{expansionMethod}</span>
          <span className="info-tip" aria-label="The original query is expanded with terms from a co-occurrence cluster over top-ranked documents, then re-searched using the same relevance model.">?</span>
        </h2>
      </div>

      <div className="expbanner">
        <div className="expbanner-label">Query Evolution</div>
        <div className="expflow">
          <div className="expterms">
            {originalTerms.map((t) => <span key={`o-${t}`} className="oterm">{t}</span>)}
          </div>
          <span className="arrow">→</span>
          <div className="expterms">
            {originalTerms.map((t) => <span key={`on-${t}`} className="oterm">{t}</span>)}
            {expandedTerms.map((t) => <span key={`n-${t}`} className="nterm">+{t}</span>)}
          </div>
        </div>
      </div>

      <div className="results">
        {panel.data.results.map((item) => (
          <ResultCard
            key={`expanded-${item.url}-${item.rank}`}
            item={item}
            items={panel.data.results}
          />
        ))}
      </div>
    </section>
  );
}

/* ================================================================
   COMPARE PANE (GeoSearch vs Google vs Bing)
   ================================================================ */
function ComparePane({ results }) {
  const geo    = results.models.bm25;
  const google = results.google;
  const bing   = results.bing;

  const geoItems    = geo.data?.results    || [];
  const googleItems = google.data?.results || [];
  const bingItems   = bing.data?.results   || [];

  const googleDomains = new Set(googleItems.map((r) => domainFromUrl(r.url)));
  const bingDomains   = new Set(bingItems.map((r)   => domainFromUrl(r.url)));
  const geoDomains    = new Set(geoItems.map((r)    => domainFromUrl(r.url)));

  const geoVsGoogle = geoItems.filter((r) => googleDomains.has(domainFromUrl(r.url))).length;
  const geoVsBing   = geoItems.filter((r) => bingDomains.has(domainFromUrl(r.url))).length;

  const maxRows = Math.max(geoItems.length, googleItems.length, bingItems.length, 1);

  function overlapTags(url) {
    const d = domainFromUrl(url);
    const tags = [];
    if (googleDomains.has(d)) tags.push({ label: "G", title: "Also in Google", cls: "ov-g" });
    if (bingDomains.has(d))   tags.push({ label: "B", title: "Also in Bing",   cls: "ov-b" });
    return tags;
  }

  function ColHeader({ label, cls, count, overlap, overlapLabel }) {
    return (
      <div className={`cmp-head ${cls}`}>
        <div className="cmp-head-top">
          <span className="cmp-head-dot" />
          <span className="cmp-head-label">{label}</span>
          {count > 0 && <span className="cmp-head-count">{count}</span>}
        </div>
        {overlap != null && (
          <div className="cmp-head-overlap">
            {overlap} result{overlap !== 1 ? "s" : ""} in common with {overlapLabel}
          </div>
        )}
      </div>
    );
  }

  function GeoCol() {
    const status = getStatus(geo, "Run a search to see results.", 5);
    return (
      <div className="cmp-col">
        <ColHeader label="GeoSearch" cls="cmp-geo" count={geoItems.length}
          overlap={geoVsGoogle + geoVsBing > 0 ? Math.max(geoVsGoogle, geoVsBing) : null}
          overlapLabel="web engines" />
        {status || geoItems.map((item) => {
          const tags = overlapTags(item.url);
          return (
            <div key={item.url} className="cmp-row">
              <span className="cmp-rank">{item.rank}</span>
              <div className="cmp-body">
                <div className="cmp-domain">{item.display_url || domainFromUrl(item.url)}</div>
                <a className="cmp-title" href={item.url} target="_blank" rel="noreferrer">
                  {item.title || item.url}
                </a>
                {item.snippet && <p className="cmp-snippet">{item.snippet}</p>}
                {tags.length > 0 && (
                  <div className="cmp-overlap-tags">
                    {tags.map((t) => (
                      <span key={t.cls} className={`cmp-ov ${t.cls}`} title={t.title}>{t.label}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  function ExCol({ panel, label, cls, geoDomainSet }) {
    const status = getStatus(panel, `${label} results unavailable.`, 5);
    const items  = panel.data?.results || [];
    return (
      <div className="cmp-col">
        <ColHeader label={label} cls={cls} count={items.length} />
        {status || items.map((item) => {
          const inGeo = geoDomainSet.has(domainFromUrl(item.url));
          return (
            <div key={item.url} className={`cmp-row${inGeo ? " cmp-row-shared" : ""}`}>
              <span className="cmp-rank">{item.rank || item.position}</span>
              <div className="cmp-body">
                <div className="cmp-domain">{item.display_url || domainFromUrl(item.url)}</div>
                <a className="cmp-title" href={item.url} target="_blank" rel="noreferrer">
                  {item.title || item.url}
                </a>
                {item.snippet && <p className="cmp-snippet">{item.snippet}</p>}
                {inGeo && <span className="cmp-also-geo">In GeoSearch</span>}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <section>
      <div className="pane-intro">
        <h2>
          Compare Engines
          <span className="info-tip" aria-label="GeoSearch BM25 results alongside live Google and Bing results for the same query. Shared domains are highlighted across columns.">?</span>
        </h2>
      </div>
      <div className="cmp-grid">
        <GeoCol />
        <ExCol panel={google} label="Google" cls="cmp-google" geoDomainSet={geoDomains} />
        <ExCol panel={bing}   label="Bing"   cls="cmp-bing"   geoDomainSet={geoDomains} />
      </div>
    </section>
  );
}

/* ================================================================
   ABOUT VIEW
   ================================================================ */
function AboutView({ onBack }) {
  return (
    <div className="about">
      <button className="about-back" type="button" onClick={onBack}>← Back to search</button>
      <div className="about-hd">
        <h2>About GeoSearch</h2>
        <p>
          GeoSearch is a domain-specific search engine for geology, built for UTD CS 6322 (Information Retrieval).
          The interface you are using embeds five distinct result frames: four of our own relevance models, a clustered
          view, a query-expansion view, and live Google and Bing results for side-by-side comparison.
        </p>
      </div>

      <div className="about-q">
        <h3>How is the interface designed?</h3>
        <p>
          The interface is a minimal light-mode single-page React app. The landing page is intentionally sparse —
          a single serif wordmark, one-line tagline, a pill-shaped search input, and six demo queries — so nothing
          competes with the query itself. Once a user searches, a sticky top bar keeps the query editable at all
          times, a large serif header restates the query, and five tabs present each result frame.
        </p>
        <p>
          Visual hierarchy on the results page follows three layers: (1) the query header (serif, large),
          (2) section intros that explain the ranker or expansion method in one sentence, (3) result cards with
          rank, URL, serif title, snippet, and a right-aligned score bar so rankings are scannable at a glance.
          Typography is Inter for UI and Fraunces for content, with near-black text on white and a single dark
          accent; badges and cluster tags provide color only where they carry information.
        </p>
      </div>

      <div className="about-q">
        <h3>How did you work with the student that built the index?</h3>
        <p>
          The indexing student owns <code>indexer/src/</code> and exposes a <code>SearchEngine</code> class that loads
          the inverted index, document store, and PageRank scores from <code>indexer/data/</code> at startup. We agreed
          on a stable FastAPI contract at <code>/api/search</code> (<code>query</code>, <code>method</code>, <code>top_k</code>)
          returning a ranked list of <code>{`{rank, url, title, snippet, display_url, score}`}</code>. The frontend is
          completely decoupled from index internals — the backend proxy (<code>:8020</code>) orchestrates the search
          backend, the cluster service, and the SerpAPI bridge, which allowed us to iterate on the UI while the index
          team swapped in BM25, tuned PageRank damping, and added HITS.
        </p>
      </div>

      <div className="about-q">
        <h3>How are the relevance models accessed?</h3>
        <p>
          When a user submits a query, the UI fires one <code>POST /api/search</code> per method (BM25, TF-IDF, PageRank,
          HITS) in parallel. Each tab renders the matching panel as soon as its response lands, so the user can start
          reading the first model without waiting on the slowest one. The same request path also powers query expansion
          via <code>POST /api/expand</code> and clustered reranking via <code>POST /api/clustered-search</code>.
        </p>
      </div>

      <div className="about-q">
        <h3>How many queries were used for testing?</h3>
        <p>
          We tested with <strong>50 benchmark queries</strong> drawn from <code>cluster_service/benchmarks/queries_50.json</code>.
          Of these:
        </p>
        <ul>
          <li><strong>30 queries</strong> were co-authored with the relevance-model student to probe specific ranker
            behaviours (rare-term recall, BM25 vs TF-IDF divergence, link-heavy topics for PageRank, authoritative
            hubs for HITS).</li>
          <li><strong>20 queries</strong> were generated independently by the frontend team as representative user
            searches — mix of single-term, phrase, and natural-language questions drawn from geology textbook indexes
            and USGS topic pages.</li>
        </ul>
      </div>

      <div className="about-q">
        <h3>How did you collaborate with the clustering student?</h3>
        <p>
          The clustering student runs a separate service on port 8010 that accepts a list of results and returns the
          same results annotated with a <code>cluster_id</code>, <code>cluster_name</code>, and a new score, plus a
          <code>clusters</code> list with representatives. We agreed that the UI would always call the baseline ranker
          first and pass those results into <code>/rerank</code>, so clustering never replaces relevance — it
          <em> reorders</em> within the top-k. The interface uses this contract three ways: a left-hand cluster sidebar
          acts as a filter, each result card carries a cluster tag, and representative documents for the active cluster
          appear below the list.
        </p>
      </div>

      <div className="about-q">
        <h3>How does clustering shape the presentation?</h3>
        <p>
          Clustering is surfaced as a navigable dimension rather than a different ranking. The sidebar lists every
          cluster with its color, label, and count; clicking one filters the result list to that topic. The reranked
          score formula is shown at the bottom of the pane so the user can see the weighting
          (<code>baseline × 0.7 + cluster_affinity × 0.2 + cluster_support × 0.1</code>), and any movement from the
          baseline rank is shown on each card as <code>↑ +N</code> or <code>↓ -N</code>. Cluster representatives are
          shown when a cluster is active, giving a concise "what's in this topic?" summary.
        </p>
      </div>

      <div className="about-q">
        <h3>How does GeoSearch compare to Google and Bing?</h3>
        <p>
          Google and Bing have an effectively unbounded index and years of learned ranking signals, so for broad
          informational queries ("volcanic eruption hawaii") they retrieve better-known authorities faster. For
          domain-scoped queries inside our crawl ("metamorphic facies of the Wopmay orogen", "garnet almandine
          zoning textures") GeoSearch is often comparable and sometimes better: because the corpus is restricted
          to geology sources, noisy commercial results don't dilute the top-k, and PageRank plus HITS surface
          authoritative hubs within the subgraph. Clustering adds something the commercial engines don't expose —
          a visible topical decomposition of the result set — and query expansion lets the user see what vocabulary
          the system is inferring. Overall, we do not try to beat Google and Bing on coverage; we aim to beat them
          on <em>interpretability</em> within the geology domain.
        </p>
      </div>

      <div className="about-q">
        <h3>How did you select the demonstration queries?</h3>
        <p>
          We picked queries that exercise each system component: a broad query that benefits from expansion, a
          multi-topic query that clusters split cleanly, and a link-heavy query where PageRank pulls up authoritative
          hubs. We also made sure each query returns non-empty results on both Google and Bing so comparison is
          meaningful.
        </p>

        <div className="example-block">
          <strong>1. "volcanic eruption hawaii"</strong>
          <dl className="example-rows">
            <dt>GeoSearch</dt><dd>USGS HVO pages, Smithsonian GVP entries, university field guides — top-3 are all government/academic.</dd>
            <dt>Google</dt><dd>News articles, Wikipedia, USGS, travel sites mixed in top-10.</dd>
            <dt>Bing</dt><dd>Similar to Google with more video/image results interleaved.</dd>
          </dl>
        </div>

        <div className="example-block">
          <strong>2. "metamorphic rock classification"</strong>
          <dl className="example-rows">
            <dt>GeoSearch</dt><dd>Textbook chapter pages, GeoSciWorld, course notes; clusters split into "protolith-based" vs "grade-based" classification.</dd>
            <dt>Google</dt><dd>Wikipedia, Britannica, tutoring sites.</dd>
            <dt>Bing</dt><dd>Wikipedia, encyclopedia.com, educational video transcripts.</dd>
          </dl>
        </div>

        <div className="example-block">
          <strong>3. "earthquake fault san andreas"</strong>
          <dl className="example-rows">
            <dt>GeoSearch</dt><dd>USGS earthquake hazard pages, SCEC research, peer-reviewed summaries. Expansion adds "strike-slip", "transform", "tremor".</dd>
            <dt>Google</dt><dd>USGS, Wikipedia, LA Times, National Geographic.</dd>
            <dt>Bing</dt><dd>USGS, Wikipedia, history.com, news features.</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   ROOT APP
   ================================================================ */
export default function App() {
  const initial      = readInitialState();
  const requestIdRef = useRef(0);

  const [queryInput,       setQueryInput]       = useState(initial.query);
  const [topK,             setTopK]             = useState(initial.topK);
  const [expansionMethod,  setExpansionMethod]  = useState(initial.expansionMethod);
  const [clusterMethod,    setClusterMethod]    = useState(initial.clusterMethod);
  const [activeTab,        setActiveTab]        = useState(initial.tab);
  const [activeModel,      setActiveModel]      = useState("bm25");
  const [activeClusterId,  setActiveClusterId]  = useState(null);
  const [results,          setResults]          = useState(createInitialResults);
  const [lastSearch,       setLastSearch]       = useState(initial.query);
  const [view,             setView]             = useState(initial.view);

  const hasSearched = !!lastSearch;

  const tabCounts = useMemo(() => ({
    models:    results.models[activeModel]?.data?.results?.length,
    clustered: results.clustered.data?.reranked?.length,
    expanded:  results.expanded.data?.results?.length,
    compare:   results.google.status === "success" || results.bing.status === "success" ? 3 : undefined,
  }), [results, activeModel]);

  useEffect(() => {
    if (!initial.query) return;
    void runSearch(initial.query);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function patchResults(requestId, updater) {
    if (requestIdRef.current !== requestId) return;
    startTransition(() => setResults((cur) => updater(cur)));
  }

  function changeTab(next) {
    startTransition(() => setActiveTab(next));
    syncUrl({ query: lastSearch || queryInput.trim(), topK, expansionMethod, clusterMethod, tab: next, view });
  }

  function goHome() {
    setView("search");
    setLastSearch("");
    setQueryInput("");
    setResults(createInitialResults());
    syncUrl({ query: "", topK, expansionMethod, clusterMethod, tab: activeTab, view: "search" });
  }

  async function runSearch(explicitQuery) {
    const q = (explicitQuery ?? queryInput).trim();
    if (!q) return;

    const rid = requestIdRef.current + 1;
    requestIdRef.current = rid;
    setLastSearch(q);
    setActiveClusterId(null);
    setView("search");
    syncUrl({ query: q, topK, expansionMethod, clusterMethod, tab: activeTab, view: "search" });

    startTransition(() =>
      setResults({
        models: Object.fromEntries(
          MODEL_CONFIG.map((m) => [m.key, { status: "loading", data: null, error: "" }]),
        ),
        expanded:  { status: "loading", data: null, error: "" },
        clustered: { status: "loading", data: null, error: "" },
        google:    { status: "loading", data: null, error: "" },
        bing:      { status: "loading", data: null, error: "" },
      }),
    );

    MODEL_CONFIG.forEach((m) => {
      requestJson("/api/search", { method: "POST", body: { query: q, method: m.key, top_k: topK } })
        .then((data) =>
          patchResults(rid, (cur) => ({
            ...cur,
            models: { ...cur.models, [m.key]: { status: "success", data, error: "" } },
          })),
        )
        .catch((err) =>
          patchResults(rid, (cur) => ({
            ...cur,
            models: { ...cur.models, [m.key]: { status: "error", data: null, error: err.message } },
          })),
        );
    });

    requestJson("/api/expand", { method: "POST", body: { query: q, method: expansionMethod, top_k: topK } })
      .then((data) => patchResults(rid, (cur) => ({ ...cur, expanded:  { status: "success", data, error: "" } })))
      .catch((err) => patchResults(rid, (cur) => ({ ...cur, expanded:  { status: "error", data: null, error: err.message } })));

    requestJson("/api/clustered-search", {
      method: "POST",
      body: { query: q, cluster_method: clusterMethod, baseline_method: "combined", top_k: topK },
    })
      .then((data) => patchResults(rid, (cur) => ({ ...cur, clustered: { status: "success", data, error: "" } })))
      .catch((err) => patchResults(rid, (cur) => ({ ...cur, clustered: { status: "error", data: null, error: err.message } })));

    ["google", "bing"].forEach((engine) => {
      requestJson("/api/external-search", { method: "POST", body: { engine, query: q, top_k: topK } })
        .then((data) => patchResults(rid, (cur) => ({ ...cur, [engine]: { status: "success", data, error: "" } })))
        .catch((err) => patchResults(rid, (cur) => ({ ...cur, [engine]: { status: "error", data: null, error: err.message } })));
    });
  }

  function onSubmit(e) {
    e.preventDefault();
    void runSearch();
  }

  if (view === "about") {
    return (
      <div className="w">
        <AboutView onBack={() => setView("search")} />
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="w">
        <Landing
          queryInput={queryInput}
          setQueryInput={setQueryInput}
          onSubmit={onSubmit}
          onAbout={() => setView("about")}
        />
      </div>
    );
  }

  return (
    <div className="w">
      <TopBar
        queryInput={queryInput}
        setQueryInput={setQueryInput}
        onSubmit={onSubmit}
        topK={topK}
        setTopK={setTopK}
        expansionMethod={expansionMethod}
        setExpansionMethod={setExpansionMethod}
        clusterMethod={clusterMethod}
        setClusterMethod={setClusterMethod}
        onHome={goHome}
      />
      <TabBar activeTab={activeTab} setActiveTab={changeTab} counts={tabCounts} />

      <main className={`content${activeTab === "clustered" ? " split" : ""}`}>
        {activeTab === "models" && (
          <ModelsPane results={results} activeModel={activeModel} setActiveModel={setActiveModel} />
        )}
        {activeTab === "clustered" && (
          <ClusteredPane
            panel={results.clustered}
            activeClusterId={activeClusterId}
            setActiveClusterId={setActiveClusterId}
            clusterMethod={clusterMethod}
          />
        )}
        {activeTab === "expanded" && (
          <ExpandedPane
            panel={results.expanded}
            searchQuery={lastSearch}
            expansionMethod={expansionMethod}
          />
        )}
        {activeTab === "compare" && (
          <ComparePane results={results} />
        )}
      </main>
    </div>
  );
}
