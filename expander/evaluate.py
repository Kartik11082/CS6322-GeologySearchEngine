"""Expansion evaluation script.

Run with spaCy available (morphological lemmas in reports), e.g.::

    conda run -n nlp python expander/evaluate.py

Generates ``expansion_report.txt`` aligned with the X5 rubric:
  - 20 Rocchio queries (explicit feedback; judgment protocol documented in-report)
  - 50 PRF queries from ``cluster_service/benchmarks/queries_50.json`` (shared with X2/X3)
  - PRF parameters match ``backend/app.py`` (top_k_docs=50, HITS post-expansion by default).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "expander"))

from search import SearchEngine
from core import QueryExpander
from query_sets import (
    ROCCHIO_QUERIES,
    load_prf_benchmark_rows,
    m_neighbors_for_query,
)

# Match backend/app.py perform_expansion for PRF paths.
TOP_K_DOCS_LOCAL = 50
MAX_NEW_TERMS_LOCAL = 5
ROCCHIO_ALPHA = 1.0
ROCCHIO_BETA = 0.75
ROCCHIO_GAMMA = 0.25
ROCCHIO_NUM_NEW_TERMS = 5
POST_EXPAND_SEARCH_METHOD = "hits"
POST_EXPAND_TOP_K = 10


def _doc_summary(engine: SearchEngine, doc_id: str) -> tuple[str, str]:
    doc = engine.doc_store.get(str(doc_id), {})
    return (doc.get("title", "").strip() or "(no title)", doc.get("url", "").strip() or "(no url)")


def write_top10_results(out, engine: SearchEngine, query_text: str, label: str) -> None:
    """Write top-k ranked doc_ids and URLs (same method/top_k as /api/expand defaults)."""
    results = engine.search(
        query_text, method=POST_EXPAND_SEARCH_METHOD, top_k=POST_EXPAND_TOP_K
    )
    out.write(
        f"{label} (Top {POST_EXPAND_TOP_K} {POST_EXPAND_SEARCH_METHOD.upper()})\n"
    )
    if not results:
        out.write("  [no results]\n")
        return
    for r in results:
        out.write(f"  {r['rank']:>2}. doc_id={r['doc_id']} | {r.get('url', '')}\n")


def write_rubric_preamble(out) -> None:
    out.write(
        "\n".join(
            [
                "RUBRIC COVERAGE (Student X5 — Query expansion)",
                "-" * 70,
                "",
                "1) TWENTY ROCCHIO QUERIES — selection rationale",
                "   We use 20 strings chosen to stress explicit feedback:",
                "   - Core geology topics (volcano, earthquake, fossils, minerals, etc.).",
                "   - Multi-word technical queries (magma composition, tectonic plates).",
                "   - Intentional typos (erthsquake, volcnos) to show spelling correction",
                "     via normalized/stem matching before Rocchio.",
                "   - Named places/features (Yellowstone, San Andreas, Vesuvius, pahoehoe).",
                "   - Four off-domain negatives (software, AI, stocks, football) so gamma",
                "     can pull the vector away from irrelevant centroid when those appear",
                "     in the pseudo-labeled pool.",
                "",
                "2) RELEVANT vs IRRELEVANT pages (batch protocol for this report)",
                "   For reproducibility without manual labeling in CI, each query uses an",
                "   initial BM25 run (top 3). Ranks 1–2 are treated as RELEVANT and rank 3",
                "   as NON-RELEVANT for Rocchio inputs. Below, each query lists titles/URLs",
                "   so you (or X3) can replace doc_ids with true judgments in production.",
                "   For graded reports, paste 2–3 fully manual examples into your PDF and",
                "   reference the URLs printed here.",
                "",
                "3) FIFTY PSEUDO-RELEVANCE (PRF) QUERIES",
                "   Loaded from: cluster_service/benchmarks/queries_50.json",
                "   This is the shared benchmark set (collaborate with X2 for corpus/qrels,",
                "   X3 for UI). It is disjoint in *role* from the Rocchio suite: those 20",
                "   use explicit labels; these 50 use only top-retrieved docs inside the",
                "   expander (association / scalar / metric).",
                "",
                "4) PRF + post-expansion search parameters",
                f"   Local doc set: top {TOP_K_DOCS_LOCAL} HITS (see core.QueryExpander).",
                f"   m_neighbors: 2 if <=3 query tokens else 6 (matches backend).",
                f"   max_new_terms: {MAX_NEW_TERMS_LOCAL} (matches backend).",
                f"   Results after expansion: {POST_EXPAND_TOP_K} {POST_EXPAND_SEARCH_METHOD.upper()}.",
                "",
                "=" * 70,
                "",
            ]
        )
    )


def run_experiments() -> None:
    print("Loading Search Engine... This might take ~5 seconds.")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)

    prf_rows = load_prf_benchmark_rows()
    prf_pairs = [(str(r["query_id"]), str(r["query_text"])) for r in prf_rows]

    report_path = PROJECT_ROOT / "expander" / "expansion_report.txt"

    with open(report_path, "w", encoding="utf-8") as out:
        out.write("=" * 70 + "\n")
        out.write("GEOLOGY SEARCH ENGINE - QUERY EXPANSION EVALUATION REPORT\n")
        out.write("Student X5 deliverable — Rocchio (20) + PRF clustering (50 benchmark)\n")
        out.write("=" * 70 + "\n\n")
        write_rubric_preamble(out)

        out.write("LIST: 20 ROCCHIO QUERIES\n")
        out.write("-" * 40 + "\n")
        for i, q in enumerate(ROCCHIO_QUERIES, 1):
            out.write(f"  {i:2}. {q}\n")
        out.write("\n")

        out.write("LIST: 50 PRF BENCHMARK QUERIES (query_id | text)\n")
        out.write("-" * 40 + "\n")
        for qid, qtext in prf_pairs:
            out.write(f"  {qid}  {qtext}\n")
        out.write("\n")

        # ----------------------------------------------------------------
        # EXPERIMENT 1: ROCCHIO
        # ----------------------------------------------------------------
        out.write("=" * 60 + "\n")
        out.write("EXPERIMENT 1: ROCCHIO RELEVANCE FEEDBACK (20 QUERIES)\n")
        out.write("=" * 60 + "\n\n")

        print("Running Rocchio Evaluation against 20 benchmark queries...")
        for q in ROCCHIO_QUERIES:
            normalized_q = expander.normalize_query(q)
            initial_results = engine.search(normalized_q, method="bm25", top_k=3)
            rel_docs = [str(res["doc_id"]) for res in initial_results[:2]]
            irrel_docs = [str(res["doc_id"]) for res in initial_results[2:3]]

            expanded = expander.expand_rocchio(
                q,
                relevant_doc_ids=rel_docs,
                irrelevant_doc_ids=irrel_docs,
                alpha=ROCCHIO_ALPHA,
                beta=ROCCHIO_BETA,
                gamma=ROCCHIO_GAMMA,
                num_new_terms=ROCCHIO_NUM_NEW_TERMS,
            )

            out.write(f"Original Query   : {q}\n")
            out.write(f"Normalized Query : {normalized_q}\n")
            out.write("Relevant doc_ids (ranks 1-2, BM25)    : " + ", ".join(rel_docs) + "\n")
            out.write("Non-relevant doc_id (rank 3, BM25)   : " + ", ".join(irrel_docs) + "\n")
            out.write("Relevant pages:\n")
            for rid in rel_docs:
                title, url = _doc_summary(engine, rid)
                out.write(f"  - [{rid}] {title}\n    {url}\n")
            out.write("Non-relevant page (as labeled here):\n")
            for uid in irrel_docs:
                title, url = _doc_summary(engine, uid)
                out.write(f"  - [{uid}] {title}\n    {url}\n")
            out.write(
                "Judgment note: ranks 1-2 assumed relevant (strongest BM25 match to the "
                "query); rank 3 weaker tail used as a single negative for Rocchio. "
                "Replace with human labels for final demo.\n"
            )
            out.write(f"Expanded Query   : {expanded}\n")
            write_top10_results(out, engine, normalized_q, "Original Query Results")
            write_top10_results(out, engine, expanded, "Expanded Query Results")
            out.write("-" * 40 + "\n")

        # ----------------------------------------------------------------
        # EXPERIMENT 2: LOCAL CLUSTERING / PRF
        # ----------------------------------------------------------------
        out.write("\n" + "=" * 60 + "\n")
        out.write("EXPERIMENT 2: LOCAL CLUSTERING / PRF (50 BENCHMARK QUERIES)\n")
        out.write("=" * 60 + "\n\n")

        print(f"Running local clustering on {len(prf_pairs)} benchmark queries...")
        for i, (qid, q) in enumerate(prf_pairs, 1):
            if i % 10 == 0:
                print(f"Processed {i}/{len(prf_pairs)} queries...")

            m_n = m_neighbors_for_query(q)
            assoc = expander.expand_association(
                q,
                top_k_docs=TOP_K_DOCS_LOCAL,
                m_neighbors=m_n,
                normalized=True,
                max_new_terms=MAX_NEW_TERMS_LOCAL,
            )
            scalar = expander.expand_scalar(
                q,
                top_k_docs=TOP_K_DOCS_LOCAL,
                m_neighbors=m_n,
                max_new_terms=MAX_NEW_TERMS_LOCAL,
            )
            metric = expander.expand_metric(
                q,
                top_k_docs=TOP_K_DOCS_LOCAL,
                m_neighbors=m_n,
                max_new_terms=MAX_NEW_TERMS_LOCAL,
            )

            out.write(f"Query {qid} ({i}/50): {q}\n")
            out.write(f"Association      : {assoc}\n")
            out.write(f"Scalar           : {scalar}\n")
            out.write(f"Metric           : {metric}\n")
            write_top10_results(out, engine, assoc, "Association Results")
            write_top10_results(out, engine, scalar, "Scalar Results")
            write_top10_results(out, engine, metric, "Metric Results")
            out.write("-" * 60 + "\n")

    print(f"\n✅ Experiments complete! Results saved to: {report_path}")


if __name__ == "__main__":
    run_experiments()
