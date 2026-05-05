"""Unified search interface — combines index, relevance models, PageRank, HITS."""

import argparse
import time
from typing import Any

from config import DEFAULT_TOP_K, SNIPPET_CHAR_LIMIT
from graph import WebGraph, hits, load_pagerank, pagerank, save_pagerank
from index import build_index, load_index, save_index
from loader import build_url_to_docid, load_edges, load_pages
from relevance import rank_bm25, rank_tfidf


class SearchEngine:
    """
    Geology search engine combining:
      - TF-IDF cosine similarity
      - BM25
      - PageRank (query-independent boost)
      - HITS (query-dependent authority/hub scores)
    """

    def __init__(self) -> None:
        self.inverted_index: dict = {}
        self.doc_store: dict = {}
        self.N: int = 0
        self.avg_dl: float = 0.0
        self.graph: WebGraph | None = None
        self.pagerank: dict[int, float] = {}

    # ── building from raw data ────────────────────────────────────

    def build(self) -> None:
        """Build everything from the crawler_new data files."""
        print("-- Loading crawler data --")
        pages = load_pages()
        edges = load_edges()
        url_map = build_url_to_docid(pages)
        print(f"   {len(pages)} pages, {len(edges)} edges")

        print("-- Building inverted index --")
        t0 = time.time()
        self.inverted_index, self.doc_store, self.N, self.avg_dl = build_index(pages)
        print(
            f"   {len(self.inverted_index):,} terms, avg_dl={self.avg_dl:.1f}  ({time.time() - t0:.2f}s)"
        )

        print("-- Building web graph --")
        t0 = time.time()
        self.graph = WebGraph.build_from_data(pages, edges, url_map)
        stats = self.graph.stats()
        print(
            f"   {stats['num_nodes']} nodes, {stats['num_edges']} edges  ({time.time() - t0:.2f}s)"
        )
        print(
            f"   max in-degree={stats['max_in_degree']}, max out-degree={stats['max_out_degree']}"
        )

        print("-- Computing PageRank --")
        t0 = time.time()
        self.pagerank = pagerank(self.graph)
        print(f"   done ({time.time() - t0:.2f}s)")

        # persist
        save_index(self.inverted_index, self.doc_store, self.N, self.avg_dl)
        self.graph.save()
        save_pagerank(self.pagerank)

    # ── loading from disk ─────────────────────────────────────────

    def load(self) -> None:
        """Load pre-built index, graph, and PageRank from disk."""
        self.inverted_index, self.doc_store, self.N, self.avg_dl = load_index()
        self.graph = WebGraph.load()
        self.pagerank = load_pagerank()
        print(
            f"Loaded index ({len(self.inverted_index):,} terms, {self.N} docs), "
            f"graph ({len(self.graph.nodes)} nodes), PageRank"
        )

    # ── search methods ────────────────────────────────────────────

    def search(
        self,
        query: str,
        method: str = "bm25",
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Search the index.

        Parameters
        ----------
        query : str
            Free-text search query.
        method : str
            One of: 'tfidf', 'bm25', 'pagerank', 'hits'.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of dicts with keys: rank, doc_id, score, url, title, snippet
        """
        if method == "tfidf":
            ranked = rank_tfidf(
                query, self.inverted_index, self.doc_store, self.N, top_k
            )
        elif method == "pagerank":
            # Pure PageRank relevance model (filtered by term matching)
            base_ranked = rank_tfidf(query, self.inverted_index, self.doc_store, self.N, top_k * 5)
            pr_only = []
            for doc_id_str, _ in base_ranked:
                pr_score = self.pagerank.get(int(doc_id_str), 0.0)
                pr_only.append((doc_id_str, pr_score))
            ranked = sorted(pr_only, key=lambda x: x[1], reverse=True)[:top_k]
            
        elif method == "hits":
            # Pure HITS relevance model
            if self.graph is None:
                return []
            auth_ranking, _ = hits(query, self.graph, self.inverted_index, top_k)
            ranked = [(str(doc_id), score) for doc_id, score in auth_ranking]
            
        elif method == "tfidf_pagerank":
            # Combination: TF-IDF + PageRank
            base_ranked = rank_tfidf(query, self.inverted_index, self.doc_store, self.N, top_k * 5)
            if base_ranked:
                max_tfidf = max(s for _, s in base_ranked) or 1.0
                combined = []
                for doc_id_str, tfidf_score in base_ranked:
                    pr_score = self.pagerank.get(int(doc_id_str), 0.0)
                    combined_score = 0.7 * (tfidf_score / max_tfidf) + 0.3 * (pr_score * 1000)
                    combined.append((doc_id_str, combined_score))
                ranked = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
            else:
                ranked = []
                
        elif method == "tfidf_hits":
            # Combination: TF-IDF + HITS
            base_ranked = rank_tfidf(query, self.inverted_index, self.doc_store, self.N, top_k * 5)
            if base_ranked and self.graph:
                auth_ranking, _ = hits(query, self.graph, self.inverted_index, top_k * 5)
                hits_map = {str(doc_id): score for doc_id, score in auth_ranking}
                
                max_tfidf = max(s for _, s in base_ranked) or 1.0
                combined = []
                for doc_id_str, tfidf_score in base_ranked:
                    hits_score = hits_map.get(doc_id_str, 0.0)
                    combined_score = 0.7 * (tfidf_score / max_tfidf) + 0.3 * hits_score
                    combined.append((doc_id_str, combined_score))
                ranked = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
            else:
                ranked = []
                
        elif method == "bm25":
            ranked = rank_bm25(
                query, self.inverted_index, self.doc_store, self.N, self.avg_dl, top_k
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return self._format_results(ranked)

    # ── formatting ────────────────────────────────────────────────

    def _format_results(self, ranked: list[tuple[str, float]]) -> list[dict[str, Any]]:
        results = []
        for rank, (doc_id_str, score) in enumerate(ranked, 1):
            doc = self.doc_store.get(doc_id_str, {})
            snippet = doc.get("text_preview", "")[:SNIPPET_CHAR_LIMIT]
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(doc_id_str),
                    "score": round(score, 6),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "snippet": snippet,
                }
            )
        return results


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Geology search engine CLI")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build index + graph + PageRank from scratch",
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="Search query")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="tfidf",
        choices=["tfidf", "pagerank", "hits", "tfidf_pagerank", "tfidf_hits", "bm25"],
        help="Ranking method (default: tfidf)",
    )
    parser.add_argument("--top", "-k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    engine = SearchEngine()

    if args.build:
        engine.build()
        print("\n[OK] Build complete.\n")

    if args.query:
        if not args.build:
            engine.load()

        print(
            f'\n-- Query: "{args.query}"  |  Method: {args.method}  |  Top {args.top} --\n'
        )
        results = engine.search(args.query, method=args.method, top_k=args.top)

        if not results:
            print("  No results found.")
        for r in results:
            print(f"  #{r['rank']:>2}  [score={r['score']:.4f}]  doc_id={r['doc_id']}")
            print(f"       {r['title']}")
            print(f"       {r['url']}")
            print(f"       {r['snippet'][:120]}...")
            print()


if __name__ == "__main__":
    main()
