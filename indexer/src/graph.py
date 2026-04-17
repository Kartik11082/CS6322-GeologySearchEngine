"""Web graph construction, PageRank, and HITS."""

import json
import math
import time
from typing import Any

import numpy as np

from config import (
    GRAPH_PATH,
    GRAPH_STATS_PATH,
    HITS_BASE_SET_EXPANSION,
    HITS_MAX_ITER,
    HITS_TOL,
    PAGERANK_DAMPING,
    PAGERANK_MAX_ITER,
    PAGERANK_PATH,
    PAGERANK_TOL,
    ensure_directories,
)
from loader import build_url_to_docid, load_edges, load_pages
from preprocessor import preprocess


# ═══════════════════════════════════════════════════════════════════
#  Web graph construction
# ═══════════════════════════════════════════════════════════════════


class WebGraph:
    """Directed graph of crawled pages."""

    def __init__(self) -> None:
        self.nodes: set[int] = set()
        # adjacency: source_doc_id → set of target_doc_ids
        self.out_links: dict[int, set[int]] = {}
        self.in_links: dict[int, set[int]] = {}

    def add_edge(self, source: int, target: int) -> None:
        """Add a directed edge source → target."""
        self.nodes.add(source)
        self.nodes.add(target)
        self.out_links.setdefault(source, set()).add(target)
        self.in_links.setdefault(target, set()).add(source)

    @staticmethod
    def build_from_data(
        pages: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        url_to_docid: dict[str, int],
    ) -> "WebGraph":
        """
        Build graph from crawler_new output.

        Edges come from web_graph.csv with ``src_url`` and ``dst_url``.
        Both URLs are resolved to doc_ids via url_to_docid.
        Only includes edges where *both* source and target are in our
        crawled page set.
        """
        graph = WebGraph()
        crawled_doc_ids = {p["doc_id"] for p in pages}

        # make sure every crawled page is a node
        for doc_id in crawled_doc_ids:
            graph.nodes.add(doc_id)

        for edge in edges:
            src_url = edge.get("src_url", "")
            dst_url = edge.get("dst_url", "")

            source_id = url_to_docid.get(src_url)
            target_id = url_to_docid.get(dst_url)

            if source_id is None or target_id is None:
                continue
            if source_id not in crawled_doc_ids or target_id not in crawled_doc_ids:
                continue
            if source_id == target_id:
                continue  # skip self-links

            graph.add_edge(source_id, target_id)

        return graph

    def stats(self) -> dict[str, Any]:
        """Compute graph statistics."""
        num_nodes = len(self.nodes)
        num_edges = sum(len(targets) for targets in self.out_links.values())

        max_out_degree = 0
        max_out_node = None
        for node, targets in self.out_links.items():
            if len(targets) > max_out_degree:
                max_out_degree = len(targets)
                max_out_node = node

        max_in_degree = 0
        max_in_node = None
        for node, sources in self.in_links.items():
            if len(sources) > max_in_degree:
                max_in_degree = len(sources)
                max_in_node = node

        avg_out = num_edges / num_nodes if num_nodes > 0 else 0
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "max_out_degree": max_out_degree,
            "max_out_degree_node": max_out_node,
            "max_in_degree": max_in_degree,
            "max_in_degree_node": max_in_node,
            "avg_out_degree": round(avg_out, 2),
        }

    # ── serialization ─────────────────────────────────────────────

    def save(self, path=GRAPH_PATH, stats_path=GRAPH_STATS_PATH) -> None:
        ensure_directories()
        data = {
            "nodes": sorted(self.nodes),
            "edges": [
                {"source": s, "target": t}
                for s, targets in self.out_links.items()
                for t in targets
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats(), f, indent=2)
        print(f"Saved graph ({len(self.nodes)} nodes) -> {path}")

    @staticmethod
    def load(path=GRAPH_PATH) -> "WebGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = WebGraph()
        for node in data["nodes"]:
            graph.nodes.add(node)
        for edge in data["edges"]:
            graph.add_edge(edge["source"], edge["target"])
        return graph


# ═══════════════════════════════════════════════════════════════════
#  PageRank (standard — no topic bias since geology_score is removed)
# ═══════════════════════════════════════════════════════════════════


def pagerank(
    graph: WebGraph,
    damping: float = PAGERANK_DAMPING,
    max_iter: int = PAGERANK_MAX_ITER,
    tol: float = PAGERANK_TOL,
) -> dict[int, float]:
    """
    Compute standard PageRank.

    Since the new data does not include a geology_score, this is a
    uniform-teleport PageRank (the classic formulation).
    """
    nodes = sorted(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # uniform teleport vector
    teleport = np.full(n, 1.0 / n, dtype=np.float64)

    pr = np.full(n, 1.0 / n, dtype=np.float64)

    for iteration in range(max_iter):
        new_pr = np.zeros(n, dtype=np.float64)

        for node in nodes:
            idx = node_to_idx[node]
            out = graph.out_links.get(node, set())
            if len(out) == 0:
                # dangling node: distribute evenly
                new_pr += pr[idx] / n
            else:
                share = pr[idx] / len(out)
                for target in out:
                    new_pr[node_to_idx[target]] += share

        new_pr = damping * new_pr + (1 - damping) * teleport

        delta = np.abs(new_pr - pr).sum()
        pr = new_pr
        if delta < tol:
            break

    return {node: float(pr[node_to_idx[node]]) for node in nodes}


def save_pagerank(scores: dict[int, float], path=PAGERANK_PATH) -> None:
    ensure_directories()
    # convert int keys to str for JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in scores.items()}, f)
    print(f"Saved PageRank scores -> {path}")


def load_pagerank(path=PAGERANK_PATH) -> dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════════
#  HITS (Hyperlink-Induced Topic Search)
# ═══════════════════════════════════════════════════════════════════


def hits(
    query: str,
    graph: WebGraph,
    inverted_index: dict[str, dict[str, int]],
    top_k: int = 10,
    max_iter: int = HITS_MAX_ITER,
    tol: float = HITS_TOL,
    expansion_limit: int = HITS_BASE_SET_EXPANSION,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Run HITS for a query.

    Steps:
    1. Use the inverted index to find a *root set* of relevant docs.
    2. Expand to a *base set* by adding pages that link to/from the root set.
    3. Iteratively compute hub and authority scores.

    Returns (authority_ranking, hub_ranking) — each a list of (doc_id, score).
    """
    query_terms = preprocess(query)
    if not query_terms:
        return [], []

    # ── Step 1: root set – docs containing any query term ─────────
    # Limit to top 200 docs by term frequency to ensure HITS runs fast
    root_scores: dict[int, int] = {}
    for term in query_terms:
        posting = inverted_index.get(term, {})
        for doc_id_str, tf in posting.items():
            doc_id = int(doc_id_str)
            root_scores[doc_id] = root_scores.get(doc_id, 0) + tf

    root_set: set[int] = {
        doc_id for doc_id, _ in sorted(
            root_scores.items(), key=lambda x: x[1], reverse=True
        )[:200]
    }

    if not root_set:
        return [], []

    # ── Step 2: expand to base set ────────────────────────────────
    base_set = set(root_set)
    for node in list(root_set):
        # add pages pointing TO this node
        for src in list(graph.in_links.get(node, set()))[:expansion_limit]:
            base_set.add(src)
        # add pages this node points TO
        for tgt in list(graph.out_links.get(node, set()))[:expansion_limit]:
            base_set.add(tgt)

    # only keep nodes that exist in the graph
    base_set = base_set & graph.nodes
    if not base_set:
        return [], []

    nodes = sorted(base_set)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # ── Step 3: iterate hub / authority scores ────────────────────
    auth = np.ones(n, dtype=np.float64)
    hub = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        new_auth = np.zeros(n, dtype=np.float64)
        new_hub = np.zeros(n, dtype=np.float64)

        # authority update: auth(p) = sum of hub(q) for all q→p
        for node in nodes:
            idx = node_to_idx[node]
            for src in graph.in_links.get(node, set()):
                if src in node_to_idx:
                    new_auth[idx] += hub[node_to_idx[src]]

        # hub update: hub(p) = sum of auth(q) for all p→q
        for node in nodes:
            idx = node_to_idx[node]
            for tgt in graph.out_links.get(node, set()):
                if tgt in node_to_idx:
                    new_hub[idx] += auth[node_to_idx[tgt]]

        # normalize
        auth_norm = np.linalg.norm(new_auth)
        hub_norm = np.linalg.norm(new_hub)
        if auth_norm > 0:
            new_auth /= auth_norm
        if hub_norm > 0:
            new_hub /= hub_norm

        if np.abs(new_auth - auth).sum() < tol and np.abs(new_hub - hub).sum() < tol:
            auth, hub = new_auth, new_hub
            break
        auth, hub = new_auth, new_hub

    authority_ranking = sorted(
        [(nodes[i], float(auth[i])) for i in range(n)],
        key=lambda x: x[1],
        reverse=True,
    )
    hub_ranking = sorted(
        [(nodes[i], float(hub[i])) for i in range(n)],
        key=lambda x: x[1],
        reverse=True,
    )

    return authority_ranking[:top_k], hub_ranking[:top_k]


# ═══════════════════════════════════════════════════════════════════
#  CLI: Build graph, compute PageRank, print stats
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from index import build_index

    print("Loading data...")
    pages = load_pages()
    edges = load_edges()
    url_map = build_url_to_docid(pages)
    print(f"  -> {len(pages)} pages, {len(edges)} edges")

    print("Building web graph...")
    t0 = time.time()
    wg = WebGraph.build_from_data(pages, edges, url_map)
    elapsed = time.time() - t0
    print(f"  -> built in {elapsed:.2f}s")

    st = wg.stats()
    print("\n-- Graph Statistics --")
    for k, v in st.items():
        print(f"  {k}: {v}")

    wg.save()

    print("\nComputing PageRank...")
    _, doc_store, _, _ = build_index(pages)
    pr = pagerank(wg)
    save_pagerank(pr)

    top10 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n-- Top 10 PageRank --")
    for doc_id, score in top10:
        title = doc_store.get(str(doc_id), {}).get("title", "?")
        print(f"  doc_id={doc_id:>4}  score={score:.6f}  {title}")
