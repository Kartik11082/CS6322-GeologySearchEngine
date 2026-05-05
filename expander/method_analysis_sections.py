"""Build text blocks shared by association / scalar / metric analysis scripts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from preprocessor import preprocess


def local_analysis_context(
    expander: Any,
    engine: Any,
    query: str,
    top_k: int = 50,
) -> dict[str, Any]:
    normalized_query = expander._normalize_query_for_expansion(query)
    query_stems = preprocess(query)
    local_doc_ids = expander._get_local_doc_set(query, top_k)
    sorted_ids = sorted(local_doc_ids)
    local_tf = expander._get_local_term_frequencies(local_doc_ids)
    return {
        "query": query,
        "normalized_query": normalized_query,
        "query_stems": query_stems,
        "local_doc_ids": local_doc_ids,
        "sorted_ids": sorted_ids,
        "local_tf": local_tf,
        "engine": engine,
        "expander": expander,
    }


def format_lds_section(ctx: dict[str, Any]) -> str:
    engine = ctx["engine"]
    local_doc_ids_sorted = ctx["sorted_ids"]
    normalized_query = ctx["normalized_query"]
    local_doc_ids = ctx["local_doc_ids"]
    lines: list[str] = []
    lines.append(f"Number of documents: {len(local_doc_ids_sorted)}")
    lines.append(
        f"Method: Top-50 BM25 results for normalized query '{normalized_query}'"
    )
    lines.append("")
    lines.append("Document IDs:")
    for i in range(0, len(local_doc_ids_sorted), 10):
        chunk = local_doc_ids_sorted[i : i + 10]
        lines.append(f"    {', '.join(chunk)}")
    lines.append("")
    lines.append("Sample Documents with URLs:")
    for doc_id in local_doc_ids_sorted[:8]:
        doc = engine.doc_store.get(str(doc_id), {})
        lines.append(f"    [{doc_id}] {doc.get('url', 'N/A')}")
    return "\n".join(lines)


def format_stems_section(ctx: dict[str, Any]) -> str:
    query = ctx["query"]
    query_stems = ctx["query_stems"]
    local_tf = ctx["local_tf"]
    alpha_vocab = sorted(t for t in local_tf.keys() if t.isalpha())
    lines: list[str] = []
    lines.append(f'Query: "{query}"')
    lines.append(f"Query stems (after preprocessing): {query_stems}")
    lines.append("")
    lines.append(f"Total unique stems in local vocabulary: {len(local_tf)}")
    lines.append("")
    lines.append("Local vocabulary sample (first 80 alphabetical stems):")
    for i in range(0, min(80, len(alpha_vocab)), 10):
        chunk = alpha_vocab[i : i + 10]
        lines.append(f"    {', '.join(chunk)}")
    lines.append("")
    lines.append("Vocabulary statistics:")
    lines.append(f"    - Total terms in local set: {len(local_tf)}")
    lines.append(f"    - Alphabetic terms only: {len(alpha_vocab)}")
    lines.append(
        f"    - Terms appearing in 2+ local docs: {sum(1 for _, p in local_tf.items() if len(p) >= 2)}"
    )
    return "\n".join(lines)


def format_association_correlations(
    ctx: dict[str, Any],
    expander: Any,
) -> str:
    query_stems = ctx["query_stems"]
    local_doc_ids = ctx["local_doc_ids"]
    local_tf = ctx["local_tf"]
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    lines: list[str] = []
    lines.append("Formula: C(u,v) = Σ [f(u,j) × f(v,j)] for all docs j in local set")
    lines.append("Normalized by: C(u,v) / (Σf(u) × Σf(v))")
    lines.append("")

    for q_term in query_stems:
        if q_term not in local_tf:
            lines.append(f"\nQuery stem '{q_term}': Not found in local vocabulary")
            continue

        q_sum_f = sum(local_tf[q_term].values())
        lines.append(f"\nQuery stem: '{q_term}'")
        lines.append(f"    Frequency sum Σf({q_term}) = {q_sum_f}")
        lines.append(f"    Appears in {len(local_tf[q_term])} local documents")
        lines.append("")
        lines.append(f"    {'Candidate Term':<20} {'Raw C(u,v)':>12} {'Normalized':>12}")
        lines.append(f"    {'-'*20} {'-'*12} {'-'*12}")

        correlations = []
        for v_term, v_postings in local_tf.items():
            if not expander._is_candidate_term(
                v_term, query_stem_set, query_len=query_len
            ):
                continue
            raw_c_uv = sum(
                local_tf[q_term].get(d, 0) * v_postings.get(d, 0)
                for d in local_doc_ids
            )
            v_sum_f = sum(v_postings.values())
            norm_c_uv = (
                raw_c_uv / (q_sum_f * v_sum_f) if (q_sum_f * v_sum_f) > 0 else 0
            )
            if raw_c_uv > 0:
                correlations.append((v_term, raw_c_uv, norm_c_uv))

        correlations.sort(key=lambda x: -x[2])

        for term, raw, norm in correlations[:12]:
            lines.append(f"    {term:<20} {raw:>12.2f} {norm:>12.6f}")

    lines.append("\n" + "-" * 40)
    lines.append("CLUSTER SELECTION DISCUSSION:")
    lines.append("-" * 40)
    lines.append("""
Normalized co-occurrence C(u,v) / (Σf(u)Σf(v)) picks terms statistically
associated with query stems inside the BM25-local document set D_l.

expand_association uses normalized correlation, candidate filtering (_is_candidate_term),
per-stem neighbor top-m selection, IDF boosting, then _finalize_expansion merges deduplicated lemmas.
""")
    return "\n".join(lines)


def format_scalar_correlations(ctx: dict[str, Any], expander: Any) -> str:
    query_stems = ctx["query_stems"]
    local_doc_ids = ctx["local_doc_ids"]
    local_tf = ctx["local_tf"]
    terms = list(local_tf.keys())
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    lines: list[str] = []
    lines.append("Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|)")
    lines.append(
        "Where s_u[j] = f(u, doc_j) for doc j in local set (document-frequency vectors on D_l)."
    )
    lines.append("")

    for q_term in query_stems:
        if q_term not in local_tf:
            lines.append(f"\nQuery stem '{q_term}': Not found in local vocabulary")
            continue

        norm_u = pow(
            sum(local_tf[q_term].get(d, 0) ** 2 for d in local_doc_ids), 0.5
        )
        lines.append(f"\nQuery stem: '{q_term}'")
        lines.append(f"    Association vector norm |s_{q_term}| = {norm_u:.4f}")
        lines.append(f"    Appears in {len(local_tf[q_term])} local documents")
        lines.append("")
        lines.append(f"    {'Candidate Term':<20} {'Dot Product':>14} {'|s_v|':>10} {'S(u,v)':>12}")
        lines.append(f"    {'-'*20} {'-'*14} {'-'*10} {'-'*12}")

        if norm_u == 0:
            lines.append("    [No valid correlations - zero norm]")
            continue

        correlations = []
        for v_term in terms:
            if not expander._is_candidate_term(
                v_term, query_stem_set, query_len=query_len
            ):
                continue

            sq = sum(
                local_tf[q_term].get(d, 0) * local_tf[v_term].get(d, 0)
                for d in local_doc_ids
            )
            s_uv = expander.association_cosine_uv(
                q_term, v_term, local_doc_ids, local_tf
            )
            norm_v = pow(
                sum(local_tf[v_term].get(d, 0) ** 2 for d in local_doc_ids), 0.5
            )
            if s_uv > 0:
                correlations.append((v_term, sq, norm_v, s_uv))

        correlations.sort(key=lambda x: -x[3])
        for term, dot, norm_v, s_uv in correlations[:12]:
            lines.append(f"    {term:<20} {dot:>14.2f} {norm_v:>10.2f} {s_uv:>12.6f}")

    lines.append("\n" + "-" * 40)
    lines.append("CLUSTER SELECTION DISCUSSION:")
    lines.append("-" * 40)
    lines.append("""
Scalar cosine on document-frequency vectors matches expand_scalar in core.py:

    S(u,v) = Σ_d f(u,d)f(v,d) / (sqrt(Σ_d f(u,d)^2) sqrt(Σ_d f(v,d)^2))

Neighborhoods ranked by cosine are then boosted by expander._score_expansion_term (IDF, etc.).
""")
    return "\n".join(lines)


def format_metric_correlations(ctx: dict[str, Any], expander: Any) -> str:
    query_stems = ctx["query_stems"]
    local_doc_ids = ctx["local_doc_ids"]
    engine = ctx["engine"]
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    lines: list[str] = []
    lines.append("Formula: C(u,v) = Σ [1 / |pos(u) - pos(v)|]")
    lines.append("Summed across all documents and position pairs")
    lines.append("")

    global_correlations = defaultdict(lambda: defaultdict(float))
    docs_with_positions = 0

    for d_id in local_doc_ids:
        doc = engine.doc_store.get(str(d_id), {})
        doc_tokens = preprocess(doc.get("text_preview", ""))
        if not doc_tokens:
            continue
        docs_with_positions += 1

        positions = defaultdict(list)
        for i, token in enumerate(doc_tokens):
            positions[token].append(i)

        for q_term in query_stems:
            if q_term not in positions:
                continue
            for v_term, v_pos_list in positions.items():
                if not expander._is_candidate_term(
                    v_term, query_stem_set, query_len=query_len
                ):
                    continue

                for q_pos in positions[q_term]:
                    for v_pos in v_pos_list:
                        distance = abs(q_pos - v_pos)
                        if distance > 0:
                            global_correlations[q_term][v_term] += 1.0 / distance

    lines.append(f"Documents with position data: {docs_with_positions}")

    for q_term in query_stems:
        if q_term not in global_correlations:
            lines.append(f"\nQuery stem '{q_term}': No position data found")
            continue

        lines.append(f"\nQuery stem: '{q_term}'")
        lines.append(
            f"    Co-located with {len(global_correlations[q_term])} candidate terms"
        )
        lines.append("")
        lines.append(
            f"    {'Candidate Term':<20} {'Σ(1/distance)':>15} {'Interpretation':<30}"
        )
        lines.append(f"    {'-'*20} {'-'*15} {'-'*30}")

        correlations = sorted(global_correlations[q_term].items(), key=lambda x: -x[1])[
            :15
        ]

        for term, score in correlations:
            if score > 5:
                interp = "Very close proximity"
            elif score > 2:
                interp = "Close proximity"
            elif score > 1:
                interp = "Moderate proximity"
            else:
                interp = "Distant"
            lines.append(f"    {term:<20} {score:>15.6f} {interp:<30}")

    lines.append("\n" + "-" * 40)
    lines.append("CLUSTER SELECTION DISCUSSION:")
    lines.append("-" * 40)
    lines.append("""
Proximity on preprocess(text_preview) matches expand_metric in core.py: inverse word distance summed
within each snippet, gated by expander._is_candidate_term before IDF boosting and final lemmas.
""")
    return "\n".join(lines)
