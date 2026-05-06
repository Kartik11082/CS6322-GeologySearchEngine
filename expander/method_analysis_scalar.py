"""
Scalar clustering analysis — writes expander/textfiles/Scalar/q{1,2,3}_*.txt
Student: Uddesh Singh | NetID: UXS230004
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "expander"))

from search import SearchEngine
from core import QueryExpander

from method_analysis_io import merge_textfiles_into_report, write_query_textfiles
from method_analysis_queries import METHOD_ANALYSIS_QUERIES
from method_analysis_sections import (
    format_expansion_section,
    format_lds_section,
    format_scalar_correlations,
    format_stems_section,
    local_analysis_context,
)
from query_sets import m_neighbors_for_query


def main() -> None:
    print("Loading Search Engine...")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)

    METHOD = "Scalar"
    for q_idx, query in enumerate(METHOD_ANALYSIS_QUERIES, start=1):
        ctx = local_analysis_context(expander, engine, query, top_k=50)
        query_stems = ctx["query_stems"]

        m_neighbors = m_neighbors_for_query(query)
        expanded = expander.expand_scalar(
            query,
            top_k_docs=50,
            m_neighbors=m_neighbors,
            max_new_terms=5,
        )
        new_terms = [
            t for t in expanded.split() if t not in set(query_stems)
        ]

        lds = format_lds_section(ctx)
        stems = format_stems_section(ctx)
        corr = format_scalar_correlations(ctx, expander)

        expansion = format_expansion_section(engine, expanded)
        write_query_textfiles(METHOD, q_idx, lds, stems, corr, expansion)

        print(f"[{METHOD}] q{q_idx} expanded: {expanded}")
        print(f"  new terms (stem set diff): {new_terms}")

    header = "\n".join(
        [
            "=" * 80,
            "SCALAR CLUSTERING - DETAILED ANALYSIS (merged from textfiles/)",
            "Student: Uddesh Singh | NetID: UXS230004",
            "=" * 80,
            "",
            "Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|) on document-frequency vectors over D_l.",
            "",
        ]
    )
    report_path = merge_textfiles_into_report(
        METHOD, header, "report_SCALAR_clustering.txt"
    )
    print(f"\n✅ textfiles/{METHOD}/ written; merged report: {report_path}")


if __name__ == "__main__":
    main()
