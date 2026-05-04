"""
Association Clustering Analysis - 3 Example Queries
Student: Uddesh Singh | NetID: UXS230004
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "expander"))

from search import SearchEngine
from core import QueryExpander
from preprocessor import preprocess

# 3 queries for Association Clustering analysis
QUERIES = [
    "pyroclastic flow danger",
    "metamorphic gneiss formation",
    "groundwater aquifer permeability"
]

def main():
    print("Loading Search Engine...")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)
    
    output_path = PROJECT_ROOT / "expander" / "report_ASSOCIATION_clustering.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        def write(text=""):
            print(text)
            f.write(text + "\n")
        
        write("=" * 80)
        write("ASSOCIATION CLUSTERING - DETAILED ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
        write()
        write("Formula: C(u,v) = Σ(d_j ∈ D_l) [f(u,j) × f(v,j)]")
        write("Normalized: C(u,v) / (Σf(u) × Σf(v))")
        write()
        write("Association clustering measures how often two terms co-occur in the")
        write("same documents within the local document set. Higher values indicate")
        write("stronger statistical association between terms.")
        write()
        
        for q_idx, query in enumerate(QUERIES, 1):
            write("\n" + "=" * 80)
            write(f"EXAMPLE QUERY {q_idx}: \"{query}\"")
            write("=" * 80)
            
            # Normalize and get stems
            normalized_query = expander._normalize_query_for_expansion(query)
            query_stems = preprocess(query)
            query_stem_set = set(query_stems)
            query_len = len(query_stems)
            
            # Get local document set
            local_doc_ids = expander._get_local_doc_set(query, top_k=50)
            local_doc_ids_sorted = sorted(local_doc_ids)
            
            # Get local term frequencies
            local_tf = expander._get_local_term_frequencies(local_doc_ids)
            
            # ================================================================
            # (1) LOCAL DOCUMENT SET
            # ================================================================
            write("\n" + "-" * 70)
            write("(1) LOCAL DOCUMENT SET")
            write("-" * 70)
            write(f"\nNumber of documents: {len(local_doc_ids)}")
            write(f"Method: Top-50 BM25 results for normalized query '{normalized_query}'")
            write("\nDocument IDs:")
            
            # Print all 50 doc IDs in rows of 10
            for i in range(0, len(local_doc_ids_sorted), 10):
                chunk = local_doc_ids_sorted[i:i+10]
                write(f"    {', '.join(chunk)}")
            
            write("\nSample Documents with URLs:")
            for doc_id in local_doc_ids_sorted[:8]:
                doc = engine.doc_store.get(doc_id, {})
                url = doc.get('url', 'N/A')
                title = doc.get('title', 'N/A')[:50]
                write(f"    [{doc_id}] {url}")
            
            # ================================================================
            # (2) LOCAL VOCABULARY AND STEMS
            # ================================================================
            write("\n" + "-" * 70)
            write("(2) LOCAL VOCABULARY AND STEMS")
            write("-" * 70)
            write(f"\nQuery: \"{query}\"")
            write(f"Query stems (after preprocessing): {query_stems}")
            write(f"\nTotal unique stems in local vocabulary: {len(local_tf)}")
            
            # Get alphabetically sorted vocabulary (alpha only)
            alpha_vocab = sorted([t for t in local_tf.keys() if t.isalpha()])
            
            write(f"\nLocal vocabulary sample (first 80 alphabetical stems):")
            for i in range(0, min(80, len(alpha_vocab)), 10):
                chunk = alpha_vocab[i:i+10]
                write(f"    {', '.join(chunk)}")
            
            write(f"\nVocabulary statistics:")
            write(f"    - Total terms in local set: {len(local_tf)}")
            write(f"    - Alphabetic terms only: {len(alpha_vocab)}")
            write(f"    - Terms appearing in 2+ local docs: {sum(1 for t, p in local_tf.items() if len(p) >= 2)}")
            
            # ================================================================
            # (3) CORRELATION VALUES
            # ================================================================
            write("\n" + "-" * 70)
            write("(3) CORRELATION VALUES - Association C(u,v)")
            write("-" * 70)
            write("\nFormula: C(u,v) = Σ [f(u,j) × f(v,j)] for all docs j in local set")
            write("Normalized by: C(u,v) / (Σf(u) × Σf(v))")
            write()
            
            for q_term in query_stems:
                if q_term not in local_tf:
                    write(f"\nQuery stem '{q_term}': Not found in local vocabulary")
                    continue
                
                q_sum_f = sum(local_tf[q_term].values())
                write(f"\nQuery stem: '{q_term}'")
                write(f"    Frequency sum Σf({q_term}) = {q_sum_f}")
                write(f"    Appears in {len(local_tf[q_term])} local documents")
                write()
                write(f"    {'Candidate Term':<20} {'Raw C(u,v)':>12} {'Normalized':>12}")
                write(f"    {'-'*20} {'-'*12} {'-'*12}")
                
                # Calculate correlations
                correlations = []
                for v_term, v_postings in local_tf.items():
                    if not expander._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                        continue
                    
                    # Raw correlation
                    raw_c_uv = sum(local_tf[q_term].get(d, 0) * v_postings.get(d, 0) 
                                  for d in local_doc_ids)
                    
                    # Normalized
                    v_sum_f = sum(v_postings.values())
                    norm_c_uv = raw_c_uv / (q_sum_f * v_sum_f) if (q_sum_f * v_sum_f) > 0 else 0
                    
                    if raw_c_uv > 0:
                        correlations.append((v_term, raw_c_uv, norm_c_uv))
                
                correlations.sort(key=lambda x: -x[2])  # Sort by normalized
                
                for term, raw, norm in correlations[:12]:
                    write(f"    {term:<20} {raw:>12.2f} {norm:>12.6f}")
            
            # Cluster selection discussion
            write("\n" + "-" * 40)
            write("CLUSTER SELECTION DISCUSSION:")
            write("-" * 40)
            write("""
The association clustering method selects expansion terms based on normalized
co-occurrence frequency. Terms are selected if they:
1. Pass the candidate filter (alphabetic, 3-24 chars, not stopword, df >= 3)
2. Have high normalized correlation with query stems
3. Are weighted by IDF to prefer discriminative terms

For each query stem, the top m_neighbors (default=4) terms are selected.
Terms that appear as neighbors for multiple query stems receive boosted scores.
""")
            
            # ================================================================
            # (4) EXPANDED QUERY
            # ================================================================
            write("\n" + "-" * 70)
            write("(4) EXPANDED QUERY")
            write("-" * 70)
            
            expanded = expander.expand_association(query, top_k_docs=50, m_neighbors=4, 
                                                    normalized=True, max_new_terms=6)
            
            write(f"\nOriginal query: \"{query}\"")
            write(f"Normalized:     \"{normalized_query}\"")
            write(f"Expanded query: \"{expanded}\"")
            
            # Identify new terms
            orig_terms = set(query_stems)
            expanded_terms = expanded.split()
            new_terms = [t for t in expanded_terms if t not in orig_terms]
            
            write(f"\nNew terms added: {new_terms}")
            write(f"Number of expansion terms: {len(new_terms)}")
            
            # Search results comparison
            write("\n" + "-" * 40)
            write("SEARCH RESULTS COMPARISON:")
            write("-" * 40)
            
            orig_results = engine.search(normalized_query, method="bm25", top_k=5)
            write(f"\nOriginal Query Results ('{normalized_query}'):")
            for r in orig_results:
                write(f"    {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:65]}")
            
            exp_results = engine.search(expanded, method="bm25", top_k=5)
            write(f"\nExpanded Query Results ('{expanded[:50]}...'):")
            for r in exp_results:
                write(f"    {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:65]}")
        
        write("\n" + "=" * 80)
        write("END OF ASSOCIATION CLUSTERING ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
    
    print(f"\n✅ Report saved to: {output_path}")

if __name__ == "__main__":
    main()
