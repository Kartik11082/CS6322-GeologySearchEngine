"""
Scalar Clustering Analysis - 3 Example Queries
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

# 3 queries for Scalar Clustering analysis
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
    
    output_path = PROJECT_ROOT / "expander" / "report_SCALAR_clustering.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        def write(text=""):
            print(text)
            f.write(text + "\n")
        
        write("=" * 80)
        write("SCALAR CLUSTERING - DETAILED ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
        write()
        write("Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|)")
        write()
        write("Scalar clustering computes the cosine similarity between association")
        write("vectors. This captures second-order relationships - terms that have")
        write("similar co-occurrence patterns are considered related even if they")
        write("don't directly co-occur with the query term.")
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
            terms = list(local_tf.keys())
            
            # ================================================================
            # (1) LOCAL DOCUMENT SET
            # ================================================================
            write("\n" + "-" * 70)
            write("(1) LOCAL DOCUMENT SET")
            write("-" * 70)
            write(f"\nNumber of documents: {len(local_doc_ids)}")
            write(f"Method: Top-50 BM25 results for normalized query '{normalized_query}'")
            write("\nDocument IDs:")
            
            for i in range(0, len(local_doc_ids_sorted), 10):
                chunk = local_doc_ids_sorted[i:i+10]
                write(f"    {', '.join(chunk)}")
            
            write("\nSample Documents with URLs:")
            for doc_id in local_doc_ids_sorted[:8]:
                doc = engine.doc_store.get(doc_id, {})
                url = doc.get('url', 'N/A')
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
            # (3) CORRELATION VALUES - Scalar (Cosine Similarity)
            # ================================================================
            write("\n" + "-" * 70)
            write("(3) CORRELATION VALUES - Scalar S(u,v)")
            write("-" * 70)
            write("\nFormula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|)")
            write("Where s_u is the association vector for term u")
            write()
            
            # Build association matrix for query terms
            C = defaultdict(lambda: defaultdict(float))
            for u in query_stems:
                if u not in local_tf:
                    continue
                for v in terms:
                    C[u][v] = sum(local_tf[u].get(d, 0) * local_tf[v].get(d, 0) 
                                 for d in local_doc_ids)
            
            for q_term in query_stems:
                if q_term not in local_tf:
                    write(f"\nQuery stem '{q_term}': Not found in local vocabulary")
                    continue
                
                norm_u = math.sqrt(sum(val**2 for val in C[q_term].values()))
                
                write(f"\nQuery stem: '{q_term}'")
                write(f"    Association vector norm |s_{q_term}| = {norm_u:.4f}")
                write(f"    Appears in {len(local_tf[q_term])} local documents")
                write()
                write(f"    {'Candidate Term':<20} {'Dot Product':>14} {'|s_v|':>10} {'S(u,v)':>12}")
                write(f"    {'-'*20} {'-'*14} {'-'*10} {'-'*12}")
                
                if norm_u == 0:
                    write("    [No valid correlations - zero norm]")
                    continue
                
                # Calculate scalar correlations
                correlations = []
                for v_term in terms:
                    if not expander._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                        continue
                    
                    # Dot product of association vectors
                    dot_product = sum(C[q_term].get(x, 0) * C.get(v_term, {}).get(x, 0) 
                                     for x in terms if C[q_term].get(x, 0) > 0)
                    
                    # Norm of v's association vector (approximated)
                    norm_v = math.sqrt(sum(local_tf[v_term].get(d, 0)**2 for d in local_doc_ids))
                    
                    if dot_product > 0 and norm_v > 0:
                        s_uv = dot_product / (norm_u * norm_v)
                        correlations.append((v_term, dot_product, norm_v, s_uv))
                
                correlations.sort(key=lambda x: -x[3])  # Sort by cosine similarity
                
                for term, dot, norm_v, s_uv in correlations[:12]:
                    write(f"    {term:<20} {dot:>14.2f} {norm_v:>10.2f} {s_uv:>12.6f}")
            
            # Cluster selection discussion
            write("\n" + "-" * 40)
            write("CLUSTER SELECTION DISCUSSION:")
            write("-" * 40)
            write("""
Scalar clustering selects terms based on cosine similarity of their association
vectors. This method captures semantic similarity - terms with similar 
co-occurrence patterns are selected even if they don't directly co-occur.

Key advantages:
1. Captures synonyms and related concepts (e.g., "schist" similar to "gneiss")
2. Less sensitive to document frequency variations
3. Finds terms used in similar contexts

Selection process:
1. Build association vectors for all terms in local vocabulary
2. Compute cosine similarity between query stem vectors and candidate vectors
3. Select top m_neighbors for each query stem
4. Weight by IDF and combine scores from multiple query stems
""")
            
            # ================================================================
            # (4) EXPANDED QUERY
            # ================================================================
            write("\n" + "-" * 70)
            write("(4) EXPANDED QUERY")
            write("-" * 70)
            
            expanded = expander.expand_scalar(query, top_k_docs=50, m_neighbors=4, 
                                               max_new_terms=6)
            
            write(f"\nOriginal query: \"{query}\"")
            write(f"Normalized:     \"{normalized_query}\"")
            write(f"Expanded query: \"{expanded}\"")
            
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
        write("END OF SCALAR CLUSTERING ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
    
    print(f"\n✅ Report saved to: {output_path}")

if __name__ == "__main__":
    main()
