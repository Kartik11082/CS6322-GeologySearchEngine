"""
Metric Clustering Analysis - 3 Example Queries
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

# 3 queries for Metric Clustering analysis
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
    
    output_path = PROJECT_ROOT / "expander" / "report_METRIC_clustering.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        def write(text=""):
            print(text)
            f.write(text + "\n")
        
        write("=" * 80)
        write("METRIC CLUSTERING - DETAILED ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
        write()
        write("Formula: C(u,v) = Σ [1 / r(k_i, k_j)]")
        write("Where r(k_i, k_j) is the word distance between positions of u and v")
        write()
        write("Metric clustering measures proximity between terms within documents.")
        write("Terms that appear close to query terms (small word distance) receive")
        write("higher correlation scores. This captures contextual relationships.")
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
            # (3) CORRELATION VALUES - Metric (Distance-based)
            # ================================================================
            write("\n" + "-" * 70)
            write("(3) CORRELATION VALUES - Metric C(u,v)")
            write("-" * 70)
            write("\nFormula: C(u,v) = Σ [1 / |pos(u) - pos(v)|]")
            write("Summed across all documents and all position pairs")
            write()
            
            # Compute metric correlations
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
                        if not expander._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                            continue
                        
                        for q_pos in positions[q_term]:
                            for v_pos in v_pos_list:
                                distance = abs(q_pos - v_pos)
                                if distance > 0:
                                    global_correlations[q_term][v_term] += (1.0 / distance)
            
            write(f"Documents with position data: {docs_with_positions}")
            
            for q_term in query_stems:
                if q_term not in global_correlations:
                    write(f"\nQuery stem '{q_term}': No position data found")
                    continue
                
                write(f"\nQuery stem: '{q_term}'")
                write(f"    Co-located with {len(global_correlations[q_term])} candidate terms")
                write()
                write(f"    {'Candidate Term':<20} {'Σ(1/distance)':>15} {'Interpretation':<30}")
                write(f"    {'-'*20} {'-'*15} {'-'*30}")
                
                correlations = sorted(global_correlations[q_term].items(), 
                                      key=lambda x: -x[1])[:15]
                
                for term, score in correlations:
                    # Interpretation based on score
                    if score > 5:
                        interp = "Very close proximity"
                    elif score > 2:
                        interp = "Close proximity"
                    elif score > 1:
                        interp = "Moderate proximity"
                    else:
                        interp = "Distant"
                    write(f"    {term:<20} {score:>15.6f} {interp:<30}")
            
            # Cluster selection discussion
            write("\n" + "-" * 40)
            write("CLUSTER SELECTION DISCUSSION:")
            write("-" * 40)
            write("""
Metric clustering selects terms based on physical proximity within documents.
The inverse distance formula (1/r) gives higher weight to terms appearing
immediately adjacent to query terms.

Key characteristics:
1. Captures phrase-like relationships (terms often used together)
2. Sensitive to writing style and word order
3. Works well for finding action words and modifiers

Selection process:
1. For each document, extract term positions from text_preview
2. For each query stem position, compute 1/distance to all other terms
3. Sum inverse distances across all documents
4. Select top m_neighbors for each query stem
5. Weight by IDF and combine scores
""")
            
            # ================================================================
            # (4) EXPANDED QUERY
            # ================================================================
            write("\n" + "-" * 70)
            write("(4) EXPANDED QUERY")
            write("-" * 70)
            
            expanded = expander.expand_metric(query, top_k_docs=50, m_neighbors=4, 
                                               max_new_terms=4)
            
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
        write("END OF METRIC CLUSTERING ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
    
    print(f"\n✅ Report saved to: {output_path}")

if __name__ == "__main__":
    main()
