"""
Detailed Local Clustering Analysis for Query Expansion Report
Student: Uddesh Singh | NetID: UXS230004

This script generates detailed output for Section 6 of the project report,
showing local document sets, vocabularies, correlation values, and expanded
queries for 3 example queries across all 3 clustering methods.
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "indexer" / "src"))
sys.path.insert(0, str(REPO_ROOT / "expander"))

from search import SearchEngine
from core import QueryExpander, EXPANSION_STOPLIST
from preprocessor import preprocess
from query_sets import m_neighbors_for_query

# Three example queries from different geology domains
EXAMPLE_QUERIES = [
    "pyroclastic flow danger",      # Volcanology
    "metamorphic gneiss formation", # Petrology
    "groundwater aquifer permeability"  # Hydrogeology
]

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_subsection(title: str):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

def analyze_query_with_method(engine: SearchEngine, expander: QueryExpander, 
                               query: str, method: str):
    """
    Analyze a single query with a specific clustering method.
    Returns detailed information about local doc set, vocabulary, correlations.
    """
    # Normalize query first
    normalized_query = expander._normalize_query_for_expansion(query)
    query_stems = preprocess(query)
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    
    # Get local document set (top-50 HITS — same as QueryExpander._get_local_doc_set)
    local_doc_ids = expander._get_local_doc_set(query, top_k=50)
    
    # Get local term frequencies
    local_tf = expander._get_local_term_frequencies(local_doc_ids)
    
    m_neighbors = m_neighbors_for_query(query)
    # Get expanded query using the method (max_new_terms aligned with backend /api/expand)
    if method == "association":
        expanded = expander.expand_association(
            query,
            top_k_docs=50,
            m_neighbors=m_neighbors,
            normalized=True,
            max_new_terms=5,
        )
    elif method == "scalar":
        expanded = expander.expand_scalar(
            query,
            top_k_docs=50,
            m_neighbors=m_neighbors,
            max_new_terms=5,
        )
    elif method == "metric":
        expanded = expander.expand_metric(
            query,
            top_k_docs=50,
            m_neighbors=m_neighbors,
            max_new_terms=5,
        )
    
    return {
        "original_query": query,
        "normalized_query": normalized_query,
        "query_stems": query_stems,
        "local_doc_ids": sorted(local_doc_ids),
        "local_vocab_size": len(local_tf),
        "local_vocab_sample": sorted(local_tf.keys())[:100],
        "local_tf": local_tf,
        "expanded_query": expanded,
    }

def compute_association_correlations(expander: QueryExpander, query: str, 
                                      local_doc_ids: set, local_tf: dict,
                                      query_stems: list, normalized: bool = True):
    """
    Compute association correlation values C(u,v) for visualization.
    Formula: C(u,v) = SUM(d_j in Dl) [ f(u,j) * f(v,j) ]
    """
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    correlations_by_stem = {}
    
    for q_term in query_stems:
        if q_term not in local_tf:
            continue
            
        q_sum_f = sum(local_tf[q_term].values()) if normalized else 1.0
        term_correlations = []
        
        for v_term, v_postings in local_tf.items():
            if not expander._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                continue
            
            # Calculate correlation C(u, v)
            c_uv = sum(local_tf[q_term].get(d, 0) * v_postings.get(d, 0) 
                      for d in local_doc_ids)
            
            if normalized:
                v_sum_f = sum(v_postings.values())
                c_uv = c_uv / (q_sum_f * v_sum_f) if (q_sum_f * v_sum_f) > 0 else 0
            
            if c_uv > 0:
                term_correlations.append((v_term, c_uv))
        
        term_correlations.sort(key=lambda x: -x[1])
        correlations_by_stem[q_term] = term_correlations[:15]
    
    return correlations_by_stem

def compute_scalar_correlations(expander: QueryExpander, query: str,
                                 local_doc_ids: set, local_tf: dict,
                                 query_stems: list):
    """
    Compute scalar (cosine similarity) correlation values S(u,v).
    Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|)
    """
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    terms = list(local_tf.keys())
    
    # Build association matrix for query terms
    C = defaultdict(lambda: defaultdict(float))
    for u in query_stems:
        if u not in local_tf:
            continue
        for v in terms:
            C[u][v] = sum(local_tf[u].get(d, 0) * local_tf[v].get(d, 0) 
                         for d in local_doc_ids)
    
    correlations_by_stem = {}
    
    for q_term in query_stems:
        if q_term not in local_tf:
            continue
        
        norm_u = math.sqrt(sum(val**2 for val in C[q_term].values()))
        if norm_u == 0:
            continue
        
        term_correlations = []
        for v_term in terms:
            if not expander._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                continue
            
            # Compute cosine similarity
            dot_product = sum(C[q_term].get(x, 0) * C.get(v_term, {}).get(x, 0) 
                             for x in terms if C[q_term].get(x, 0) > 0)
            norm_v = math.sqrt(sum(local_tf[v_term].get(d, 0)**2 for d in local_doc_ids))
            
            if dot_product > 0 and norm_v > 0:
                s_uv = dot_product / (norm_u * norm_v)
                term_correlations.append((v_term, s_uv))
        
        term_correlations.sort(key=lambda x: -x[1])
        correlations_by_stem[q_term] = term_correlations[:15]
    
    return correlations_by_stem

def compute_metric_correlations(expander: QueryExpander, engine: SearchEngine,
                                 local_doc_ids: set, query_stems: list):
    """
    Compute metric (distance-based) correlation values.
    Formula: C(u,v) = SUM [ 1 / r(k_i, k_j) ] where r is word distance
    """
    query_stem_set = set(query_stems)
    query_len = len(query_stems)
    global_correlations = defaultdict(lambda: defaultdict(float))
    
    for d_id in local_doc_ids:
        doc = engine.doc_store.get(str(d_id), {})
        doc_tokens = preprocess(doc.get("text_preview", ""))
        
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
    
    correlations_by_stem = {}
    for q_term in query_stems:
        if q_term not in global_correlations:
            continue
        term_correlations = sorted(global_correlations[q_term].items(), 
                                   key=lambda x: -x[1])[:15]
        correlations_by_stem[q_term] = term_correlations
    
    return correlations_by_stem

def get_search_results(engine: SearchEngine, query: str, top_k: int = 10):
    """Match expansion_report /api/expand default post-expansion ranker."""
    return engine.search(query, method="hits", top_k=top_k)

def main():
    print("Loading Search Engine...")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)
    
    print(f"Loaded: {engine.N:,} documents, {len(engine.inverted_index):,} terms")
    
    output_path = REPO_ROOT / "expander" / "clustering_detailed_report.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        def write(text=""):
            print(text)
            f.write(text + "\n")
        
        write("=" * 80)
        write("DETAILED LOCAL CLUSTERING ANALYSIS FOR QUERY EXPANSION")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
        write()
        write(f"Index Statistics: {engine.N:,} documents, {len(engine.inverted_index):,} terms")
        write()
        
        # ================================================================
        # ASSOCIATION CLUSTERING
        # ================================================================
        write("\n" + "=" * 80)
        write("METHOD 1: ASSOCIATION CLUSTERING")
        write("Formula: C(u,v) = Σ(d_j ∈ D_l) [f(u,j) × f(v,j)]")
        write("Normalized: C(u,v) / (Σf(u) × Σf(v))")
        write("=" * 80)
        
        for i, query in enumerate(EXAMPLE_QUERIES, 1):
            write(f"\n{'─' * 70}")
            write(f"EXAMPLE QUERY {i}: \"{query}\"")
            write(f"{'─' * 70}")
            
            info = analyze_query_with_method(engine, expander, query, "association")
            
            # (1) Local Document Set
            write(f"\n(1) LOCAL DOCUMENT SET:")
            write(f"    Number of documents: {len(info['local_doc_ids'])}")
            write(f"    Document IDs (first 30):")
            for j in range(0, min(30, len(info['local_doc_ids'])), 10):
                chunk = info['local_doc_ids'][j:j+10]
                write(f"        {', '.join(chunk)}")
            
            # Show sample URLs
            write(f"\n    Sample Document URLs:")
            for doc_id in info['local_doc_ids'][:5]:
                doc = engine.doc_store.get(doc_id, {})
                url = doc.get('url', 'N/A')
                write(f"        [{doc_id}] {url[:80]}...")
            
            # (2) Local Vocabulary
            write(f"\n(2) LOCAL VOCABULARY AND STEMS:")
            write(f"    Total unique stems in local set: {info['local_vocab_size']}")
            write(f"    Query stems: {info['query_stems']}")
            write(f"\n    Local vocabulary sample (first 60 alphabetically):")
            vocab_sample = [t for t in info['local_vocab_sample'] if t.isalpha()][:60]
            for j in range(0, len(vocab_sample), 12):
                chunk = vocab_sample[j:j+12]
                write(f"        {', '.join(chunk)}")
            
            # (3) Correlation Values
            write(f"\n(3) CORRELATION VALUES (Normalized Association):")
            correlations = compute_association_correlations(
                expander, query, set(info['local_doc_ids']), 
                info['local_tf'], info['query_stems'], normalized=True
            )
            
            for q_stem, corr_list in correlations.items():
                write(f"\n    Query stem: '{q_stem}'")
                write(f"    {'Term':<20} {'C(u,v) normalized':>20}")
                write(f"    {'-'*20} {'-'*20}")
                for term, score in corr_list[:10]:
                    write(f"    {term:<20} {score:>20.6f}")
            
            # (4) Expanded Query
            write(f"\n(4) EXPANDED QUERY:")
            write(f"    Original:  {query}")
            write(f"    Expanded:  {info['expanded_query']}")
            
            # New terms added
            orig_terms = set(info['query_stems'])
            expanded_terms = info['expanded_query'].split()
            new_terms = [t for t in expanded_terms if t not in orig_terms]
            write(f"    New terms: {new_terms}")
            
            # Search Results
            write(f"\n(5) SEARCH RESULTS COMPARISON:")
            
            orig_results = get_search_results(engine, info['normalized_query'], top_k=5)
            write(f"\n    ORIGINAL QUERY RESULTS ('{info['normalized_query']}'):")
            for r in orig_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
            
            exp_results = get_search_results(engine, info['expanded_query'], top_k=5)
            write(f"\n    EXPANDED QUERY RESULTS ('{info['expanded_query'][:50]}...'):")
            for r in exp_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
        
        # ================================================================
        # SCALAR CLUSTERING
        # ================================================================
        write("\n\n" + "=" * 80)
        write("METHOD 2: SCALAR CLUSTERING")
        write("Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|)")
        write("Cosine similarity of association vectors")
        write("=" * 80)
        
        for i, query in enumerate(EXAMPLE_QUERIES, 1):
            write(f"\n{'─' * 70}")
            write(f"EXAMPLE QUERY {i}: \"{query}\"")
            write(f"{'─' * 70}")
            
            info = analyze_query_with_method(engine, expander, query, "scalar")
            
            # (1) Local Document Set
            write(f"\n(1) LOCAL DOCUMENT SET:")
            write(f"    Number of documents: {len(info['local_doc_ids'])}")
            write(f"    Document IDs (first 30):")
            for j in range(0, min(30, len(info['local_doc_ids'])), 10):
                chunk = info['local_doc_ids'][j:j+10]
                write(f"        {', '.join(chunk)}")
            
            # (2) Local Vocabulary
            write(f"\n(2) LOCAL VOCABULARY AND STEMS:")
            write(f"    Total unique stems in local set: {info['local_vocab_size']}")
            write(f"    Query stems: {info['query_stems']}")
            vocab_sample = [t for t in info['local_vocab_sample'] if t.isalpha()][:60]
            write(f"\n    Local vocabulary sample (first 60 alphabetically):")
            for j in range(0, len(vocab_sample), 12):
                chunk = vocab_sample[j:j+12]
                write(f"        {', '.join(chunk)}")
            
            # (3) Correlation Values
            write(f"\n(3) CORRELATION VALUES (Scalar/Cosine Similarity):")
            correlations = compute_scalar_correlations(
                expander, query, set(info['local_doc_ids']),
                info['local_tf'], info['query_stems']
            )
            
            for q_stem, corr_list in correlations.items():
                write(f"\n    Query stem: '{q_stem}'")
                write(f"    {'Term':<20} {'S(u,v) cosine':>20}")
                write(f"    {'-'*20} {'-'*20}")
                for term, score in corr_list[:10]:
                    write(f"    {term:<20} {score:>20.6f}")
            
            # (4) Expanded Query
            write(f"\n(4) EXPANDED QUERY:")
            write(f"    Original:  {query}")
            write(f"    Expanded:  {info['expanded_query']}")
            
            orig_terms = set(info['query_stems'])
            expanded_terms = info['expanded_query'].split()
            new_terms = [t for t in expanded_terms if t not in orig_terms]
            write(f"    New terms: {new_terms}")
            
            # Search Results
            write(f"\n(5) SEARCH RESULTS COMPARISON:")
            
            orig_results = get_search_results(engine, info['normalized_query'], top_k=5)
            write(f"\n    ORIGINAL QUERY RESULTS:")
            for r in orig_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
            
            exp_results = get_search_results(engine, info['expanded_query'], top_k=5)
            write(f"\n    EXPANDED QUERY RESULTS:")
            for r in exp_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
        
        # ================================================================
        # METRIC CLUSTERING
        # ================================================================
        write("\n\n" + "=" * 80)
        write("METHOD 3: METRIC CLUSTERING")
        write("Formula: C(u,v) = Σ[1 / r(k_i, k_j)]")
        write("Where r(k_i, k_j) is the word distance between term positions")
        write("=" * 80)
        
        for i, query in enumerate(EXAMPLE_QUERIES, 1):
            write(f"\n{'─' * 70}")
            write(f"EXAMPLE QUERY {i}: \"{query}\"")
            write(f"{'─' * 70}")
            
            info = analyze_query_with_method(engine, expander, query, "metric")
            
            # (1) Local Document Set
            write(f"\n(1) LOCAL DOCUMENT SET:")
            write(f"    Number of documents: {len(info['local_doc_ids'])}")
            write(f"    Document IDs (first 30):")
            for j in range(0, min(30, len(info['local_doc_ids'])), 10):
                chunk = info['local_doc_ids'][j:j+10]
                write(f"        {', '.join(chunk)}")
            
            # (2) Local Vocabulary
            write(f"\n(2) LOCAL VOCABULARY AND STEMS:")
            write(f"    Total unique stems in local set: {info['local_vocab_size']}")
            write(f"    Query stems: {info['query_stems']}")
            vocab_sample = [t for t in info['local_vocab_sample'] if t.isalpha()][:60]
            write(f"\n    Local vocabulary sample (first 60 alphabetically):")
            for j in range(0, len(vocab_sample), 12):
                chunk = vocab_sample[j:j+12]
                write(f"        {', '.join(chunk)}")
            
            # (3) Correlation Values (Metric)
            write(f"\n(3) CORRELATION VALUES (Metric/Distance-based):")
            correlations = compute_metric_correlations(
                expander, engine, set(info['local_doc_ids']), info['query_stems']
            )
            
            for q_stem, corr_list in correlations.items():
                write(f"\n    Query stem: '{q_stem}'")
                write(f"    {'Term':<20} {'Σ(1/distance)':>20}")
                write(f"    {'-'*20} {'-'*20}")
                for term, score in corr_list[:10]:
                    write(f"    {term:<20} {score:>20.6f}")
            
            # (4) Expanded Query
            write(f"\n(4) EXPANDED QUERY:")
            write(f"    Original:  {query}")
            write(f"    Expanded:  {info['expanded_query']}")
            
            orig_terms = set(info['query_stems'])
            expanded_terms = info['expanded_query'].split()
            new_terms = [t for t in expanded_terms if t not in orig_terms]
            write(f"    New terms: {new_terms}")
            
            # Search Results
            write(f"\n(5) SEARCH RESULTS COMPARISON:")
            
            orig_results = get_search_results(engine, info['normalized_query'], top_k=5)
            write(f"\n    ORIGINAL QUERY RESULTS:")
            for r in orig_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
            
            exp_results = get_search_results(engine, info['expanded_query'], top_k=5)
            write(f"\n    EXPANDED QUERY RESULTS:")
            for r in exp_results:
                write(f"        {r['rank']:>2}. [{r['doc_id']}] {r.get('url', 'N/A')[:70]}")
        
        # ================================================================
        # CLUSTER SELECTION DISCUSSION
        # ================================================================
        write("\n\n" + "=" * 80)
        write("CLUSTER SELECTION METHODOLOGY")
        write("=" * 80)
        
        write("""
The cluster selection process in our implementation (expander/core.py) follows
these steps for each method:

1. CANDIDATE FILTERING (_is_candidate_term method):
   - Terms must be purely alphabetical (no numbers/symbols)
   - Length must be 3-24 characters
   - Must not be in the stoplist (NLTK + web boilerplate)
   - Document frequency must be >= 3 (avoids noise from rare terms)
   - Document frequency ratio must be < 5% of corpus (avoids common terms)
   
2. CORRELATION SCORING:
   - Each method computes correlation values differently:
     * Association: Raw co-occurrence frequency (normalized)
     * Scalar: Cosine similarity of association vectors  
     * Metric: Inverse word distance
   
3. IDF WEIGHTING (_score_expansion_term method):
   - Raw correlation scores are multiplied by IDF
   - This prefers discriminative terms over common ones
   - Formula: final_score = raw_score × log10(N/df)

4. NEIGHBOR SELECTION:
   - For each query stem, top m_neighbors (default=4) are selected
   - Terms appearing as neighbors for multiple query stems get boosted
   
5. FINAL EXPANSION:
   - Top max_new_terms (6 for association/scalar, 4 for metric) are added
   - Original query terms are preserved at the start
   - Duplicate terms are removed
""")
        
        write("\n" + "=" * 80)
        write("END OF DETAILED CLUSTERING ANALYSIS")
        write("Student: Uddesh Singh | NetID: UXS230004")
        write("=" * 80)
    
    print(f"\n✅ Report saved to: {output_path}")

if __name__ == "__main__":
    main()
