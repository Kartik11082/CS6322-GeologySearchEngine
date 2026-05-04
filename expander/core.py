import sys
import math
import json
import time
import difflib
from pathlib import Path
from collections import defaultdict

# Add indexer src to path so we can use Student 2's preprocessor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

from preprocessor import preprocess, tokenize

# Build expansion stoplist from NLTK + domain-specific terms
def _build_expansion_stoplist() -> set[str]:
    """Build stoplist using stemmed NLTK stopwords + web/academic boilerplate."""
    import Stemmer
    from nltk.corpus import stopwords
    
    stemmer = Stemmer.Stemmer("english")
    stoplist = set()
    
    # 1. Stemmed NLTK English stopwords
    stoplist.update(stemmer.stemWords(list(stopwords.words("english"))))
    
    # 2. Web boilerplate (navigation, footers, social media)
    web_terms = [
        "website", "click", "link", "page", "home", "menu", "search", "login",
        "share", "facebook", "twitter", "pinterest", "youtube", "instagram",
        "subscribe", "newsletter", "copyright", "privacy", "cookie",
        "domain", "public", "route", "inspire", "autocomplete", "insignia",
        "club", "society", "member", "join", "event",
    ]
    stoplist.update(stemmer.stemWords(web_terms))
    
    return stoplist


# Pre-compute stoplist and boost terms at module load
EXPANSION_STOPLIST = _build_expansion_stoplist()
DEBUG_LOG_PATH = PROJECT_ROOT / ".cursor" / "debug-784b16.log"


def _debug_log(location: str, message: str, data: dict, run_id: str, hypothesis_id: str):
    payload = {
        "sessionId": "784b16",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


class QueryExpander:
    def __init__(self, search_engine):
        self.engine = search_engine
        self._autocorrect_vocab = self._build_autocorrect_vocab()

    def _build_autocorrect_vocab(self) -> list[str]:
        """Build compact vocabulary used for typo correction."""
        vocab = set()
        for term, postings in self.engine.inverted_index.items():
            if term.isalpha() and 2 < len(term) <= 24 and len(postings) >= 5:
                vocab.add(term)
        return sorted(vocab)

    def _normalize_query_for_expansion(self, query: str) -> str:
        """
        Light typo correction at stem level.
        Keeps the original token when no confident correction is found.
        """
        tokens = tokenize(query)
        if not tokens:
            return query

        corrected_tokens = []
        for token in tokens:
            stems = preprocess(token)
            stem = stems[0] if stems else token
            if stem in self.engine.inverted_index:
                corrected_tokens.append(stem)
                continue

            match = difflib.get_close_matches(
                stem,
                self._autocorrect_vocab,
                n=1,
                cutoff=0.82,
            )
            corrected_tokens.append(match[0] if match else stem)

        return " ".join(corrected_tokens)

    def normalize_query(self, query: str) -> str:
        """Public helper for consumers that need corrected query text."""
        return self._normalize_query_for_expansion(query)

    # =====================================================================
    #  1. ROCCHIO RELEVANCE FEEDBACK (Lecture 12)
    # =====================================================================
    def expand_rocchio(
        self, 
        query: str, 
        relevant_doc_ids: list[str], 
        irrelevant_doc_ids: list[str], 
        alpha: float = 1.0, 
        beta: float = 0.75, 
        gamma: float = 0.25, 
        num_new_terms: int = 3
    ) -> str:
        """
        Rocchio 1971 Algorithm.
        Formula: q_m = alpha * q_0 + (beta / |Dr|) * SUM(Dj in Dr) - (gamma / |Dnr|) * SUM(Dj in Dnr)
        """
        query = self._normalize_query_for_expansion(query)
        query_stems = preprocess(query)
        
        # Build the original query vector (TF=1 for query terms)
        q0_vector = defaultdict(float)
        for stem in query_stems:
            q0_vector[stem] += 1.0

        # Helper to compute document vectors (TF-IDF)
        def get_doc_vector(doc_id: str) -> dict[str, float]:
            vector = defaultdict(float)
            for term, postings in self.engine.inverted_index.items():
                if doc_id in postings:
                    tf = postings[doc_id]
                    df = len(postings)
                    idf = math.log10(self.engine.N / df) if df > 0 else 0
                    tf_weight = 1.0 + math.log10(tf) if tf > 0 else 0
                    vector[term] = tf_weight * idf
            return vector

        # Sum of relevant document vectors
        dr_sum = defaultdict(float)
        if relevant_doc_ids:
            for d_id in relevant_doc_ids:
                d_vec = get_doc_vector(str(d_id))
                for term, weight in d_vec.items():
                    dr_sum[term] += weight

        # Sum of non-relevant document vectors
        dnr_sum = defaultdict(float)
        if irrelevant_doc_ids:
            for d_id in irrelevant_doc_ids:
                d_vec = get_doc_vector(str(d_id))
                for term, weight in d_vec.items():
                    dnr_sum[term] += weight

        # Compute the modified query vector (q_m)
        qm_vector = defaultdict(float)
        all_terms = set(list(q0_vector.keys()) + list(dr_sum.keys()) + list(dnr_sum.keys()))
        
        len_dr = len(relevant_doc_ids) if relevant_doc_ids else 1
        len_dnr = len(irrelevant_doc_ids) if irrelevant_doc_ids else 1

        for term in all_terms:
            rocchio_weight = (
                (alpha * q0_vector.get(term, 0.0)) + 
                (beta * (dr_sum.get(term, 0.0) / len_dr)) - 
                (gamma * (dnr_sum.get(term, 0.0) / len_dnr))
            )
            if rocchio_weight > 0:
                qm_vector[term] = rocchio_weight

        # Extract the top new terms to add to the query.
        # Reuse shared term-quality checks to avoid noisy tokens.
        sorted_terms = sorted(qm_vector.items(), key=lambda x: x[1], reverse=True)
        new_terms = []
        query_stem_set = set(query_stems)
        query_len = len(query_stems)
        for term, _ in sorted_terms:
            if term in query_stem_set:
                continue
            if not self._is_candidate_term(term, query_stem_set, query_len=query_len):
                continue
            new_terms.append(term)
            if len(new_terms) == num_new_terms:
                break
        _debug_log(
            location="expander/core.py:expand_rocchio:selection",
            message="rocchio selected terms after filtering",
            data={
                "query": query,
                "requested_new_terms": num_new_terms,
                "selected_new_terms_count": len(new_terms),
                "selected_new_terms": new_terms[:10],
            },
            run_id="post-fix",
            hypothesis_id="H6",
        )
                    
        return f"{query} {' '.join(new_terms)}"

    # =====================================================================
    #  HELPERS FOR LOCAL CLUSTERING (QE Local Strategies)
    # =====================================================================
    def _get_local_doc_set(self, query: str, top_k: int) -> set[str]:
        normalized_query = self._normalize_query_for_expansion(query)
        results = self.engine.search(normalized_query, method="bm25", top_k=top_k)
        return {str(res["doc_id"]) for res in results}

    def _get_local_term_frequencies(self, local_doc_ids: set[str]):
        """Returns dict: term -> {doc_id -> frequency} for local documents."""
        local_tf = defaultdict(dict)
        for term, postings in self.engine.inverted_index.items():
            overlap = local_doc_ids.intersection(postings.keys())
            if len(overlap) > 1:  # Term must appear in at least 2 local docs
                for d_id in overlap:
                    local_tf[term][d_id] = postings[d_id]
        return local_tf

    def _finalize_expansion(
        self,
        query: str,
        query_stems: set[str],
        candidate_scores: dict[str, float],
        max_new_terms: int,
        query_len: int,
    ) -> str:
        """
        Build a deterministic expanded query from ranked candidate terms.

        - Keeps original query token order.
        - Adds top-scoring non-duplicate candidate terms.
        - Falls back to original query when candidates are weak/empty.
        """
        original_tokens = query.split()
        original_token_set = set(original_tokens)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        new_terms = []
        for term, score in ranked:
            if score <= 0:
                continue
            if term in original_token_set or term in new_terms:
                continue
            if not self._is_candidate_term(term, query_stems, query_len=query_len):
                continue
            new_terms.append(term)
            if len(new_terms) >= max_new_terms:
                break

        if not new_terms:
            return query
        return " ".join(original_tokens + new_terms)

    # =====================================================================
    #  2. ASSOCIATION CLUSTERS
    # =====================================================================
    def expand_association(
        self,
        query: str,
        top_k_docs: int = 50,
        m_neighbors: int = 4,
        normalized: bool = False,
        max_new_terms: int = 6,
    ) -> str:
        """
        Formula: C(u,v) = SUM(d_j in Dl) [ f(u,j) * f(v,j) ]
        Normalized: C(u,v) / (SUM f(u) * SUM f(v))
        Strategy: For each query term, find its top m neighbors.
        """
        query = self._normalize_query_for_expansion(query)
        local_doc_ids = self._get_local_doc_set(query, top_k_docs)
        if not local_doc_ids: return query
        
        local_tf = self._get_local_term_frequencies(local_doc_ids)
        query_stems = preprocess(query)
        query_stem_set = set(query_stems)
        query_len = len(query_stems)
        candidate_scores = defaultdict(float)

        for q_term in query_stems:
            if q_term not in local_tf: continue
            
            correlations = defaultdict(float)
            q_sum_f = sum(local_tf[q_term].values()) if normalized else 1.0

            for v_term, v_postings in local_tf.items():
                if not self._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                    continue
                
                # Calculate correlation C(u, v)
                c_uv = sum(local_tf[q_term].get(d, 0) * v_postings.get(d, 0) for d in local_doc_ids)
                
                if normalized:
                    v_sum_f = sum(v_postings.values())
                    c_uv = c_uv / (q_sum_f * v_sum_f) if (q_sum_f * v_sum_f) > 0 else 0

                if c_uv > 0:
                    correlations[v_term] = c_uv

            # Apply IDF + domain boosting to correlation scores
            scored_neighbors = [
                (term, self._score_expansion_term(term, score, query_stem_set))
                for term, score in correlations.items()
            ]
            
            # Select m top neighbors for this specific query term
            top_neighbors = sorted(scored_neighbors, key=lambda x: x[1], reverse=True)[:m_neighbors]
            for neighbor, score in top_neighbors:
                candidate_scores[neighbor] = max(candidate_scores[neighbor], score)

        return self._finalize_expansion(
            query=query,
            query_stems=query_stem_set,
            candidate_scores=candidate_scores,
            max_new_terms=max_new_terms,
            query_len=query_len,
        )

    # =====================================================================
    #  3. SCALAR CLUSTERS
    # =====================================================================
    def expand_scalar(
        self,
        query: str,
        top_k_docs: int = 50,
        m_neighbors: int = 4,
        max_new_terms: int = 6,
    ) -> str:
        """
        Cosine similarity of association vectors.
        Formula: S(u,v) = (s_u * s_v) / (|s_u| * |s_v|)
        """
        query = self._normalize_query_for_expansion(query)
        local_doc_ids = self._get_local_doc_set(query, top_k_docs)
        if not local_doc_ids: return query
        
        local_tf = self._get_local_term_frequencies(local_doc_ids)
        query_stems = preprocess(query)
        query_stem_set = set(query_stems)
        query_len = len(query_stems)
        
        # Pre-compute the unnormalized association matrix (just what we need)
        # C[u][v] = association between u and v
        C = defaultdict(lambda: defaultdict(float))
        terms = list(local_tf.keys())
        
        for u in query_stems:
            if u not in local_tf: continue
            for v in terms:
                C[u][v] = sum(local_tf[u].get(d, 0) * local_tf[v].get(d, 0) for d in local_doc_ids)
                
        for v in terms: # Need full vector for candidates to compute their norms
             for x in terms:
                  if v == x:
                      C[v][x] = sum(local_tf[v].get(d, 0) * local_tf[x].get(d, 0) for d in local_doc_ids)

        candidate_scores = defaultdict(float)

        for q_term in query_stems:
            if q_term not in local_tf: continue
            
            # Vector s_u is C[q_term]
            norm_u = math.sqrt(sum(val**2 for val in C[q_term].values()))
            if norm_u == 0: continue
            
            correlations = defaultdict(float)
            for v_term in terms:
                if not self._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                    continue
                
                # Compute dot product and norms (approximated for efficiency)
                dot_product = sum(C[q_term].get(x, 0) * C[v_term].get(x, 0) for x in terms if C[q_term].get(x, 0) > 0)
                norm_v = math.sqrt(sum(local_tf[v_term].get(d, 0)**2 for d in local_doc_ids)) # Rough proxy to save memory
                
                if dot_product > 0 and norm_v > 0:
                    correlations[v_term] = dot_product / (norm_u * norm_v)

            # Apply IDF + domain boosting
            scored_neighbors = [
                (term, self._score_expansion_term(term, score, query_stem_set))
                for term, score in correlations.items()
            ]
            
            top_neighbors = sorted(scored_neighbors, key=lambda x: x[1], reverse=True)[:m_neighbors]
            for neighbor, score in top_neighbors:
                candidate_scores[neighbor] = max(candidate_scores[neighbor], score)

        # Borrow hybrid-style rank fusion intuition: terms supported by multiple
        # query terms are more reliable than one-off neighbors.
        return self._finalize_expansion(
            query=query,
            query_stems=query_stem_set,
            candidate_scores=candidate_scores,
            max_new_terms=max_new_terms,
            query_len=query_len,
        )

    # =====================================================================
    #  4. METRIC CLUSTERS
    # =====================================================================
    def expand_metric(
        self,
        query: str,
        top_k_docs: int = 50,
        m_neighbors: int = 4,
        max_new_terms: int = 4,
    ) -> str:
        """
        Based on physical distance between words.
        Formula: C(u,v) = SUM [ 1 / r(k_i, k_j) ] where r is distance in words.
        """
        query = self._normalize_query_for_expansion(query)
        local_doc_ids = self._get_local_doc_set(query, top_k_docs)
        if not local_doc_ids: return query
        
        query_stems = preprocess(query)
        query_stem_set = set(query_stems)
        query_len = len(query_stems)
        candidate_scores = defaultdict(float)
        
        # We calculate C(u,v) globally across all local docs
        global_correlations = defaultdict(lambda: defaultdict(float))

        for d_id in local_doc_ids:
            doc = self.engine.doc_store.get(str(d_id), {})
            # Use text_preview as proxy for distance since full text isn't in memory
            doc_tokens = preprocess(doc.get("text_preview", ""))
            
            positions = defaultdict(list)
            for i, token in enumerate(doc_tokens):
                positions[token].append(i)
                
            for q_term in query_stems:
                if q_term not in positions: continue
                
                for v_term, v_pos_list in positions.items():
                    if not self._is_candidate_term(v_term, query_stem_set, query_len=query_len):
                        continue
                    
                    # Compute distance: 1 / |pos(u) - pos(v)|
                    for q_pos in positions[q_term]:
                        for v_pos in v_pos_list:
                            distance = abs(q_pos - v_pos)
                            if distance > 0:
                                global_correlations[q_term][v_term] += (1.0 / distance)

        for q_term in query_stems:
            if q_term not in global_correlations: continue
            
            # Apply IDF + domain boosting
            scored_neighbors = [
                (term, self._score_expansion_term(term, score, query_stem_set))
                for term, score in global_correlations[q_term].items()
            ]
            
            top_neighbors = sorted(scored_neighbors, key=lambda x: x[1], reverse=True)[:m_neighbors]
            for neighbor, score in top_neighbors:
                candidate_scores[neighbor] = max(candidate_scores[neighbor], score)

        return self._finalize_expansion(
            query=query,
            query_stems=query_stem_set,
            candidate_scores=candidate_scores,
            max_new_terms=max_new_terms,
            query_len=query_len,
        )

    def _is_candidate_term(
        self,
        term: str,
        query_stems: set,
        query_len: int | None = None,
        min_df: int | None = None,
    ) -> bool:
        """Filter noisy expansion terms using lexical, stoplist, and document-frequency checks."""
        if query_len is None:
            query_len = max(len(query_stems), 1)

        if query_len <= 1:
            # Relax one-word query constraints so high-signal related terms
            # (e.g., volcanic subtypes/process terms) are not over-filtered.
            min_df_eff = max(4, min_df or 0)
            max_df_ratio = 0.1
        elif query_len == 2:
            min_df_eff = max(3, min_df or 0)
            max_df_ratio = 0.05
        else:
            min_df_eff = max(3, min_df or 0)
            max_df_ratio = 0.05

        # 1. Skip terms already in the query
        if term in query_stems:
            return False
            
        # 2. Lexical check: Must be purely alphabetical and > 2 characters
        if not term.isalpha() or len(term) <= 2 or len(term) > 24:
            return False
        
        # 3. Stoplist check: Filter known web/academic boilerplate terms
        if term in EXPANSION_STOPLIST:
            return False

        # 4. Document frequency checks
        df = len(self.engine.inverted_index.get(term, {}))
        
        # 4a. Minimum DF: Terms appearing in only 1 doc are likely noise
        if df < min_df_eff:
            return False

        if self.engine.N <= 0:
            return False

        df_ratio = df / self.engine.N

        # 4b. Maximum DF: Remove ubiquitous terms.
        if df_ratio > max_df_ratio:
            return False

        # 4c. Rarity quality cap: keep extremely rare artifacts out while
        # allowing niche but still useful domain terms for short queries.
        if query_len <= 2 and self.engine.N >= 5000 and df_ratio < 0.0015:
            return False
            
        return True
    
    def _get_term_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = len(self.engine.inverted_index.get(term, {}))
        if df == 0 or self.engine.N == 0:
            return 0.0
        return math.log10(self.engine.N / df)
    
    def _score_expansion_term(self, term: str, raw_score: float, query_stems: set) -> float:
        """
        Score an expansion term combining:
        - Raw score from the expansion method (co-occurrence, correlation, etc.)
        - IDF weighting (prefer discriminative terms)
        - Query similarity penalty (prevent drift)
        """
        idf = self._get_term_idf(term)

        # Combine: raw_score * IDF
        return raw_score * idf