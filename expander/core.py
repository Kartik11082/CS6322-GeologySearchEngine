"""Query expansion module.

Queries use Porter stems internally (same as the inverted index). Expansion *outputs*
can surface morphological lemmas via spaCy for readability; BM25 still re-stems at search time.

Optional: install spaCy model for lemmas::

    python -m spacy download en_core_web_sm
"""

import sys
import math
import warnings
import difflib
from pathlib import Path
from collections import Counter, defaultdict

# Add indexer src to path so we can use Student 2's preprocessor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

from preprocessor import preprocess, tokenize

from scalar_association_math import association_cosine_doc_frequency

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
# debug logging removed


class LemmaResolver:
    """
    Map Porter index stems to spaCy morphological lemmas for display-only expanded queries.

    If spaCy / en_core_web_sm is unavailable, ``stem_to_lemma`` returns the stem unchanged.
    """

    _missing_model_warned = False

    def __init__(self, search_engine) -> None:
        self._stem_to_surface: dict[str, str] = self._build_stem_to_surface(search_engine)
        self._cache: dict[str, str] = {}
        self._nlp = None
        self._load_failed = False

    def _build_stem_to_surface(self, engine) -> dict[str, str]:
        stem_counts: dict[str, Counter] = defaultdict(Counter)
        for doc in engine.doc_store.values():
            preview = doc.get("text_preview", "") or ""
            for tok in tokenize(preview):
                stems = preprocess(tok)
                if stems:
                    stem_counts[stems[0]][tok] += 1
        return {s: c.most_common(1)[0][0] for s, c in stem_counts.items()}

    def _ensure_nlp(self) -> None:
        if self._nlp is not None or self._load_failed:
            return
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            self._load_failed = True
            if not LemmaResolver._missing_model_warned:
                warnings.warn(
                    "spaCy model 'en_core_web_sm' not available; expanded queries use Porter stems. "
                    "Install: pip install spacy && python -m spacy download en_core_web_sm",
                    UserWarning,
                    stacklevel=3,
                )
                LemmaResolver._missing_model_warned = True

    def stem_to_lemma(self, stem: str) -> str:
        """Return a morphological lemma for a Porter stem; fallback to stem if unavailable."""
        if stem in self._cache:
            return self._cache[stem]

        surface = self._stem_to_surface.get(stem, stem)
        self._ensure_nlp()
        if self._nlp is None:
            self._cache[stem] = stem
            return stem

        doc = self._nlp(surface)
        if not doc:
            self._cache[stem] = stem
            return stem

        lemma = doc[0].lemma_.lower().strip()
        if not lemma or not lemma.isalpha():
            self._cache[stem] = stem
            return stem

        self._cache[stem] = lemma
        return lemma

    def stem_to_surface(self, stem: str) -> str | None:
        """Return most common corpus surface form for a stem, if available."""
        surface = self._stem_to_surface.get(stem)
        if not surface:
            return None
        surface = surface.lower().strip()
        if not surface.isalpha() or len(surface) <= 2:
            return None
        return surface


class QueryExpander:
    def __init__(self, search_engine):
        self.engine = search_engine
        self._autocorrect_vocab = self._build_autocorrect_vocab()
        self._lemma = LemmaResolver(search_engine)

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

    def _natural_word_for_stem(self, stem: str) -> str | None:
        """
        Render a stem as a natural display token.

        Preference order:
        1) spaCy lemma when available and clean
        2) most-common corpus surface form
        3) no token (avoid exposing stem artifacts to API consumers)
        """
        lemma = self._lemma.stem_to_lemma(stem).lower().strip()
        if lemma != stem and lemma.isalpha() and len(lemma) > 2:
            return lemma

        surface = self._lemma.stem_to_surface(stem)
        if surface:
            return surface

        return None

    def _safe_stem_as_display(self, stem: str) -> str | None:
        """
        Last-resort display token: Porter stem that still looks like a real word.

        Used only when lemma/surface are unavailable and policy allows (see m_neighbors).
        """
        if not stem.isalpha() or len(stem) <= 2 or len(stem) > 24:
            return None
        if stem in EXPANSION_STOPLIST:
            return None
        df = len(self.engine.inverted_index.get(stem, {}))
        if df < 4:
            return None
        return stem

    @staticmethod
    def _expansion_stem_fallback_slots(m_neighbors: int | None) -> int:
        """
        How many expansion terms may fall back to a clean stem for display.

        Tight neighbor sets (few alternatives) may use one stem fallback;
        large neighbor sets should prefer lemma/surface only.
        """
        if m_neighbors is None:
            return 1
        if m_neighbors >= 6:
            return 0
        return 1

    def _natural_word_for_base_stem(self, stem: str) -> str | None:
        """Prefer lemma/surface for query tokens; allow safe stem so the query is not empty."""
        n = self._natural_word_for_stem(stem)
        if n:
            return n
        return self._safe_stem_as_display(stem)

    def _stem_key(self, token: str) -> str:
        """Canonical dedupe key across inflections (e.g., type/types)."""
        stems = preprocess(token)
        return stems[0] if stems else token.lower()

    def _compose_display_query(
        self,
        normalized_query: str,
        new_stems: list[str],
        m_neighbors: int | None = None,
    ) -> str:
        """
        Build display query with natural words only.

        - Base query uses normalized terms converted to natural words.
        - Added expansion terms are naturalized and deduped by stem family.
        - Optional clean stem fallback for *expansion* terms when m_neighbors is small
          (see _expansion_stem_fallback_slots); no stem fallback when m_neighbors >= 6.
        """
        base_stems = normalized_query.split()
        base_tokens: list[str] = []
        seen_keys: set[str] = set()

        for stem in base_stems:
            natural = self._natural_word_for_base_stem(stem)
            if not natural:
                continue
            key = self._stem_key(natural)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            base_tokens.append(natural)

        fallback_left = self._expansion_stem_fallback_slots(m_neighbors)
        added_tokens: list[str] = []
        for stem in new_stems:
            # Drop glued compounds that simply append/prepend original query stems,
            # e.g. "eruptionvolcano" for query stem "volcano".
            if any(
                q_stem in stem and q_stem != stem and len(stem) > len(q_stem) + 2
                for q_stem in base_stems
            ):
                continue
            natural = self._natural_word_for_stem(stem)
            if not natural and fallback_left > 0:
                sf = self._safe_stem_as_display(stem)
                if sf:
                    natural = sf
                    fallback_left -= 1
            if not natural:
                continue
            key = self._stem_key(natural)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            added_tokens.append(natural)

        if base_tokens or added_tokens:
            return " ".join(base_tokens + added_tokens)
        return normalized_query

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
        # debug logging removed

        return self._compose_display_query(query, new_terms, m_neighbors=None)

    # =====================================================================
    #  HELPERS FOR LOCAL CLUSTERING (QE Local Strategies)
    # =====================================================================
    def _get_local_doc_set(self, query: str, top_k: int) -> set[str]:
        normalized_query = self._normalize_query_for_expansion(query)
        results = self.engine.search(normalized_query, method="hits", top_k=top_k)
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
        m_neighbors: int,
    ) -> str:
        """
        Build a deterministic expanded query from ranked candidate terms.

        - Keeps original query token order.
        - Adds top-scoring non-duplicate candidate terms.
        - Falls back to original query when candidates are weak/empty.
        """
        original_stem_set = set(query_stems)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        new_terms = []
        for term, score in ranked:
            if score <= 0:
                continue
            if term in original_stem_set or term in new_terms:
                continue
            if not self._is_candidate_term(term, query_stems, query_len=query_len):
                continue
            new_terms.append(term)
            if len(new_terms) >= max_new_terms:
                break

        return self._compose_display_query(query, new_terms, m_neighbors=m_neighbors)

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
            m_neighbors=m_neighbors,
        )

    # =====================================================================
    #  3. SCALAR CLUSTERS
    # =====================================================================
    def association_cosine_uv(
        self,
        u: str,
        v: str,
        local_doc_ids: set[str],
        local_tf,
    ) -> float:
        """See module-level association_cosine_doc_frequency; thin wrapper."""
        if u not in local_tf or v not in local_tf:
            return 0.0
        return association_cosine_doc_frequency(
            local_tf[u], local_tf[v], local_doc_ids
        )

    def expand_scalar(
        self,
        query: str,
        top_k_docs: int = 50,
        m_neighbors: int = 4,
        max_new_terms: int = 6,
    ) -> str:
        """
        Cosine similarity of association vectors (document-frequency on D_l).
        Formula: S(u,v) = (s_u · s_v) / (|s_u| × |s_v|).
        """
        query = self._normalize_query_for_expansion(query)
        local_doc_ids = self._get_local_doc_set(query, top_k_docs)
        if not local_doc_ids:
            return query

        local_tf = self._get_local_term_frequencies(local_doc_ids)
        query_stems = preprocess(query)
        query_stem_set = set(query_stems)
        query_len = len(query_stems)
        terms = list(local_tf.keys())

        candidate_scores = defaultdict(float)

        for q_term in query_stems:
            if q_term not in local_tf:
                continue
            correlations = defaultdict(float)
            for v_term in terms:
                if not self._is_candidate_term(
                    v_term, query_stem_set, query_len=query_len
                ):
                    continue
                s_uv = self.association_cosine_uv(
                    q_term, v_term, local_doc_ids, local_tf
                )
                if s_uv > 0:
                    correlations[v_term] = s_uv

            scored_neighbors = [
                (term, self._score_expansion_term(term, score, query_stem_set))
                for term, score in correlations.items()
            ]

            top_neighbors = sorted(
                scored_neighbors, key=lambda x: x[1], reverse=True
            )[:m_neighbors]
            for neighbor, score in top_neighbors:
                candidate_scores[neighbor] = max(candidate_scores[neighbor], score)

        return self._finalize_expansion(
            query=query,
            query_stems=query_stem_set,
            candidate_scores=candidate_scores,
            max_new_terms=max_new_terms,
            query_len=query_len,
            m_neighbors=m_neighbors,
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
            m_neighbors=m_neighbors,
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