"""
Query refinement: spelling correction, PRF expansion, and history-based suggestions.
"""

import difflib
from collections import Counter, defaultdict

from config import (
    PRF_TOP_DOCS,
    PRF_EXPAND_TERMS,
    SPELL_MIN_TOKEN_LEN,
    SPELL_CORRECTION_CUTOFF,
    HISTORY_MAX_SUGGESTIONS,
    HISTORY_MIN_OVERLAP,
    HISTORY_WINDOW,
)
from utils import setup_logger

logger = setup_logger("QueryRefinement")


class QueryRefiner:
    """Refine queries before retrieval."""

    def __init__(self, index, processed_docs, preprocessor):
        self.index = index
        self.processed_docs = processed_docs
        self.preprocessor = preprocessor
        self.vocabulary = set(index.index.keys())
        self._spell_buckets = self._build_spelling_buckets()
        self._spell_cache = {}

    def _build_spelling_buckets(self):
        """Bucket vocabulary by first char + length for fast spelling lookup."""
        buckets = defaultdict(list)
        for term in self.vocabulary:
            if len(term) >= SPELL_MIN_TOKEN_LEN:
                buckets[(term[0], len(term))].append(term)
        return buckets

    def _spelling_candidates(self, token):
        candidates = []
        for delta in (-1, 0, 1):
            key = (token[0], len(token) + delta)
            candidates.extend(self._spell_buckets.get(key, []))
        return candidates

    def correct_spelling(self, tokens):
        """Map OOV tokens to nearest in-vocabulary terms."""
        corrected = []
        changes = []
        for token in tokens:
            if token in self.vocabulary or len(token) < SPELL_MIN_TOKEN_LEN:
                corrected.append(token)
                continue

            if token in self._spell_cache:
                fixed = self._spell_cache[token]
                corrected.append(fixed)
                if fixed != token:
                    changes.append({"from": token, "to": fixed})
                continue

            candidates = self._spelling_candidates(token)
            matches = difflib.get_close_matches(
                token,
                candidates,
                n=1,
                cutoff=SPELL_CORRECTION_CUTOFF,
            ) if candidates else []

            fixed = matches[0] if matches else token
            self._spell_cache[token] = fixed
            corrected.append(fixed)
            if fixed != token:
                changes.append({"from": token, "to": fixed})

        return corrected, changes

    def prf_expand_from_hits(self, query_tokens, initial_hits, num_terms=None):
        """PRF expansion using precomputed BM25 hits (no extra retrieval)."""
        num_terms = num_terms or PRF_EXPAND_TERMS
        if not initial_hits:
            return list(query_tokens), []

        query_set = set(query_tokens)
        term_scores = defaultdict(float)

        for doc_id, _ in initial_hits:
            for term, tf in self._doc_term_counts(doc_id).items():
                if term in query_set:
                    continue
                term_scores[term] += tf * self.index.get_idf(term)

        ranked = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        added = [term for term, _ in ranked[:num_terms]]
        return list(query_tokens) + added, added

    def prf_expand(self, query_tokens, search_engine, top_docs=None, num_terms=None):
        """Pseudo-relevance feedback: add top terms from initial BM25 hits."""
        top_docs = top_docs or PRF_TOP_DOCS
        initial = search_engine.retrieve(query_tokens, model="bm25", top_k=top_docs)
        return self.prf_expand_from_hits(query_tokens, initial, num_terms=num_terms)

    def _doc_term_counts(self, doc_id):
        tokens = self.processed_docs.get(doc_id, [])
        return Counter(tokens)

    def suggest_from_history_tokens(self, query_tokens, history_token_lists, history_raw):
        if not history_token_lists:
            return []

        query_set = set(query_tokens)
        if not query_set:
            return []

        scored = []
        for past_tokens, past_text in zip(history_token_lists, history_raw):
            past_set = set(past_tokens)
            if not past_set:
                continue
            overlap = len(query_set & past_set)
            if overlap < HISTORY_MIN_OVERLAP:
                continue
            jaccard = overlap / len(query_set | past_set)
            scored.append((jaccard, past_text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:HISTORY_MAX_SUGGESTIONS]]

    def history_expand_from_tokens(self, query_tokens, history_token_lists, history_raw):
        """Add frequent terms from similar historical queries (pre-tokenized)."""
        suggestions = self.suggest_from_history_tokens(
            query_tokens, history_token_lists, history_raw
        )
        if not suggestions:
            return list(query_tokens), [], []

        suggestion_sets = {
            text: set(tokens)
            for text, tokens in zip(history_raw, history_token_lists)
            if text in suggestions
        }

        query_set = set(query_tokens)
        extra = Counter()
        for text in suggestions:
            for token in suggestion_sets.get(text, set()):
                if token not in query_set:
                    extra[token] += 1

        added = [term for term, _ in extra.most_common(PRF_EXPAND_TERMS)]
        return list(query_tokens) + added, added, suggestions

    def suggest_from_history(self, query_tokens, history_queries):
        """Suggest similar past queries by token overlap (Jaccard)."""
        if not history_queries:
            return []

        history_tokens = [self.preprocessor.process_text(text) for text in history_queries]
        return self.suggest_from_history_tokens(query_tokens, history_tokens, history_queries)

    def history_expand(self, query_tokens, history_queries):
        """Add frequent terms from similar historical queries."""
        if not history_queries:
            return query_tokens, []

        history_tokens = [self.preprocessor.process_text(text) for text in history_queries]
        expanded, added, _ = self.history_expand_from_tokens(
            query_tokens, history_tokens, history_queries
        )
        return expanded, added

    def refine(self, query_tokens, raw_query, search_engine, history_queries=None):
        """Full refinement pipeline (single-query API path)."""
        history_queries = history_queries or []
        history_tokens = [self.preprocessor.process_text(text) for text in history_queries]

        tokens, spelling_changes = self.correct_spelling(list(query_tokens))
        tokens, prf_terms = self.prf_expand(tokens, search_engine)
        tokens, history_terms, suggestions = self.history_expand_from_tokens(
            tokens, history_tokens, history_queries
        )

        return {
            "tokens": tokens,
            "original_tokens": list(query_tokens),
            "raw_query": raw_query,
            "spelling_changes": spelling_changes,
            "prf_terms": prf_terms,
            "history_terms": history_terms,
            "suggestions": suggestions,
        }

    def refine_collection(self, processed_queries, raw_queries, search_engine):
        """Refine all queries using one batch BM25 pass for PRF."""
        refined = {}
        refinement_log = {}
        history_tokens = []
        history_raw = []

        logger.info(f"Refining {len(processed_queries)} queries...")
        logger.info("Batch BM25 retrieval for PRF (one pass)...")
        prf_hits = search_engine.retrieve_batch(
            processed_queries,
            model="bm25",
            top_k=PRF_TOP_DOCS,
        )

        for i, q_id in enumerate(processed_queries):
            tokens, spelling_changes = self.correct_spelling(list(processed_queries[q_id]))
            tokens, prf_terms = self.prf_expand_from_hits(
                tokens,
                prf_hits.get(q_id, []),
            )
            tokens, history_terms, suggestions = self.history_expand_from_tokens(
                tokens,
                history_tokens,
                history_raw,
            )

            refined[q_id] = tokens
            refinement_log[q_id] = {
                "raw_query": raw_queries[q_id],
                "original_token_count": len(processed_queries[q_id]),
                "refined_token_count": len(tokens),
                "spelling_changes": spelling_changes,
                "prf_terms": prf_terms,
                "history_terms": history_terms,
                "suggestions": suggestions,
            }

            history_tokens.append(processed_queries[q_id])
            history_raw.append(raw_queries[q_id])
            if len(history_tokens) > HISTORY_WINDOW:
                history_tokens = history_tokens[-HISTORY_WINDOW:]
                history_raw = history_raw[-HISTORY_WINDOW:]

            if (i + 1) % 500 == 0:
                logger.info(f"Refined {i + 1} queries...")

        expanded_count = sum(
            1 for entry in refinement_log.values()
            if entry["refined_token_count"] > entry["original_token_count"]
        )
        logger.info(
            f"Refinement done: {expanded_count}/{len(refined)} queries expanded."
        )
        return refined, refinement_log
