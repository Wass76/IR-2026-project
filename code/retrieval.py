import math
from collections import defaultdict

from config import (
    TOP_K,
    BM25_K1,
    BM25_B,
    HYBRID_SERIAL_CANDIDATES,
    HYBRID_PARALLEL_DEPTH,
    RRF_K,
    HYBRID_FUSION_METHOD,
    HYBRID_BM25_WEIGHT,
)
from utils import setup_logger

logger = setup_logger("Retrieval")


class SearchEngine:
    def __init__(self, index, bm25_k1=None, bm25_b=None):
        self.index = index
        self.bm25_k1 = BM25_K1 if bm25_k1 is None else bm25_k1
        self.bm25_b = BM25_B if bm25_b is None else bm25_b

    def score_tfidf(self, query_tokens):
        scores = defaultdict(float)
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1
        for term, q_tf in query_tf.items():
            postings = self.index.get_term_postings(term)
            if not postings:
                continue
            idf = self.index.get_idf(term)
            for doc_id, doc_tf in postings.items():
                scores[doc_id] += (1 + math.log10(doc_tf)) * idf * (1 + math.log10(q_tf)) * idf
        return scores

    def score_bm25(self, query_tokens):
        scores = defaultdict(float)
        k1, b = self.bm25_k1, self.bm25_b
        avg_dl, N = self.index.avg_doc_length, self.index.total_docs
        for term in query_tokens:
            postings = self.index.get_term_postings(term)
            if not postings:
                continue
            df = self.index.get_df(term)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in postings.items():
                dl = self.index.doc_lengths[doc_id]
                scores[doc_id] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl)))
        return scores

    def retrieve(self, query_tokens, model="bm25", top_k=TOP_K):
        if not query_tokens:
            return []
        scores = self.score_tfidf(query_tokens) if model.lower() == "tfidf" else self.score_bm25(query_tokens)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def retrieve_batch(self, processed_queries, model="bm25", top_k=TOP_K):
        logger.info(f"Retrieving results for {len(processed_queries)} queries using {model}...")
        results = {}
        for i, (query_id, tokens) in enumerate(processed_queries.items()):
            results[query_id] = self.retrieve(tokens, model, top_k)
            if (i + 1) % 100 == 0:
                logger.info(f"Retrieved {i + 1} queries...")
        return results


class DenseSearchEngine:
    def __init__(self, vector_index, embedding_model):
        self.vector_index = vector_index
        self.embedding_model = embedding_model

    def _encode_query(self, query_text):
        return self.embedding_model.encode_single(query_text)

    def retrieve(self, query_text, top_k=TOP_K):
        query_vector = self._encode_query(query_text)
        if query_vector is None:
            return []
        return self.vector_index.search(query_vector, top_k=top_k)

    def rerank(self, query_text, candidate_doc_ids, top_k=TOP_K, query_vector=None):
        """Re-rank BM25 candidates using dense cosine similarity (serial hybrid stage)."""
        if query_vector is None:
            query_vector = self._encode_query(query_text)
        if query_vector is None or not candidate_doc_ids:
            return []
        ranked = self.vector_index.score_candidates(query_vector, candidate_doc_ids)
        return ranked[:top_k]

    def rerank_batch(self, processed_queries, raw_queries, bm25_engine, top_k=TOP_K):
        """
        Serial hybrid batch: BM25 candidates per query, then dense re-rank.
        Encodes all queries once to avoid per-query embedding overhead.
        """
        logger.info(
            f"Serial hybrid batch: {len(processed_queries)} queries "
            f"(batch query encoding)..."
        )
        query_vectors = self.embedding_model.encode_queries(raw_queries)
        results = {}

        for i, q_id in enumerate(processed_queries):
            candidates = bm25_engine.retrieve(
                processed_queries[q_id],
                model="bm25",
                top_k=HYBRID_SERIAL_CANDIDATES,
            )
            candidate_ids = [doc_id for doc_id, _ in candidates]
            results[q_id] = self.rerank(
                raw_queries[q_id],
                candidate_ids,
                top_k=top_k,
                query_vector=query_vectors.get(q_id),
            )
            if (i + 1) % 100 == 0:
                logger.info(f"Serial hybrid: processed {i + 1} queries...")

        return results

    def retrieve_batch(self, raw_queries, top_k=TOP_K):
        logger.info(f"Dense retrieval for {len(raw_queries)} queries...")
        query_vectors = self.embedding_model.encode_queries(raw_queries)
        results = {}
        for i, (query_id, query_vector) in enumerate(query_vectors.items()):
            results[query_id] = self.vector_index.search(query_vector, top_k=top_k)
            if (i + 1) % 100 == 0:
                logger.info(f"Retrieved {i + 1} queries...")
        return results


class HybridSearchEngine:
    """
    Hybrid retrieval per project spec:
    - serial: BM25 candidate generation -> dense re-ranking
    - parallel: independent BM25 + dense retrieval -> score-level fusion (RRF or weighted)
    """

    def __init__(self, bm25_engine, dense_engine):
        self.bm25_engine = bm25_engine
        self.dense_engine = dense_engine

    @staticmethod
    def _normalize_scores(ranked_results):
        if not ranked_results:
            return {}
        scores = [score for _, score in ranked_results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {doc_id: 1.0 for doc_id, _ in ranked_results}
        return {
            doc_id: (score - min_s) / (max_s - min_s)
            for doc_id, score in ranked_results
        }

    def rrf_fusion(self, bm25_results, dense_results, k=RRF_K):
        scores = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            scores[doc_id] += 1.0 / (k + rank)
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def weighted_fusion(self, bm25_results, dense_results, bm25_weight=HYBRID_BM25_WEIGHT):
        bm25_norm = self._normalize_scores(bm25_results)
        dense_norm = self._normalize_scores(dense_results)
        all_docs = set(bm25_norm) | set(dense_norm)
        dense_weight = 1.0 - bm25_weight
        fused = [
            (
                doc_id,
                bm25_weight * bm25_norm.get(doc_id, 0.0)
                + dense_weight * dense_norm.get(doc_id, 0.0),
            )
            for doc_id in all_docs
        ]
        return sorted(fused, key=lambda x: x[1], reverse=True)

    def fuse_results(self, bm25_results, dense_results, fusion_method=None):
        method = (fusion_method or HYBRID_FUSION_METHOD).lower()
        if method == "weighted":
            return self.weighted_fusion(bm25_results, dense_results)
        return self.rrf_fusion(bm25_results, dense_results)

    def retrieve_serial(self, query_tokens, query_text, top_k=TOP_K):
        """Lexical retrieval first, then dense re-ranking on BM25 candidates."""
        candidates = self.bm25_engine.retrieve(
            query_tokens,
            model="bm25",
            top_k=HYBRID_SERIAL_CANDIDATES,
        )
        candidate_ids = [doc_id for doc_id, _ in candidates]
        return self.dense_engine.rerank(query_text, candidate_ids, top_k=top_k)

    def retrieve_parallel(self, query_tokens, query_text, top_k=TOP_K, fusion_method=None):
        """Run BM25 and dense in parallel, then fuse ranked lists."""
        bm25_results = self.bm25_engine.retrieve(
            query_tokens,
            model="bm25",
            top_k=HYBRID_PARALLEL_DEPTH,
        )
        dense_results = self.dense_engine.retrieve(
            query_text,
            top_k=HYBRID_PARALLEL_DEPTH,
        )
        return self.fuse_results(bm25_results, dense_results, fusion_method)[:top_k]

    def retrieve_batch(self, processed_queries, raw_queries, top_k=TOP_K, mode="parallel"):
        if mode == "serial":
            return self._retrieve_batch_serial(processed_queries, raw_queries, top_k)
        return self._retrieve_batch_parallel(processed_queries, raw_queries, top_k)

    def _retrieve_batch_serial(self, processed_queries, raw_queries, top_k):
        return self.dense_engine.rerank_batch(
            processed_queries,
            raw_queries,
            self.bm25_engine,
            top_k=top_k,
        )

    def _retrieve_batch_parallel(self, processed_queries, raw_queries, top_k):
        logger.info(
            f"Parallel hybrid: BM25 + dense with {HYBRID_FUSION_METHOD} fusion..."
        )
        bm25_results = self.bm25_engine.retrieve_batch(
            processed_queries,
            model="bm25",
            top_k=HYBRID_PARALLEL_DEPTH,
        )
        dense_results = self.dense_engine.retrieve_batch(
            raw_queries,
            top_k=HYBRID_PARALLEL_DEPTH,
        )
        return {
            q_id: self.fuse_results(
                bm25_results.get(q_id, []),
                dense_results.get(q_id, []),
            )[:top_k]
            for q_id in processed_queries
        }
