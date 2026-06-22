"""
Service-oriented IR system: loads indexes once and exposes search/refinement operations.
"""

import time

from config import (
    DATASET_NAME,
    MAX_DOCS,
    TOP_K,
    REBUILD_VECTOR_INDEX,
    VECTOR_INDEX_PATH,
    VECTOR_DOC_IDS_PATH,
    EMBEDDING_METADATA_PATH,
    BM25_PARAMS_PATH,
)
from data_loader import load_dataset, get_documents
from preprocessing import TextPreprocessor
from indexing import InvertedIndex
from retrieval import SearchEngine, DenseSearchEngine, HybridSearchEngine
from embeddings import EmbeddingModel
from vector_index import VectorIndex
from bm25_tuning import load_tuned_params
from query_refinement import QueryRefiner
from utils import setup_logger, load_json

logger = setup_logger("IRSystem")


class IRSystem:
    """In-memory IR services layer (preprocess, index, retrieve, refine)."""

    def __init__(self):
        self.loaded = False
        self.max_docs = MAX_DOCS
        self.docs = {}
        self.processed_docs = {}
        self.preprocessor = None
        self.index = None
        self.search_engine = None
        self.dense_engine = None
        self.hybrid_engine = None
        self.refiner = None
        self.query_history = []
        self.search_count = 0
        self.last_query = None
        self.last_model = None
        self.bm25_k1 = None
        self.bm25_b = None
        self.vector_cache_hit = False
        self.load_seconds = 0.0

    def _build_or_load_vector_index(self, docs, max_docs):
        doc_ids = list(docs.keys())
        doc_count = len(doc_ids)
        cached = EmbeddingModel.load_metadata(EMBEDDING_METADATA_PATH)
        index_files_exist = VECTOR_INDEX_PATH.exists() and VECTOR_DOC_IDS_PATH.exists()
        embedder = EmbeddingModel()

        if (
            not REBUILD_VECTOR_INDEX
            and index_files_exist
            and EmbeddingModel.metadata_matches(cached, doc_count, max_docs)
        ):
            vector_index = VectorIndex()
            vector_index.load(VECTOR_INDEX_PATH, VECTOR_DOC_IDS_PATH)
            return vector_index, embedder, True

        texts = [docs[did] for did in doc_ids]
        embeddings = embedder.encode_documents(texts)
        vector_index = VectorIndex()
        vector_index.build(doc_ids, embeddings)
        vector_index.save(VECTOR_INDEX_PATH, VECTOR_DOC_IDS_PATH)
        metadata = EmbeddingModel.build_metadata(doc_count, max_docs)
        EmbeddingModel.save_metadata(metadata, EMBEDDING_METADATA_PATH)
        return vector_index, embedder, False

    def load(self, max_docs=None):
        if self.loaded:
            return self.status()

        max_docs = max_docs if max_docs is not None else MAX_DOCS
        start = time.time()
        logger.info(f"Loading IR system (dataset={DATASET_NAME}, max_docs={max_docs})...")

        dataset = load_dataset(DATASET_NAME)
        self.docs = get_documents(dataset, max_docs=max_docs)
        self.max_docs = max_docs

        self.preprocessor = TextPreprocessor(use_stemming=True)
        self.processed_docs = self.preprocessor.process_collection(self.docs)

        self.index = InvertedIndex()
        self.index.build(self.processed_docs)

        k1, b, _ = load_tuned_params()
        self.bm25_k1, self.bm25_b = k1, b
        self.search_engine = SearchEngine(self.index, bm25_k1=k1, bm25_b=b)

        vector_index, embedder, cache_hit = self._build_or_load_vector_index(
            self.docs, max_docs
        )
        self.vector_cache_hit = cache_hit
        self.dense_engine = DenseSearchEngine(vector_index, embedder)
        self.hybrid_engine = HybridSearchEngine(self.search_engine, self.dense_engine)
        self.refiner = QueryRefiner(self.index, self.processed_docs, self.preprocessor)

        self.loaded = True
        self.load_seconds = round(time.time() - start, 2)
        logger.info(f"IR system ready in {self.load_seconds}s")
        return self.status()

    def status(self):
        return {
            "loaded": self.loaded,
            "dataset": DATASET_NAME,
            "max_docs": self.max_docs,
            "doc_count": len(self.docs),
            "unique_terms": len(self.index.index) if self.index else 0,
            "vector_index_loaded": self.dense_engine is not None,
            "vector_cache_hit": self.vector_cache_hit,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "bm25_params_tuned": BM25_PARAMS_PATH.exists(),
            "load_seconds": self.load_seconds,
            "session": {
                "search_count": self.search_count,
                "history_size": len(self.query_history),
                "last_query": self.last_query,
                "last_model": self.last_model,
                "recent_queries": self.query_history[-5:],
            },
        }

    def preprocess(self, text):
        tokens = self.preprocessor.process_text(text)
        return {"text": text, "tokens": tokens, "token_count": len(tokens)}

    def refine(self, text):
        self._ensure_loaded()
        tokens = self.preprocessor.process_text(text)
        result = self.refiner.refine(
            tokens,
            text,
            self.search_engine,
            history_queries=self.query_history,
        )
        return result

    def search(self, query, model="bm25", top_k=TOP_K, use_refinement=False):
        self._ensure_loaded()
        top_k = top_k or TOP_K
        model = model.lower()
        refinement = None
        query_tokens = self.preprocessor.process_text(query)

        if use_refinement:
            refinement = self.refine(query)
            query_tokens = refinement["tokens"]

        if model in ("bm25", "tfidf"):
            ranked = self.search_engine.retrieve(query_tokens, model=model, top_k=top_k)
        elif model == "embedding":
            ranked = self.dense_engine.retrieve(query, top_k=top_k)
        elif model == "hybrid_serial":
            ranked = self.hybrid_engine.retrieve_serial(
                query_tokens, query, top_k=top_k
            )
        elif model in ("hybrid_parallel", "hybrid"):
            ranked = self.hybrid_engine.retrieve_parallel(
                query_tokens, query, top_k=top_k
            )
        else:
            raise ValueError(
                f"Unknown model '{model}'. "
                "Use: bm25, tfidf, embedding, hybrid_serial, hybrid_parallel"
            )

        self.query_history.append(query)
        self.search_count += 1
        self.last_query = query
        self.last_model = model
        results = [
            {
                "doc_id": doc_id,
                "score": round(float(score), 6),
                "snippet": self._snippet(doc_id),
            }
            for doc_id, score in ranked
        ]

        return {
            "query": query,
            "model": model,
            "top_k": top_k,
            "use_refinement": use_refinement,
            "refinement": refinement,
            "result_count": len(results),
            "results": results,
        }

    def _snippet(self, doc_id, max_len=200):
        text = self.docs.get(doc_id, "")
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    def _ensure_loaded(self):
        if not self.loaded:
            self.load()

    @staticmethod
    def latest_metrics():
        """Load latest saved evaluation summaries from results folder."""
        from config import RESULTS_DIR

        summaries = {}
        paths = {
            "baseline": RESULTS_DIR / "baseline",
            "dense": RESULTS_DIR / "dense",
            "refinement": RESULTS_DIR / "refinement",
            "bm25_tuning": RESULTS_DIR / "bm25_tuning",
        }
        for name, base in paths.items():
            if not base.exists():
                continue
            runs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not runs:
                continue
            run_dir = runs[0]
            for candidate in (
                "full_comparison.json",
                "baseline_comparison.json",
                "refinement_comparison.json",
                "bm25_tuning_report.json",
            ):
                path = run_dir / candidate
                if path.exists():
                    summaries[name] = {
                        "run_dir": str(run_dir),
                        "file": candidate,
                        "data": load_json(path),
                    }
                    break
        return summaries
