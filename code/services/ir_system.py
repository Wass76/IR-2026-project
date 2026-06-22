"""
Service-oriented IR system: loads indexes once and exposes search/refinement operations.
"""

import time
import math

from config import (
    DATASET_NAME,
    MAX_DOCS,
    TOP_K,
    REBUILD_VECTOR_INDEX,
    VECTOR_INDEX_PATH,
    VECTOR_DOC_IDS_PATH,
    EMBEDDING_METADATA_PATH,
    BM25_PARAMS_PATH,
    HYBRID_SERIAL_CANDIDATES,
    HYBRID_PARALLEL_DEPTH,
    RRF_K,
    HYBRID_FUSION_METHOD,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    EMBEDDING_MODEL,
)
from data_loader import load_or_build_processed_corpus
from preprocessing import TextPreprocessor
from index_cache import load_or_build_lexical_index
from retrieval import SearchEngine, DenseSearchEngine, HybridSearchEngine
from embeddings import EmbeddingModel
from vector_index import VectorIndex
from bm25_tuning import load_tuned_params
from query_refinement import QueryRefiner
from document_store import DocumentStore
from utils import setup_logger, load_json

logger = setup_logger("IRSystem")


class IRSystem:
    """In-memory IR services layer (preprocess, index, retrieve, refine)."""

    def __init__(self):
        self.loaded = False
        self.max_docs = MAX_DOCS
        self.docs = {}
        self.processed_docs = {}
        self.document_store = DocumentStore()
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
        self.inverted_index_cached = False
        self.bm25_cached = False
        self.tfidf_cached = False
        self.load_seconds = 0.0
        self.documents_source = None
        self.processed_source = None

    @staticmethod
    def get_hybrid_settings():
        return {
            "serial": {
                "mode": "serial",
                "description": "BM25 candidate generation → dense re-ranking",
                "bm25_candidates": HYBRID_SERIAL_CANDIDATES,
            },
            "parallel": {
                "mode": "parallel",
                "description": "BM25 + dense retrieval → score fusion",
                "bm25_depth": HYBRID_PARALLEL_DEPTH,
                "dense_depth": HYBRID_PARALLEL_DEPTH,
                "fusion_method": HYBRID_FUSION_METHOD,
                "rrf_k": RRF_K,
                "bm25_weight": HYBRID_BM25_WEIGHT,
                "dense_weight": HYBRID_DENSE_WEIGHT,
            },
            "embedding_model": EMBEDDING_MODEL,
        }

    @staticmethod
    def model_label(model):
        labels = {
            "bm25": "BM25",
            "tfidf": "TF-IDF",
            "embedding": "Dense Embedding",
            "hybrid_serial": "Hybrid Serial",
            "hybrid_parallel": "Hybrid Parallel",
            "hybrid": "Hybrid Parallel",
        }
        return labels.get(model.lower(), model)

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
            logger.info("Vector index cache hit")
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

        self.preprocessor = TextPreprocessor(use_stemming=True)
        (
            self.docs,
            self.processed_docs,
            self.documents_source,
            self.processed_source,
        ) = load_or_build_processed_corpus(
            self.document_store,
            self.preprocessor,
            max_docs=max_docs,
            dataset_name=DATASET_NAME,
        )
        self.max_docs = max_docs

        k1, b, _ = load_tuned_params()
        self.bm25_k1, self.bm25_b = k1, b

        (
            self.index,
            bm25_cache,
            tfidf_cache,
            lexical_flags,
        ) = load_or_build_lexical_index(
            self.processed_docs,
            dataset_name=DATASET_NAME,
            max_docs=max_docs,
            preprocessor=self.preprocessor,
            bm25_k1=k1,
            bm25_b=b,
        )
        self.inverted_index_cached = lexical_flags["inverted_index_cached"]
        self.bm25_cached = lexical_flags["bm25_cached"]
        self.tfidf_cached = lexical_flags["tfidf_cached"]

        self.search_engine = SearchEngine(
            self.index,
            bm25_k1=k1,
            bm25_b=b,
            bm25_cache=bm25_cache,
            tfidf_cache=tfidf_cache,
        )

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
        documents_cached = (
            self.processed_source == "sqlite" and self.documents_source == "sqlite"
        )
        return {
            "loaded": self.loaded,
            "dataset": DATASET_NAME,
            "max_docs": self.max_docs,
            "doc_count": len(self.docs),
            "unique_terms": len(self.index.index) if self.index else 0,
            "documents_cached": documents_cached,
            "inverted_index_cached": self.inverted_index_cached,
            "bm25_cached": self.bm25_cached,
            "tfidf_cached": self.tfidf_cached,
            "vector_index_cached": self.vector_cache_hit,
            "vector_index_loaded": self.dense_engine is not None,
            "vector_cache_hit": self.vector_cache_hit,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "bm25_params_tuned": BM25_PARAMS_PATH.exists(),
            "load_seconds": self.load_seconds,
            "document_store": {
                "path": self.document_store.db_path,
                "doc_count": self.document_store.document_count(),
                "corpus_meta": self.document_store.get_corpus_meta(),
                "documents_source": self.documents_source,
                "processed_source": self.processed_source,
            },
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

    def search(
        self,
        query,
        model="bm25",
        top_k=TOP_K,
        use_refinement=False,
        bm25_weight=None,
        dense_weight=None,
    ):
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
                query_tokens,
                query,
                top_k=top_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
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
                "rank": i + 1,
                "doc_id": doc_id,
                "score": round(float(score), 6) if math.isfinite(float(score)) else 0.0,
                "snippet": self._snippet(doc_id),
                "full_content": self._original_content(doc_id),
            }
            for i, (doc_id, score) in enumerate(ranked)
        ]

        hybrid_settings = None
        if model in ("hybrid_serial", "hybrid_parallel", "hybrid"):
            all_settings = self.get_hybrid_settings()
            if model == "hybrid_serial":
                hybrid_settings = all_settings["serial"]
            else:
                hybrid_settings = dict(all_settings["parallel"])
                if bm25_weight is not None:
                    hybrid_settings["bm25_weight"] = bm25_weight
                if dense_weight is not None:
                    hybrid_settings["dense_weight"] = dense_weight
                if bm25_weight is not None or dense_weight is not None:
                    hybrid_settings["fusion_method"] = "weighted"

        return {
            "query": query,
            "model": model,
            "model_label": self.model_label(model),
            "top_k": top_k,
            "use_refinement": use_refinement,
            "refinement": refinement,
            "hybrid_settings": hybrid_settings,
            "result_count": len(results),
            "results": results,
        }

    def get_document(self, doc_id):
        """Return the persisted original document for a doc_id."""
        record = self.document_store.get_document(doc_id)
        if record is not None:
            return record
        content = self.docs.get(doc_id)
        if content is None:
            return None
        return {
            "doc_id": doc_id,
            "original_content": content,
            "metadata": None,
        }

    def _original_content(self, doc_id):
        record = self.get_document(doc_id)
        return record["original_content"] if record else ""

    def _snippet(self, doc_id, max_len=200):
        text = self._original_content(doc_id)
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
