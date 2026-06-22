"""
Disk cache for inverted index, BM25, and TF-IDF retrieval structures.
"""

import time
from datetime import datetime

from config import (
    DATASET_NAME,
    INVERTED_INDEX_PATH,
    BM25_CACHE_PATH,
    TFIDF_CACHE_PATH,
    LEXICAL_INDEX_METADATA_PATH,
    REBUILD_LEXICAL_INDEX,
)
from indexing import InvertedIndex
from lexical_scoring import BM25Cache, TfidfCache, LEXICAL_CACHE_VERSION
from utils import setup_logger, save_pickle, load_pickle, save_json, load_json

logger = setup_logger("IndexCache")


class LexicalIndexCache:
    @staticmethod
    def _max_docs_key(max_docs):
        return "full" if max_docs is None else str(max_docs)

    @staticmethod
    def build_metadata(
        doc_count,
        max_docs,
        dataset_name,
        preprocessor,
        bm25_k1,
        bm25_b,
        unique_terms,
    ):
        return {
            "cache_version": LEXICAL_CACHE_VERSION,
            "bm25_library": "rank_bm25",
            "tfidf_library": "sklearn",
            "dataset": dataset_name,
            "max_docs": LexicalIndexCache._max_docs_key(max_docs),
            "doc_count": doc_count,
            "preprocessing_stemming": str(preprocessor.use_stemming).lower(),
            "preprocessing_lemmatization": str(
                preprocessor.use_lemmatization
            ).lower(),
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "unique_terms": unique_terms,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def metadata_matches(
        metadata,
        doc_count,
        max_docs,
        dataset_name,
        preprocessor,
        bm25_k1,
        bm25_b,
    ):
        if not metadata:
            return False
        return (
            int(metadata.get("cache_version", 0)) == LEXICAL_CACHE_VERSION
            and metadata.get("bm25_library") == "rank_bm25"
            and metadata.get("tfidf_library") == "sklearn"
            and metadata.get("dataset") == dataset_name
            and metadata.get("max_docs") == LexicalIndexCache._max_docs_key(max_docs)
            and int(metadata.get("doc_count", 0)) == doc_count
            and metadata.get("preprocessing_stemming")
            == str(preprocessor.use_stemming).lower()
            and metadata.get("preprocessing_lemmatization")
            == str(preprocessor.use_lemmatization).lower()
            and float(metadata.get("bm25_k1", -1)) == float(bm25_k1)
            and float(metadata.get("bm25_b", -1)) == float(bm25_b)
        )

    @staticmethod
    def _files_exist():
        return (
            INVERTED_INDEX_PATH.exists()
            and BM25_CACHE_PATH.exists()
            and TFIDF_CACHE_PATH.exists()
            and LEXICAL_INDEX_METADATA_PATH.exists()
        )

    @staticmethod
    def load(
        doc_count,
        max_docs,
        dataset_name,
        preprocessor,
        bm25_k1,
        bm25_b,
    ):
        metadata = load_json(LEXICAL_INDEX_METADATA_PATH)
        if not LexicalIndexCache.metadata_matches(
            metadata,
            doc_count,
            max_docs,
            dataset_name,
            preprocessor,
            bm25_k1,
            bm25_b,
        ):
            return None

        start = time.time()
        index = InvertedIndex.from_state(load_pickle(INVERTED_INDEX_PATH))
        logger.info(
            f"Inverted index cache hit — loaded in {time.time() - start:.2f}s"
        )

        start = time.time()
        bm25_cache = load_pickle(BM25_CACHE_PATH)
        logger.info(f"BM25 cache hit — loaded in {time.time() - start:.2f}s")

        start = time.time()
        tfidf_cache = load_pickle(TFIDF_CACHE_PATH)
        logger.info(f"TF-IDF cache hit — loaded in {time.time() - start:.2f}s")

        return index, bm25_cache, tfidf_cache

    @staticmethod
    def save(index, bm25_cache, tfidf_cache, metadata):
        save_pickle(index.to_state(), INVERTED_INDEX_PATH)
        save_pickle(bm25_cache, BM25_CACHE_PATH)
        save_pickle(tfidf_cache, TFIDF_CACHE_PATH)
        save_json(metadata, LEXICAL_INDEX_METADATA_PATH)
        logger.info(
            f"Saved lexical caches to {INVERTED_INDEX_PATH.parent} "
            f"({metadata['doc_count']} docs, {metadata['unique_terms']} terms)"
        )


def load_or_build_lexical_index(
    processed_docs,
    dataset_name=DATASET_NAME,
    max_docs=None,
    preprocessor=None,
    bm25_k1=1.5,
    bm25_b=0.75,
):
    """
    Load cached inverted index + BM25 + TF-IDF structures, or build and persist them.

    Returns:
        (index, bm25_cache, tfidf_cache, cache_flags)
    """
    doc_count = len(processed_docs)
    flags = {
        "inverted_index_cached": False,
        "bm25_cached": False,
        "tfidf_cached": False,
    }

    if (
        not REBUILD_LEXICAL_INDEX
        and LexicalIndexCache._files_exist()
        and preprocessor is not None
    ):
        loaded = LexicalIndexCache.load(
            doc_count,
            max_docs,
            dataset_name,
            preprocessor,
            bm25_k1,
            bm25_b,
        )
        if loaded is not None:
            index, bm25_cache, tfidf_cache = loaded
            flags = {
                "inverted_index_cached": True,
                "bm25_cached": True,
                "tfidf_cached": True,
            }
            return index, bm25_cache, tfidf_cache, flags

    logger.info("Building lexical retrieval structures (cache miss)...")
    index = InvertedIndex()
    index.build(processed_docs)

    start = time.time()
    bm25_cache = BM25Cache.build(processed_docs, bm25_k1, bm25_b)
    logger.info(
        f"Built BM25 model (rank_bm25) in {time.time() - start:.2f}s"
    )

    start = time.time()
    tfidf_cache = TfidfCache.build(processed_docs)
    logger.info(
        f"Built TF-IDF model (scikit-learn) in {time.time() - start:.2f}s"
    )

    metadata = LexicalIndexCache.build_metadata(
        doc_count,
        max_docs,
        dataset_name,
        preprocessor,
        bm25_k1,
        bm25_b,
        len(index.index),
    )
    LexicalIndexCache.save(index, bm25_cache, tfidf_cache, metadata)
    return index, bm25_cache, tfidf_cache, flags
