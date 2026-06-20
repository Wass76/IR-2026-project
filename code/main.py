import time
import argparse
from datetime import datetime

from config import *
from utils import setup_logger, save_json, save_pickle, load_json
from data_loader import (
    load_dataset,
    get_documents,
    get_queries,
    get_qrels,
    select_eval_queries,
)
from preprocessing import TextPreprocessor
from indexing import InvertedIndex
from retrieval import SearchEngine, DenseSearchEngine
from evaluation import Evaluator
from embeddings import EmbeddingModel
from vector_index import VectorIndex

logger = setup_logger("Main")

MAP_FORMULA = "standard AP: sum(precision@i * rel_i) / total_relevant, averaged over queries"


def _baseline_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = BASELINE_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _metrics_for_comparison(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key != "Evaluated_Queries"
    }


def run_baseline(max_docs=None):
    """
    Run lexical baseline evaluation (TF-IDF + BM25) and save artifacts.

    Args:
        max_docs: document cap (None = full corpus). Defaults to MAX_DOCS from config.
    """
    if max_docs is None:
        max_docs = MAX_DOCS

    run_dir = _baseline_run_dir(max_docs)
    timing = {}
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"Lexical baseline run — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    # 1. Load data
    logger.info("\n--- Phase 1: Loading data ---")
    phase_start = time.time()
    dataset = load_dataset(DATASET_NAME)
    docs = get_documents(dataset, max_docs=max_docs)
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)
    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=MAX_EVAL_QUERIES,
        random_seed=EVAL_RANDOM_SEED,
    )
    timing["load"] = round(time.time() - phase_start, 3)

    # 2. Preprocess
    logger.info("\n--- Phase 2: Preprocessing ---")
    phase_start = time.time()
    preprocessor = TextPreprocessor(use_stemming=True)
    processed_docs = preprocessor.process_collection(docs)
    processed_queries = preprocessor.process_collection(eval_queries)
    timing["preprocess"] = round(time.time() - phase_start, 3)

    # 3. Index
    logger.info("\n--- Phase 3: Indexing ---")
    phase_start = time.time()
    index = InvertedIndex()
    index.build(processed_docs)
    save_pickle(index, MODELS_DIR / "inverted_index.pkl")
    timing["index"] = round(time.time() - phase_start, 3)

    search_engine = SearchEngine(index)
    evaluator = Evaluator(qrels)

    # 4. Retrieve BM25
    logger.info("\n--- Phase 4: Retrieval (BM25) ---")
    phase_start = time.time()
    results_bm25 = search_engine.retrieve_batch(
        processed_queries, model="bm25", top_k=TOP_K
    )
    timing["retrieve_bm25"] = round(time.time() - phase_start, 3)

    # 5. Retrieve TF-IDF
    logger.info("\n--- Phase 5: Retrieval (TF-IDF) ---")
    phase_start = time.time()
    results_tfidf = search_engine.retrieve_batch(
        processed_queries, model="tfidf", top_k=TOP_K
    )
    timing["retrieve_tfidf"] = round(time.time() - phase_start, 3)

    # 6. Evaluate both models
    logger.info("\n--- Phase 6: Evaluation ---")
    phase_start = time.time()
    metrics_bm25 = evaluator.evaluate_all(results_bm25, k=TOP_K)
    metrics_tfidf = evaluator.evaluate_all(results_tfidf, k=TOP_K)
    timing["evaluate"] = round(time.time() - phase_start, 3)
    timing["total"] = round(time.time() - total_start, 3)

    # 7. Save artifacts
    save_json(metrics_bm25, run_dir / "evaluation_metrics_bm25.json")
    save_json(metrics_tfidf, run_dir / "evaluation_metrics_tfidf.json")

    if SAVE_RETRIEVAL_RESULTS:
        save_json(results_bm25, run_dir / "retrieval_results_bm25.json")
        save_json(results_tfidf, run_dir / "retrieval_results_tfidf.json")

    metadata = {
        "label": "before_additional_features",
        "dataset": DATASET_NAME,
        "max_docs": max_docs,
        "doc_count": len(docs),
        "total_queries": len(queries),
        "evaluated_queries": metrics_bm25.get("Evaluated_Queries", 0),
        "top_k": TOP_K,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "map_formula": MAP_FORMULA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timing_seconds": timing,
    }
    save_json(metadata, run_dir / "baseline_metadata.json")

    comparison = {
        "label": "before_additional_features",
        "dataset": DATASET_NAME,
        "doc_count": len(docs),
        "evaluated_queries": metrics_bm25.get("Evaluated_Queries", 0),
        "timing_seconds": timing,
        "models": {
            "bm25": _metrics_for_comparison(metrics_bm25),
            "tfidf": _metrics_for_comparison(metrics_tfidf),
        },
    }
    save_json(comparison, run_dir / "baseline_comparison.json")

    logger.info("\n--- Baseline summary ---")
    logger.info(f"Evaluated queries: {comparison['evaluated_queries']}")
    logger.info(f"BM25  MAP: {metrics_bm25.get('MAP', 0):.4f}  nDCG@{TOP_K}: {metrics_bm25.get(f'nDCG@{TOP_K}', 0):.4f}")
    logger.info(f"TF-IDF MAP: {metrics_tfidf.get('MAP', 0):.4f}  nDCG@{TOP_K}: {metrics_tfidf.get(f'nDCG@{TOP_K}', 0):.4f}")
    logger.info(f"Artifacts saved to: {run_dir}")

    return comparison


def _dense_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = DENSE_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_lexical_baseline_metrics(max_docs):
    """Load lexical baseline comparison if available for side-by-side report."""
    baseline_path = _baseline_run_dir(max_docs) / "baseline_comparison.json"
    if baseline_path.exists():
        return load_json(baseline_path)
    return None


def _build_or_load_vector_index(docs, max_docs, timing):
    """
    Build or load cached vector index for the document corpus.

    Returns:
        tuple (VectorIndex, EmbeddingModel, cache_hit: bool)
    """
    doc_ids = list(docs.keys())
    doc_count = len(doc_ids)
    cached = EmbeddingModel.load_metadata(EMBEDDING_METADATA_PATH)
    index_files_exist = (
        VECTOR_INDEX_PATH.exists() and VECTOR_DOC_IDS_PATH.exists()
    )

    embedder = EmbeddingModel()

    if (
        not REBUILD_VECTOR_INDEX
        and index_files_exist
        and EmbeddingModel.metadata_matches(cached, doc_count, max_docs)
    ):
        logger.info("Loading cached vector index (metadata match)...")
        phase_start = time.time()
        vector_index = VectorIndex()
        vector_index.load(VECTOR_INDEX_PATH, VECTOR_DOC_IDS_PATH)
        timing["embed_docs"] = 0.0
        timing["build_index"] = round(time.time() - phase_start, 3)
        timing["index_cache_hit"] = True
        return vector_index, embedder, True

    logger.info("Building vector index from scratch...")
    phase_start = time.time()
    texts = [docs[did] for did in doc_ids]
    embeddings = embedder.encode_documents(texts)
    timing["embed_docs"] = round(time.time() - phase_start, 3)

    phase_start = time.time()
    vector_index = VectorIndex()
    vector_index.build(doc_ids, embeddings)
    vector_index.save(VECTOR_INDEX_PATH, VECTOR_DOC_IDS_PATH)
    metadata = EmbeddingModel.build_metadata(doc_count, max_docs)
    EmbeddingModel.save_metadata(metadata, EMBEDDING_METADATA_PATH)
    timing["build_index"] = round(time.time() - phase_start, 3)
    timing["index_cache_hit"] = False

    return vector_index, embedder, False


def run_dense_eval(max_docs=None):
    """
    Run dense embedding retrieval evaluation and save artifacts.

    Args:
        max_docs: document cap (None = full corpus). Defaults to MAX_DOCS from config.
    """
    if max_docs is None:
        max_docs = MAX_DOCS

    run_dir = _dense_run_dir(max_docs)
    timing = {}
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"Dense embedding run — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    # 1. Load data (raw text — no lexical preprocessing for embeddings)
    logger.info("\n--- Phase 1: Loading data ---")
    phase_start = time.time()
    dataset = load_dataset(DATASET_NAME)
    docs = get_documents(dataset, max_docs=max_docs)
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)
    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=MAX_EVAL_QUERIES,
        random_seed=EVAL_RANDOM_SEED,
    )
    timing["load"] = round(time.time() - phase_start, 3)

    # 2. Build or load vector index
    logger.info("\n--- Phase 2: Vector index (embed + build/load) ---")
    phase_start = time.time()
    vector_index, embedder, cache_hit = _build_or_load_vector_index(docs, max_docs, timing)
    timing["vector_index_total"] = round(time.time() - phase_start, 3)

    dense_engine = DenseSearchEngine(vector_index, embedder)
    evaluator = Evaluator(qrels)

    # 3. Dense retrieval
    logger.info("\n--- Phase 3: Dense retrieval ---")
    phase_start = time.time()
    results_embedding = dense_engine.retrieve_batch(eval_queries, top_k=TOP_K)
    timing["retrieve_embedding"] = round(time.time() - phase_start, 3)

    # 4. Evaluate
    logger.info("\n--- Phase 4: Evaluation ---")
    phase_start = time.time()
    metrics_embedding = evaluator.evaluate_all(results_embedding, k=TOP_K)
    timing["evaluate"] = round(time.time() - phase_start, 3)
    timing["total"] = round(time.time() - total_start, 3)

    # 5. Save artifacts
    save_json(metrics_embedding, run_dir / "evaluation_metrics_embedding.json")

    if SAVE_RETRIEVAL_RESULTS:
        save_json(results_embedding, run_dir / "retrieval_results_embedding.json")

    lexical_baseline = _load_lexical_baseline_metrics(max_docs)

    metadata = {
        "label": "dense_embedding",
        "dataset": DATASET_NAME,
        "max_docs": max_docs,
        "doc_count": len(docs),
        "total_queries": len(queries),
        "evaluated_queries": metrics_embedding.get("Evaluated_Queries", 0),
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "index_cache_hit": cache_hit,
        "map_formula": MAP_FORMULA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timing_seconds": timing,
    }
    save_json(metadata, run_dir / "dense_metadata.json")

    comparison = {
        "label": "dense_embedding",
        "dataset": DATASET_NAME,
        "doc_count": len(docs),
        "evaluated_queries": metrics_embedding.get("Evaluated_Queries", 0),
        "timing_seconds": timing,
        "models": {
            "embedding": _metrics_for_comparison(metrics_embedding),
        },
    }
    if lexical_baseline and "models" in lexical_baseline:
        comparison["lexical_baseline"] = lexical_baseline["models"]
        comparison["lexical_baseline_timing"] = lexical_baseline.get("timing_seconds")

    save_json(comparison, run_dir / "dense_comparison.json")

    logger.info("\n--- Dense embedding summary ---")
    logger.info(f"Evaluated queries: {comparison['evaluated_queries']}")
    logger.info(
        f"Embedding MAP: {metrics_embedding.get('MAP', 0):.4f}  "
        f"nDCG@{TOP_K}: {metrics_embedding.get(f'nDCG@{TOP_K}', 0):.4f}"
    )
    logger.info(f"Index cache hit: {cache_hit}")
    logger.info(f"Artifacts saved to: {run_dir}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="IR Project pipeline")
    parser.add_argument(
        "--mode",
        choices=["baseline", "dense"],
        default="baseline",
        help="Run lexical baseline (default) or dense embedding evaluation",
    )
    args = parser.parse_args()

    start_time = time.time()
    try:
        if args.mode == "dense":
            run_dense_eval(max_docs=MAX_DOCS)
        else:
            run_baseline(max_docs=MAX_DOCS)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
    finally:
        execution_time = (time.time() - start_time) / 60
        logger.info("=" * 50)
        logger.info(f"Execution completed in {execution_time:.2f} minutes.")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
