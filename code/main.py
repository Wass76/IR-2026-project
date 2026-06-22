import time
import argparse
from datetime import datetime

from config import *
from utils import setup_logger, save_json, load_json
from data_loader import (
    load_dataset,
    load_or_build_processed_corpus,
    get_queries,
    get_qrels,
    select_eval_queries,
)
from preprocessing import TextPreprocessor
from index_cache import load_or_build_lexical_index
from retrieval import SearchEngine, DenseSearchEngine, HybridSearchEngine
from evaluation import Evaluator
from embeddings import EmbeddingModel
from vector_index import VectorIndex
from bm25_tuning import run_tuning, save_tuned_params, load_tuned_params
from query_refinement import QueryRefiner
from document_store import DocumentStore

logger = setup_logger("Main")
document_store = DocumentStore()

MAP_FORMULA = "standard AP: sum(precision@i * rel_i) / total_relevant, averaged over queries"


def _baseline_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = BASELINE_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dense_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = DENSE_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _bm25_tuning_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = BM25_TUNING_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _refinement_run_dir(max_docs):
    label = "run_full" if max_docs is None else f"run_{max_docs}"
    run_dir = REFINEMENT_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_search_engine(index, bm25_cache=None, tfidf_cache=None):
    k1, b, tuned = load_tuned_params()
    if tuned:
        logger.info(f"Using tuned BM25 params: k1={k1}, b={b}")
    else:
        logger.info(f"Using default BM25 params: k1={k1}, b={b}")
    return (
        SearchEngine(
            index,
            bm25_k1=k1,
            bm25_b=b,
            bm25_cache=bm25_cache,
            tfidf_cache=tfidf_cache,
        ),
        k1,
        b,
    )


def _load_lexical_index(processed_docs, preprocessor, max_docs):
    k1, b, _ = load_tuned_params()
    index, bm25_cache, tfidf_cache, _ = load_or_build_lexical_index(
        processed_docs,
        dataset_name=DATASET_NAME,
        max_docs=max_docs,
        preprocessor=preprocessor,
        bm25_k1=k1,
        bm25_b=b,
    )
    search_engine, k1, b = _make_search_engine(index, bm25_cache, tfidf_cache)
    return index, search_engine, k1, b


def _metrics_for_comparison(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key != "Evaluated_Queries"
    }


def _load_lexical_baseline_metrics(max_docs):
    baseline_path = _baseline_run_dir(max_docs) / "baseline_comparison.json"
    if baseline_path.exists():
        return load_json(baseline_path)
    return None


def _build_or_load_vector_index(docs, max_docs, timing):
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


def run_baseline(max_docs=None, max_eval_queries=None):
    if max_docs is None:
        max_docs = MAX_DOCS
    if max_eval_queries is None:
        max_eval_queries = MAX_EVAL_QUERIES

    run_dir = _baseline_run_dir(max_docs)
    timing = {}
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"Lexical baseline run — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    logger.info("\n--- Phase 1: Loading data ---")
    phase_start = time.time()
    dataset = load_dataset(DATASET_NAME)
    preprocessor = TextPreprocessor(use_stemming=True)
    docs, processed_docs, _, _ = load_or_build_processed_corpus(
        document_store,
        preprocessor,
        dataset=dataset,
        max_docs=max_docs,
        dataset_name=DATASET_NAME,
    )
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)

    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=max_eval_queries,
        random_seed=EVAL_RANDOM_SEED,
    )

    processed_queries = preprocessor.process_collection(eval_queries)

    index, search_engine, bm25_k1, bm25_b = _load_lexical_index(
        processed_docs, preprocessor, max_docs
    )
    evaluator = Evaluator(qrels)
    timing["load_and_index"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 2: Retrieval (BM25) ---")
    phase_start = time.time()
    results_bm25 = search_engine.retrieve_batch(
        processed_queries, model="bm25", top_k=TOP_K
    )
    timing["retrieve_bm25"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 3: Retrieval (TF-IDF) ---")
    phase_start = time.time()
    results_tfidf = search_engine.retrieve_batch(
        processed_queries, model="tfidf", top_k=TOP_K
    )
    timing["retrieve_tfidf"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 4: Evaluation ---")
    phase_start = time.time()
    metrics_bm25 = evaluator.evaluate_all(results_bm25, k=TOP_K)
    metrics_tfidf = evaluator.evaluate_all(results_tfidf, k=TOP_K)
    timing["evaluate"] = round(time.time() - phase_start, 3)
    timing["total"] = round(time.time() - total_start, 3)

    save_json(metrics_bm25, run_dir / "evaluation_metrics_bm25.json")
    save_json(metrics_tfidf, run_dir / "evaluation_metrics_tfidf.json")

    if SAVE_RETRIEVAL_RESULTS:
        save_json(results_bm25, run_dir / "retrieval_results_bm25.json")
        save_json(results_tfidf, run_dir / "retrieval_results_tfidf.json")

    retrieval_timing = {
        "bm25": timing["retrieve_bm25"],
        "tfidf": timing["retrieve_tfidf"],
    }
    comparison = Evaluator.compare_models(
        {"bm25": metrics_bm25, "tfidf": metrics_tfidf},
        k=TOP_K,
        timing=retrieval_timing,
    )
    Evaluator.log_comparison(comparison, title="Lexical baseline comparison")
    Evaluator.save_comparison(comparison, run_dir / "baseline_comparison.json")

    metadata = {
        "label": "before_additional_features",
        "dataset": DATASET_NAME,
        "max_docs": max_docs,
        "doc_count": len(docs),
        "total_queries": len(queries),
        "evaluated_queries": metrics_bm25.get("Evaluated_Queries", 0),
        "top_k": TOP_K,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_params_tuned": BM25_PARAMS_PATH.exists(),
        "map_formula": MAP_FORMULA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timing_seconds": timing,
    }
    save_json(metadata, run_dir / "baseline_metadata.json")

    return comparison


def run_dense_eval(max_docs=None, max_eval_queries=None, hybrid_type="both"):
    if max_docs is None:
        max_docs = MAX_DOCS
    if max_eval_queries is None:
        max_eval_queries = MAX_EVAL_QUERIES

    run_dir = _dense_run_dir(max_docs)
    timing = {}
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"Dense + hybrid run — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    logger.info("\n--- Phase 1: Loading data ---")
    phase_start = time.time()
    dataset = load_dataset(DATASET_NAME)
    preprocessor = TextPreprocessor(use_stemming=True)
    docs, processed_docs, _, _ = load_or_build_processed_corpus(
        document_store,
        preprocessor,
        dataset=dataset,
        max_docs=max_docs,
        dataset_name=DATASET_NAME,
    )
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)
    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=max_eval_queries,
        random_seed=EVAL_RANDOM_SEED,
    )

    processed_queries = preprocessor.process_collection(eval_queries)

    index, search_engine, bm25_k1, bm25_b = _load_lexical_index(
        processed_docs, preprocessor, max_docs
    )
    timing["load_and_index"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 2: Vector index ---")
    phase_start = time.time()
    vector_index, embedder, cache_hit = _build_or_load_vector_index(docs, max_docs, timing)
    timing["vector_index_total"] = round(time.time() - phase_start, 3)

    dense_engine = DenseSearchEngine(vector_index, embedder)
    hybrid_engine = HybridSearchEngine(search_engine, dense_engine)
    evaluator = Evaluator(qrels)
    retrieval_timing = {}

    results_hybrid_serial = {}
    results_hybrid_parallel = {}

    logger.info("\n--- Phase 3: Lexical retrieval (same corpus) ---")
    phase_start = time.time()
    results_bm25 = search_engine.retrieve_batch(
        processed_queries, model="bm25", top_k=TOP_K
    )
    retrieval_timing["bm25"] = round(time.time() - phase_start, 3)

    phase_start = time.time()
    results_tfidf = search_engine.retrieve_batch(
        processed_queries, model="tfidf", top_k=TOP_K
    )
    retrieval_timing["tfidf"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 4: Dense retrieval ---")
    phase_start = time.time()
    results_embedding = dense_engine.retrieve_batch(eval_queries, top_k=TOP_K)
    retrieval_timing["embedding"] = round(time.time() - phase_start, 3)

    run_serial = hybrid_type in ("serial", "both")
    run_parallel = hybrid_type in ("parallel", "both")

    if run_serial:
        logger.info("\n--- Phase 5: Serial hybrid (BM25 -> dense re-rank) ---")
        phase_start = time.time()
        results_hybrid_serial = hybrid_engine.retrieve_batch(
            processed_queries,
            eval_queries,
            top_k=TOP_K,
            mode="serial",
        )
        retrieval_timing["hybrid_serial"] = round(time.time() - phase_start, 3)

    if run_parallel:
        logger.info("\n--- Phase 6: Parallel hybrid (RRF fusion) ---")
        phase_start = time.time()
        results_hybrid_parallel = hybrid_engine.retrieve_batch(
            processed_queries,
            eval_queries,
            top_k=TOP_K,
            mode="parallel",
        )
        retrieval_timing["hybrid_parallel"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 7: Evaluation ---")
    phase_start = time.time()
    metrics_bm25 = evaluator.evaluate_all(results_bm25, k=TOP_K)
    metrics_tfidf = evaluator.evaluate_all(results_tfidf, k=TOP_K)
    metrics_embedding = evaluator.evaluate_all(results_embedding, k=TOP_K)
    metrics_hybrid_serial = (
        evaluator.evaluate_all(results_hybrid_serial, k=TOP_K)
        if results_hybrid_serial else {}
    )
    metrics_hybrid_parallel = (
        evaluator.evaluate_all(results_hybrid_parallel, k=TOP_K)
        if results_hybrid_parallel else {}
    )
    timing["evaluate"] = round(time.time() - phase_start, 3)
    timing["total"] = round(time.time() - total_start, 3)

    save_json(metrics_embedding, run_dir / "evaluation_metrics_embedding.json")
    save_json(metrics_bm25, run_dir / "evaluation_metrics_bm25.json")
    save_json(metrics_tfidf, run_dir / "evaluation_metrics_tfidf.json")
    if metrics_hybrid_serial:
        save_json(metrics_hybrid_serial, run_dir / "evaluation_metrics_hybrid_serial.json")
    if metrics_hybrid_parallel:
        save_json(metrics_hybrid_parallel, run_dir / "evaluation_metrics_hybrid_parallel.json")

    if SAVE_RETRIEVAL_RESULTS:
        save_json(results_embedding, run_dir / "retrieval_results_embedding.json")
        if results_hybrid_serial:
            save_json(results_hybrid_serial, run_dir / "retrieval_results_hybrid_serial.json")
        if results_hybrid_parallel:
            save_json(results_hybrid_parallel, run_dir / "retrieval_results_hybrid_parallel.json")

    all_metrics = {
        "bm25": metrics_bm25,
        "tfidf": metrics_tfidf,
        "embedding": metrics_embedding,
    }
    if metrics_hybrid_serial:
        all_metrics["hybrid_serial"] = metrics_hybrid_serial
    if metrics_hybrid_parallel:
        all_metrics["hybrid_parallel"] = metrics_hybrid_parallel

    comparison = Evaluator.compare_models(
        all_metrics,
        k=TOP_K,
        timing=retrieval_timing,
    )
    Evaluator.log_comparison(comparison, title="Full representation comparison")

    lexical_baseline = _load_lexical_baseline_metrics(max_docs)
    if lexical_baseline and "models" in lexical_baseline:
        def _baseline_metrics(model_name):
            entry = lexical_baseline["models"].get(model_name, {})
            return entry.get("metrics", entry)

        comparison["lexical_baseline_before_features"] = lexical_baseline["models"]
        comparison["before_after"] = {
            "bm25": evaluator.before_after_analysis(
                _baseline_metrics("bm25"),
                comparison["models"]["bm25"]["metrics"],
                k=TOP_K,
            ),
            "tfidf": evaluator.before_after_analysis(
                _baseline_metrics("tfidf"),
                comparison["models"]["tfidf"]["metrics"],
                k=TOP_K,
            ),
        }

    hybrid_models = {
        name: data["metrics"]
        for name, data in comparison["models"].items()
        if name.startswith("hybrid_")
    }
    if len(hybrid_models) >= 2:
        serial_map = hybrid_models.get("hybrid_serial", {}).get("MAP", 0)
        parallel_map = hybrid_models.get("hybrid_parallel", {}).get("MAP", 0)
        comparison["recommended_hybrid"] = (
            "hybrid_parallel" if parallel_map >= serial_map else "hybrid_serial"
        )
        comparison["hybrid_choice_reason"] = (
            f"Higher MAP on this run "
            f"(serial={serial_map:.4f}, parallel={parallel_map:.4f})"
        )

    Evaluator.save_comparison(comparison, run_dir / "full_comparison.json")

    metadata = {
        "label": "dense_and_hybrid",
        "dataset": DATASET_NAME,
        "max_docs": max_docs,
        "doc_count": len(docs),
        "total_queries": len(queries),
        "evaluated_queries": metrics_embedding.get("Evaluated_Queries", 0),
        "top_k": TOP_K,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "embedding_model": EMBEDDING_MODEL,
        "hybrid_serial_candidates": HYBRID_SERIAL_CANDIDATES,
        "hybrid_parallel_depth": HYBRID_PARALLEL_DEPTH,
        "hybrid_fusion_method": HYBRID_FUSION_METHOD,
        "rrf_k": RRF_K,
        "recommended_hybrid": comparison.get("recommended_hybrid"),
        "index_cache_hit": cache_hit,
        "map_formula": MAP_FORMULA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timing_seconds": timing,
        "retrieval_timing_seconds": retrieval_timing,
    }
    save_json(metadata, run_dir / "dense_metadata.json")

    logger.info(f"\nIndex cache hit: {cache_hit}")
    if comparison.get("recommended_hybrid"):
        logger.info(
            f"Recommended hybrid for UI: {comparison['recommended_hybrid']} "
            f"({comparison.get('hybrid_choice_reason', '')})"
        )
    logger.info(f"Artifacts saved to: {run_dir}")

    return comparison


def run_bm25_tuning(max_docs=None, max_eval_queries=None):
    if max_docs is None:
        max_docs = MAX_DOCS
    if max_eval_queries is None:
        max_eval_queries = BM25_TUNING_MAX_QUERIES

    run_dir = _bm25_tuning_run_dir(max_docs)
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"BM25 parameter tuning — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    logger.info("\n--- Phase 1: Load data and build index ---")
    dataset = load_dataset(DATASET_NAME)
    preprocessor = TextPreprocessor(use_stemming=True)
    docs, processed_docs, _, _ = load_or_build_processed_corpus(
        document_store,
        preprocessor,
        dataset=dataset,
        max_docs=max_docs,
        dataset_name=DATASET_NAME,
    )
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)
    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=max_eval_queries,
        random_seed=EVAL_RANDOM_SEED,
    )

    processed_queries = preprocessor.process_collection(eval_queries)

    k1, b, _ = load_tuned_params()
    index, _, _, _ = load_or_build_lexical_index(
        processed_docs,
        dataset_name=DATASET_NAME,
        max_docs=max_docs,
        preprocessor=preprocessor,
        bm25_k1=k1,
        bm25_b=b,
    )

    logger.info("\n--- Phase 2: Grid search ---")
    report = run_tuning(index, processed_docs, processed_queries, qrels, top_k=TOP_K)

    best = report["best_params"]
    save_json(report, run_dir / "bm25_tuning_report.json")
    save_json(report["grid_results"], run_dir / "bm25_tuning_grid.json")
    save_tuned_params(
        best["k1"],
        best["b"],
        {
            "source_run": str(run_dir),
            "dataset": DATASET_NAME,
            "max_docs": max_docs,
            "evaluated_queries": len(processed_queries),
            "primary_metric": report["primary_metric"],
            "metrics": best["metrics"],
            "justification_summary": report["justification"]["summary"],
        },
    )

    logger.info("\n--- BM25 tuning summary ---")
    logger.info(f"Best params: k1={best['k1']}, b={best['b']}")
    logger.info(f"MAP: {best['metrics'].get('MAP', 0):.4f}")
    logger.info(f"nDCG@{TOP_K}: {best['metrics'].get(f'nDCG@{TOP_K}', 0):.4f}")
    logger.info("\nTheoretical justification:")
    for line in report["justification"]["theoretical_justification"]:
        logger.info(f"  - {line}")

    logger.info("\nQuery-length bucket performance (tuned params):")
    for bucket, data in report["query_bucket_analysis"].items():
        m = data.get("metrics", {})
        logger.info(
            f"  {bucket}: n={data.get('query_count', 0)} "
            f"MAP={m.get('MAP', 0):.4f} nDCG@{TOP_K}={m.get(f'nDCG@{TOP_K}', 0):.4f}"
        )

    elapsed = round(time.time() - total_start, 2)
    logger.info(f"\nTuning completed in {elapsed}s. Artifacts: {run_dir}")
    return report


def run_refinement_eval(max_docs=None, max_eval_queries=None):
    """Evaluate BM25 before and after query refinement."""
    if max_docs is None:
        max_docs = MAX_DOCS
    if max_eval_queries is None:
        max_eval_queries = MAX_EVAL_QUERIES

    run_dir = _refinement_run_dir(max_docs)
    timing = {}
    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"Query refinement eval — max_docs={max_docs or 'full'}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("=" * 50)

    logger.info("\n--- Phase 1: Load data and build index ---")
    phase_start = time.time()
    dataset = load_dataset(DATASET_NAME)
    preprocessor = TextPreprocessor(use_stemming=True)
    docs, processed_docs, _, _ = load_or_build_processed_corpus(
        document_store,
        preprocessor,
        dataset=dataset,
        max_docs=max_docs,
        dataset_name=DATASET_NAME,
    )
    queries = get_queries(dataset)
    qrels = get_qrels(dataset)
    eval_queries = select_eval_queries(
        queries,
        qrels,
        max_queries=max_eval_queries,
        random_seed=EVAL_RANDOM_SEED,
    )

    processed_queries = preprocessor.process_collection(eval_queries)

    index, search_engine, bm25_k1, bm25_b = _load_lexical_index(
        processed_docs, preprocessor, max_docs
    )
    evaluator = Evaluator(qrels)
    timing["load_and_index"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 2: Retrieval BEFORE refinement ---")
    phase_start = time.time()
    results_before = search_engine.retrieve_batch(
        processed_queries, model="bm25", top_k=TOP_K
    )
    timing["retrieve_before"] = round(time.time() - phase_start, 3)
    metrics_before = evaluator.evaluate_all(results_before, k=TOP_K)

    logger.info("\n--- Phase 3: Query refinement ---")
    phase_start = time.time()
    refiner = QueryRefiner(index, processed_docs, preprocessor)
    refined_queries, refinement_log = refiner.refine_collection(
        processed_queries,
        eval_queries,
        search_engine,
    )
    timing["refine"] = round(time.time() - phase_start, 3)

    logger.info("\n--- Phase 4: Retrieval AFTER refinement ---")
    phase_start = time.time()
    results_after = search_engine.retrieve_batch(
        refined_queries, model="bm25", top_k=TOP_K
    )
    timing["retrieve_after"] = round(time.time() - phase_start, 3)
    metrics_after = evaluator.evaluate_all(results_after, k=TOP_K)
    timing["total"] = round(time.time() - total_start, 3)

    before_after = evaluator.before_after_analysis(metrics_before, metrics_after, k=TOP_K)
    comparison = Evaluator.compare_models(
        {"before_refinement": metrics_before, "after_refinement": metrics_after},
        k=TOP_K,
        timing={
            "before_refinement": timing["retrieve_before"],
            "after_refinement": timing["retrieve_after"],
        },
    )
    Evaluator.log_comparison(comparison, title="Query refinement before/after")

    save_json(metrics_before, run_dir / "evaluation_metrics_before.json")
    save_json(metrics_after, run_dir / "evaluation_metrics_after.json")
    save_json(before_after, run_dir / "before_after_analysis.json")
    Evaluator.save_comparison(comparison, run_dir / "refinement_comparison.json")

    sample_ids = list(refinement_log.keys())[:10]
    save_json(
        {qid: refinement_log[qid] for qid in sample_ids},
        run_dir / "refinement_samples.json",
    )
    if SAVE_RETRIEVAL_RESULTS:
        save_json(refinement_log, run_dir / "refinement_log.json")

    metadata = {
        "label": "query_refinement",
        "dataset": DATASET_NAME,
        "max_docs": max_docs,
        "doc_count": len(docs),
        "evaluated_queries": metrics_before.get("Evaluated_Queries", 0),
        "top_k": TOP_K,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "prf_top_docs": PRF_TOP_DOCS,
        "prf_expand_terms": PRF_EXPAND_TERMS,
        "techniques": [
            "spelling_correction",
            "pseudo_relevance_feedback",
            "history_based_suggestions",
        ],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timing_seconds": timing,
        "before_after": before_after,
    }
    save_json(metadata, run_dir / "refinement_metadata.json")

    logger.info("\n--- Refinement summary ---")
    logger.info(f"Before MAP: {metrics_before.get('MAP', 0):.4f}")
    logger.info(f"After  MAP: {metrics_after.get('MAP', 0):.4f}")
    logger.info(f"Delta MAP:  {before_after['delta'].get('MAP', 0):+.4f}")
    logger.info(f"Artifacts saved to: {run_dir}")

    return {
        "comparison": comparison,
        "before_after": before_after,
        "refinement_log": refinement_log,
    }


def run_full_eval(max_docs=None, max_eval_queries=None):
    """Run baseline (before features) then dense+hybrid (after features)."""
    logger.info("Running full evaluation pipeline (baseline + dense + hybrid)...")
    baseline = run_baseline(max_docs=max_docs, max_eval_queries=max_eval_queries)
    dense = run_dense_eval(
        max_docs=max_docs,
        max_eval_queries=max_eval_queries,
        hybrid_type="both",
    )
    return {"baseline": baseline, "dense_and_hybrid": dense}


def run_api(host=None, port=None):
    """Start the SOA REST API gateway (FastAPI + uvicorn)."""
    import uvicorn
    from api.app import create_app

    host = host or API_HOST
    port = port or API_PORT
    logger.info(f"Starting API gateway at http://{host}:{port}")
    logger.info("Open / for the search UI, /docs for API documentation")
    uvicorn.run(create_app(), host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="IR Project pipeline")
    parser.add_argument(
        "--mode",
        choices=["baseline", "dense", "hybrid", "full", "tune-bm25", "refinement", "api"],
        default="baseline",
        help="baseline | dense | hybrid | full | tune-bm25 | refinement | api",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap documents (default: MAX_DOCS from config)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap eval queries (default: MAX_EVAL_QUERIES from config)",
    )
    parser.add_argument(
        "--hybrid-type",
        choices=["none", "serial", "parallel", "both"],
        default=None,
        help="Hybrid variants: none | serial | parallel | both (default: both for hybrid/full, none for dense)",
    )
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs is not None else MAX_DOCS
    max_queries = args.max_queries if args.max_queries is not None else MAX_EVAL_QUERIES
    if args.hybrid_type is not None:
        hybrid_type = args.hybrid_type
    elif args.mode in ("hybrid", "full"):
        hybrid_type = "both"
    else:
        hybrid_type = "none"

    start_time = time.time()
    try:
        if args.mode == "full":
            run_full_eval(max_docs=max_docs, max_eval_queries=max_queries)
        elif args.mode == "tune-bm25":
            run_bm25_tuning(max_docs=max_docs, max_eval_queries=max_queries)
        elif args.mode == "refinement":
            run_refinement_eval(max_docs=max_docs, max_eval_queries=max_queries)
        elif args.mode == "api":
            run_api()
        elif args.mode in ("dense", "hybrid"):
            run_dense_eval(
                max_docs=max_docs,
                max_eval_queries=max_queries,
                hybrid_type=hybrid_type,
            )
        else:
            run_baseline(max_docs=max_docs, max_eval_queries=max_queries)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
    finally:
        execution_time = (time.time() - start_time) / 60
        logger.info("=" * 50)
        logger.info(f"Execution completed in {execution_time:.2f} minutes.")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
