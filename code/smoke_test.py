"""
End-to-end smoke test using real BEIR Quora ground truth.

Ground-truth pairs are taken from:
  ~/.ir_datasets/beir/quora/test.qrels
  ~/.ir_datasets/beir/quora/source/quora/queries.jsonl
  ~/.ir_datasets/beir/quora/source/quora/corpus.jsonl

Usage (from code/):
  python smoke_test.py
  python smoke_test.py --max-docs 1000
  python smoke_test.py --max-docs 200000
"""

import argparse
import sys

from config import DATASET_NAME, MAX_DOCS, TOP_K
from data_loader import load_dataset, load_or_build_processed_corpus, get_qrels, get_queries
from preprocessing import TextPreprocessor
from index_cache import load_or_build_lexical_index
from retrieval import SearchEngine, DenseSearchEngine, HybridSearchEngine
from evaluation import Evaluator
from embeddings import EmbeddingModel
from vector_index import VectorIndex
from bm25_tuning import load_tuned_params
from document_store import DocumentStore
from utils import setup_logger

logger = setup_logger("SmokeTest")

# Real test cases from test.qrels + queries.jsonl
# quick=True  → relevant doc_id is within the first ~1000 corpus docs
# quick=False → needs larger corpus (doc ids in 100k+ range)
TEST_CASES = [
    {
        "query_id": "187",
        "query_text": "What causes a nightmare?",
        "relevant_doc_ids": ["188", "144864", "202157", "68733", "68732"],
        "must_hit": "188",
        "quick": True,
        "note": "Doc 188 is an early corpus duplicate.",
    },
    {
        "query_id": "273",
        "query_text": "Does it matter whether humans are selfish or evil?",
        "relevant_doc_ids": ["274"],
        "must_hit": "274",
        "quick": True,
        "note": "Doc 274 is an early near-duplicate.",
    },
    {
        "query_id": "46",
        "query_text": "Which question should I ask on Quora?",
        "relevant_doc_ids": ["134031", "271267", "134030"],
        "must_hit": "134031",
        "quick": False,
        "note": "Quora duplicate-question cluster (needs large corpus).",
    },
    {
        "query_id": "313",
        "query_text": (
            "If I do not monetize YouTube videos & upload copyright content, "
            "then are there chances that Google may block my account?"
        ),
        "relevant_doc_ids": ["314", "71619", "71620"],
        "must_hit": "314",
        "quick": False,
        "note": "YouTube copyright cluster.",
    },
]


def _build_engines(docs, processed_docs, preprocessor, max_docs):
    k1, b, _ = load_tuned_params()
    index, bm25_cache, tfidf_cache, _ = load_or_build_lexical_index(
        processed_docs,
        dataset_name=DATASET_NAME,
        max_docs=max_docs,
        preprocessor=preprocessor,
        bm25_k1=k1,
        bm25_b=b,
    )
    search_engine = SearchEngine(
        index, bm25_k1=k1, bm25_b=b, bm25_cache=bm25_cache, tfidf_cache=tfidf_cache
    )

    doc_ids = list(docs.keys())
    embedder = EmbeddingModel()
    texts = [docs[did] for did in doc_ids]
    embeddings = embedder.encode_documents(texts)
    vector_index = VectorIndex()
    vector_index.build(doc_ids, embeddings)

    dense_engine = DenseSearchEngine(vector_index, embedder)
    hybrid_engine = HybridSearchEngine(search_engine, dense_engine)
    return search_engine, dense_engine, hybrid_engine, index


def _run_model(name, retrieve_fn, cases, preprocessor, evaluator, top_k):
    results = {}
    passed = 0
    for case in cases:
        qid = case["query_id"]
        tokens = preprocessor.process_text(case["query_text"])
        ranked = retrieve_fn(tokens, case["query_text"])
        results[qid] = ranked
        hits = [doc_id for doc_id, _ in ranked[:top_k]]
        ok = case["must_hit"] in hits
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        logger.info(
            f"  [{status}] {name} | q={qid} | must_hit={case['must_hit']} | "
            f"top3={[h[0] for h in ranked[:3]]}"
        )
    metrics = evaluator.evaluate_all(results, k=top_k)
    return passed, metrics


def main():
    parser = argparse.ArgumentParser(description="BEIR Quora smoke test")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--skip-dense", action="store_true", help="Skip embedding/hybrid (faster)")
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs is not None else MAX_DOCS
    top_k = args.top_k

    cases = [c for c in TEST_CASES if c["quick"] or max_docs >= 100000]
    if not cases:
        logger.error("No test cases selected. Use --max-docs 200000 for full cases.")
        sys.exit(1)

    skipped = [c["query_id"] for c in TEST_CASES if c not in cases]
    if skipped:
        logger.info(f"Skipping query ids {skipped} (need larger --max-docs).")

    logger.info("=" * 60)
    logger.info(f"Smoke test | dataset={DATASET_NAME} | max_docs={max_docs} | top_k={top_k}")
    logger.info("=" * 60)

    dataset = load_dataset(DATASET_NAME)
    store = DocumentStore()
    preprocessor = TextPreprocessor(use_stemming=True)
    docs, processed_docs, _, _ = load_or_build_processed_corpus(
        store, preprocessor, dataset=dataset, max_docs=max_docs, dataset_name=DATASET_NAME
    )
    qrels = get_qrels(dataset)
    evaluator = Evaluator(qrels)

    logger.info(f"Corpus size: {len(docs)} docs")
    for case in cases:
        logger.info(f"Case q={case['query_id']}: {case['note']}")

    search_engine, dense_engine, hybrid_engine, _ = _build_engines(
        docs, processed_docs, preprocessor, max_docs
    )

    models = [
        ("bm25", lambda tokens, text: search_engine.retrieve(tokens, model="bm25", top_k=top_k)),
        ("tfidf", lambda tokens, text: search_engine.retrieve(tokens, model="tfidf", top_k=top_k)),
    ]
    if not args.skip_dense:
        models.extend(
            [
                ("embedding", lambda tokens, text: dense_engine.retrieve(text, top_k=top_k)),
                (
                    "hybrid_serial",
                    lambda tokens, text: hybrid_engine.retrieve_serial(tokens, text, top_k=top_k),
                ),
                (
                    "hybrid_parallel",
                    lambda tokens, text: hybrid_engine.retrieve_parallel(tokens, text, top_k=top_k),
                ),
            ]
        )

    total_pass = 0
    total_checks = 0
    summary = {}

    for model_name, retrieve_fn in models:
        logger.info(f"\n--- {model_name} ---")
        passed, metrics = _run_model(
            model_name, retrieve_fn, cases, preprocessor, evaluator, top_k
        )
        total_pass += passed
        total_checks += len(cases)
        summary[model_name] = {
            "must_hit_pass": f"{passed}/{len(cases)}",
            "MAP": round(metrics.get("MAP", 0), 4),
            "nDCG@10": round(metrics.get("nDCG@10", 0), 4),
        }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (must_hit in top-k for real qrels pairs)")
    for model_name, row in summary.items():
        logger.info(
            f"  {model_name:16} pass={row['must_hit_pass']:5}  "
            f"MAP={row['MAP']:.4f}  nDCG@10={row['nDCG@10']:.4f}"
        )
    logger.info("=" * 60)

    if total_pass == 0:
        sys.exit(2)
    if total_pass < total_checks:
        logger.warning("Some checks failed — inspect top3 doc ids above.")
        sys.exit(1)
    logger.info("All must_hit checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
