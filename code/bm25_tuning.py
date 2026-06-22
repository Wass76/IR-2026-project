"""
BM25 parameter tuning: grid search over k1 and b with corpus/query analysis
and theoretical justification for the chosen values.
"""

import time
from datetime import datetime
from itertools import product

from config import (
    BM25_K1,
    BM25_B,
    BM25_K1_GRID,
    BM25_B_GRID,
    BM25_TUNING_METRIC,
    BM25_PARAMS_PATH,
    TOP_K,
)
from retrieval import SearchEngine
from lexical_scoring import BM25Cache
from evaluation import Evaluator
from utils import setup_logger, save_json, load_json

logger = setup_logger("BM25Tuning")

QUERY_LENGTH_BUCKETS = (
    ("short", 0, 3),
    ("medium", 4, 8),
    ("long", 9, None),
)


def load_tuned_params():
    """Return (k1, b, metadata) from saved tuning file, or config defaults."""
    if BM25_PARAMS_PATH.exists():
        data = load_json(BM25_PARAMS_PATH)
        return data["k1"], data["b"], data
    return BM25_K1, BM25_B, None


def save_tuned_params(k1, b, metadata):
    payload = {
        "k1": k1,
        "b": b,
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    save_json(payload, BM25_PARAMS_PATH)
    logger.info(f"Tuned BM25 params saved to {BM25_PARAMS_PATH}")


def analyze_corpus(index, processed_queries):
    """Collect corpus and query statistics used to justify BM25 parameter choices."""
    doc_lengths = list(index.doc_lengths.values())
    query_lengths = [len(tokens) for tokens in processed_queries.values()]

    def _stats(values):
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        median = (
            sorted_vals[mid]
            if len(sorted_vals) % 2
            else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        )
        return {
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": round(sum(values) / len(values), 2),
            "median": median,
        }

    bucket_counts = {}
    for label, low, high in QUERY_LENGTH_BUCKETS:
        if high is None:
            count = sum(1 for ql in query_lengths if ql >= low)
        else:
            count = sum(1 for ql in query_lengths if low <= ql <= high)
        bucket_counts[label] = count

    return {
        "total_docs": index.total_docs,
        "unique_terms": len(index.index),
        "avg_doc_length": round(index.avg_doc_length, 2),
        "doc_length_stats": _stats(doc_lengths),
        "query_length_stats": _stats(query_lengths),
        "query_length_buckets": bucket_counts,
    }


def _bucket_queries(processed_queries):
    buckets = {label: {} for label, _, _ in QUERY_LENGTH_BUCKETS}
    for qid, tokens in processed_queries.items():
        qlen = len(tokens)
        for label, low, high in QUERY_LENGTH_BUCKETS:
            if high is None and qlen >= low:
                buckets[label][qid] = tokens
                break
            if high is not None and low <= qlen <= high:
                buckets[label][qid] = tokens
                break
    return buckets


def _metric_value(metrics, metric_name, k=TOP_K):
    if metric_name == "MAP":
        return metrics.get("MAP", 0.0)
    if metric_name == "Recall":
        return metrics.get("Recall", 0.0)
    if metric_name.startswith("nDCG"):
        return metrics.get(f"nDCG@{k}", 0.0)
    return metrics.get(metric_name, 0.0)


def grid_search(
    index,
    processed_docs,
    processed_queries,
    qrels,
    k1_grid=None,
    b_grid=None,
    primary_metric=None,
    top_k=TOP_K,
):
    """
    Evaluate all (k1, b) combinations on the tuning query set.

    Returns:
        list of dicts sorted by primary metric descending
    """
    k1_grid = k1_grid or BM25_K1_GRID
    b_grid = b_grid or BM25_B_GRID
    primary_metric = primary_metric or BM25_TUNING_METRIC
    evaluator = Evaluator(qrels)

    results = []
    combos = list(product(k1_grid, b_grid))
    logger.info(
        f"BM25 grid search: {len(combos)} combinations, "
        f"metric={primary_metric}, queries={len(processed_queries)}"
    )

    for i, (k1, b) in enumerate(combos, start=1):
        bm25_cache = BM25Cache.build(processed_docs, k1, b)
        engine = SearchEngine(index, bm25_k1=k1, bm25_b=b, bm25_cache=bm25_cache)
        start = time.time()
        retrieval = engine.retrieve_batch(processed_queries, model="bm25", top_k=top_k)
        elapsed = round(time.time() - start, 3)
        metrics = evaluator.evaluate_all(retrieval, k=top_k)

        entry = {
            "k1": k1,
            "b": b,
            "metrics": Evaluator._spec_metrics(metrics, k=top_k),
            "primary_metric": primary_metric,
            "primary_score": _metric_value(metrics, primary_metric, k=top_k),
            "retrieval_seconds": elapsed,
        }
        results.append(entry)

        if i % 5 == 0 or i == len(combos):
            logger.info(f"  [{i}/{len(combos)}] latest k1={k1}, b={b} -> {primary_metric}={entry['primary_score']:.4f}")

    results.sort(key=lambda x: x["primary_score"], reverse=True)
    return results


def analyze_query_buckets(
    index, processed_docs, processed_queries, qrels, best_k1, best_b, top_k=TOP_K
):
    """Measure tuned params per query-length bucket for query-dependent justification."""
    buckets = _bucket_queries(processed_queries)
    evaluator = Evaluator(qrels)
    bm25_cache = BM25Cache.build(processed_docs, best_k1, best_b)
    engine = SearchEngine(
        index, bm25_k1=best_k1, bm25_b=best_b, bm25_cache=bm25_cache
    )
    analysis = {}

    for label, bucket_queries in buckets.items():
        if not bucket_queries:
            analysis[label] = {"query_count": 0, "metrics": {}}
            continue
        retrieval = engine.retrieve_batch(bucket_queries, model="bm25", top_k=top_k)
        metrics = evaluator.evaluate_all(retrieval, k=top_k)
        analysis[label] = {
            "query_count": len(bucket_queries),
            "metrics": Evaluator._spec_metrics(metrics, k=top_k),
        }

    return analysis


def build_justification(corpus_stats, grid_results, best_entry, defaults=None):
    """
    Build theoretical justification for BM25 k1/b based on corpus and query profile.
    Suitable for inclusion in the project report.
    """
    defaults = defaults or {"k1": BM25_K1, "b": BM25_B}
    k1, b = best_entry["k1"], best_entry["b"]
    avg_dl = corpus_stats["avg_doc_length"]
    q_median = corpus_stats["query_length_stats"]["median"]

    reasons = []

    reasons.append(
        f"Grid search over k1 in {BM25_K1_GRID} and b in {BM25_B_GRID} "
        f"on {corpus_stats['total_docs']} documents selected "
        f"k1={k1}, b={b} by maximizing {best_entry['primary_metric']} "
        f"({best_entry['primary_score']:.4f})."
    )

    reasons.append(
        f"k1 controls term-frequency saturation. "
        f"For Quora-style short questions (median query length ≈ {q_median} tokens), "
        f"k1={k1} balances repeated-term emphasis without over-weighting spammy repetition."
    )

    if avg_dl < 50:
        b_note = (
            f"Corpus average document length is short ({avg_dl:.1f} tokens), "
            f"so a moderate-to-high b={b} reduces unfair advantage of longer duplicate pairs."
        )
    elif avg_dl > 200:
        b_note = (
            f"Corpus average document length is long ({avg_dl:.1f} tokens), "
            f"b={b} compensates for length bias so shorter relevant answers are not suppressed."
        )
    else:
        b_note = (
            f"With average document length {avg_dl:.1f} tokens, "
            f"b={b} applies standard length normalization (Robertson–Walker BM25)."
        )
    reasons.append(b_note)

    default_score = next(
        (
            r["primary_score"]
            for r in grid_results
            if r["k1"] == defaults["k1"] and r["b"] == defaults["b"]
        ),
        None,
    )
    if default_score is not None:
        delta = best_entry["primary_score"] - default_score
        reasons.append(
            f"Compared with defaults (k1={defaults['k1']}, b={defaults['b']}), "
            f"tuned params improve {best_entry['primary_metric']} by {delta:+.4f}."
        )

    reasons.append(
        "Query-length buckets (short/medium/long) are reported separately because "
        "short keyword queries favor lower length-penalty (smaller b), while longer "
        "natural-language questions tolerate higher b."
    )

    return {
        "selected_k1": k1,
        "selected_b": b,
        "selection_metric": best_entry["primary_metric"],
        "selection_score": best_entry["primary_score"],
        "default_k1": defaults["k1"],
        "default_b": defaults["b"],
        "theoretical_justification": reasons,
        "summary": (
            f"Optimal BM25 for this dataset: k1={k1}, b={b}. "
            f"{reasons[1]} {reasons[2]}"
        ),
    }


def run_tuning(
    index,
    processed_docs,
    processed_queries,
    qrels,
    k1_grid=None,
    b_grid=None,
    primary_metric=None,
    top_k=TOP_K,
):
    """
    Full tuning pipeline: corpus analysis, grid search, bucket analysis, justification.
    """
    corpus_stats = analyze_corpus(index, processed_queries)
    grid_results = grid_search(
        index,
        processed_docs,
        processed_queries,
        qrels,
        k1_grid=k1_grid,
        b_grid=b_grid,
        primary_metric=primary_metric,
        top_k=top_k,
    )
    best = grid_results[0]
    bucket_analysis = analyze_query_buckets(
        index,
        processed_docs,
        processed_queries,
        qrels,
        best["k1"],
        best["b"],
        top_k=top_k,
    )
    justification = build_justification(corpus_stats, grid_results, best)

    report = {
        "corpus_stats": corpus_stats,
        "primary_metric": primary_metric or BM25_TUNING_METRIC,
        "grid_results": grid_results,
        "best_params": {"k1": best["k1"], "b": best["b"], "metrics": best["metrics"]},
        "query_bucket_analysis": bucket_analysis,
        "justification": justification,
    }
    return report
