import math
from collections import defaultdict

from utils import setup_logger, save_json

logger = setup_logger("Evaluation")

# Metrics required by IR Project 2026 spec
REQUIRED_METRICS = ("MAP", "Recall", "Precision@10", "nDCG@10")


class Evaluator:
    """Evaluate retrieval performance against qrels."""

    def __init__(self, qrels):
        """
        qrels: {query_id: {doc_id: relevance_score}}
        """
        self.qrels = qrels

    @staticmethod
    def _dcg_at_k(relevances, k):
        """DCG@k using BEIR-style gain: (2^rel - 1) / log2(rank + 1)."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg

    def ndcg_at_k(self, query_id, retrieved_docs, k=10):
        if query_id not in self.qrels:
            return 0.0

        relevant_docs = self.qrels[query_id]
        retrieved_rels = [
            relevant_docs.get(doc_id, 0)
            for doc_id, _ in retrieved_docs[:k]
        ]

        ideal_rels = sorted(relevant_docs.values(), reverse=True)
        dcg = self._dcg_at_k(retrieved_rels, k)
        idcg = self._dcg_at_k(ideal_rels, k)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_query(self, query_id, retrieved_docs, k=10):
        if query_id not in self.qrels:
            return None

        relevant_docs = self.qrels[query_id]
        retrieved_ids = [doc_id for doc_id, _ in retrieved_docs[:k]]

        relevant_retrieved = sum(
            1 for doc_id in retrieved_ids
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0
        )
        precision_at_k = relevant_retrieved / k if k > 0 else 0

        total_relevant = sum(1 for score in relevant_docs.values() if score > 0)
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

        ap = 0.0
        relevant_seen = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                relevant_seen += 1
                ap += relevant_seen / (i + 1)

        if total_relevant > 0:
            ap /= total_relevant

        rr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                rr = 1.0 / (i + 1)
                break

        ndcg = self.ndcg_at_k(query_id, retrieved_docs, k)

        return {
            "P@K": precision_at_k,
            "Recall": recall,
            "AP": ap,
            "RR": rr,
            "nDCG": ndcg,
        }

    def evaluate_all(self, all_results, k=10):
        logger.info(f"Evaluating results (K={k})...")
        metrics_sum = defaultdict(float)
        evaluated_queries = 0

        for query_id, retrieved_docs in all_results.items():
            metrics = self.evaluate_query(query_id, retrieved_docs, k)
            if metrics:
                evaluated_queries += 1
                for key, value in metrics.items():
                    metrics_sum[key] += value

        if evaluated_queries == 0:
            logger.warning("No evaluated queries (qrels) matched the retrieved results.")
            return {}

        final_metrics = {
            f"Precision@{k}": metrics_sum["P@K"] / evaluated_queries,
            "Recall": metrics_sum["Recall"] / evaluated_queries,
            "MAP": metrics_sum["AP"] / evaluated_queries,
            "MRR": metrics_sum["RR"] / evaluated_queries,
            f"nDCG@{k}": metrics_sum["nDCG"] / evaluated_queries,
            "Evaluated_Queries": evaluated_queries,
        }

        logger.info("Evaluation completed.")
        for metric, value in final_metrics.items():
            if metric != "Evaluated_Queries":
                logger.info(f"{metric}: {value:.4f}")

        return final_metrics

    @staticmethod
    def _spec_metrics(metrics, k=10):
        """Map internal metric names to project-spec labels."""
        if not metrics:
            return {}
        return {
            "MAP": round(metrics.get("MAP", 0.0), 6),
            "Recall": round(metrics.get("Recall", 0.0), 6),
            f"Precision@{k}": round(
                metrics.get(f"Precision@{k}", metrics.get(f"Mean P@{k}", 0.0)), 6
            ),
            f"nDCG@{k}": round(metrics.get(f"nDCG@{k}", 0.0), 6),
        }

    @staticmethod
    def compare_models(models_metrics, k=10, timing=None):
        """
        Build a side-by-side comparison table for all representation methods.

        Args:
            models_metrics: {model_name: metrics_dict}
            k: cutoff for Precision@k and nDCG@k
            timing: optional {model_name: seconds}

        Returns:
            dict with per-model spec metrics, rankings, and best model per metric
        """
        comparison = {
            "metrics_k": k,
            "required_metrics": list(REQUIRED_METRICS),
            "models": {},
        }

        for model_name, metrics in models_metrics.items():
            entry = {"metrics": Evaluator._spec_metrics(metrics, k=k)}
            if timing and model_name in timing:
                entry["retrieval_seconds"] = timing[model_name]
            comparison["models"][model_name] = entry

        rankings = {}
        for metric_key in ("MAP", "Recall", f"Precision@{k}", f"nDCG@{k}"):
            scored = [
                (name, data["metrics"].get(metric_key, 0.0))
                for name, data in comparison["models"].items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            rankings[metric_key] = [
                {"model": name, "value": value} for name, value in scored
            ]

        comparison["rankings"] = rankings
        comparison["best_per_metric"] = {
            metric: rows[0]["model"] if rows else None
            for metric, rows in rankings.items()
        }

        return comparison

    @staticmethod
    def log_comparison(comparison, title="Model comparison"):
        logger.info(f"\n--- {title} ---")
        k = comparison.get("metrics_k", 10)
        header = f"{'Model':<22} {'MAP':>8} {'Recall':>8} {'P@'+str(k):>8} {'nDCG@'+str(k):>8}"
        if any("retrieval_seconds" in m for m in comparison["models"].values()):
            header += f" {'Time(s)':>10}"
        logger.info(header)
        logger.info("-" * len(header))

        for model_name, data in comparison["models"].items():
            m = data["metrics"]
            line = (
                f"{model_name:<22} "
                f"{m.get('MAP', 0):>8.4f} "
                f"{m.get('Recall', 0):>8.4f} "
                f"{m.get(f'Precision@{k}', 0):>8.4f} "
                f"{m.get(f'nDCG@{k}', 0):>8.4f}"
            )
            if "retrieval_seconds" in data:
                line += f" {data['retrieval_seconds']:>10.3f}"
            logger.info(line)

        logger.info("\nBest model per metric:")
        for metric, best in comparison.get("best_per_metric", {}).items():
            logger.info(f"  {metric}: {best}")

    @staticmethod
    def save_comparison(comparison, path):
        save_json(comparison, path)
        logger.info(f"Comparison report saved to {path}")

    def before_after_analysis(self, before_metrics, after_metrics, k=10):
        """
        Quantify impact of additional features (hybrid, refinement, etc.).
        """
        before = self._spec_metrics(before_metrics, k=k)
        after = self._spec_metrics(after_metrics, k=k)

        deltas = {}
        for key in before:
            deltas[key] = round(after.get(key, 0.0) - before.get(key, 0.0), 6)

        return {
            "before": before,
            "after": after,
            "delta": deltas,
            "improved": {k: v > 0 for k, v in deltas.items()},
        }
