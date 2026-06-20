import math
from collections import defaultdict
from utils import setup_logger

logger = setup_logger("Evaluation")

class Evaluator:
    """فئة لتقييم أداء نظام الاسترجاع"""
    
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
        """
        nDCG@k: normalized discounted cumulative gain.
        Uses graded relevance from qrels; IDCG from ideal ranking of all qrel scores.
        """
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
        """
        تقييم نتائج استعلام واحد
        retrieved_docs: [(doc_id, score), ...]
        """
        if query_id not in self.qrels:
            return None
            
        relevant_docs = self.qrels[query_id]
        retrieved_ids = [doc_id for doc_id, _ in retrieved_docs[:k]]
        
        # 1. حساب Precision @ K
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
        precision_at_k = relevant_retrieved / k if k > 0 else 0
        
        # 2. حساب Recall
        total_relevant = sum(1 for score in relevant_docs.values() if score > 0)
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        # 3. حساب Average Precision (لـ MAP) — standard: divide by total_relevant
        ap = 0.0
        relevant_seen = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                relevant_seen += 1
                ap += relevant_seen / (i + 1)
                
        if total_relevant > 0:
            ap /= total_relevant
            
        # 4. حساب Reciprocal Rank (لـ MRR)
        rr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                rr = 1.0 / (i + 1)
                break

        # 5. nDCG@k
        ndcg = self.ndcg_at_k(query_id, retrieved_docs, k)
                
        return {
            'P@K': precision_at_k,
            'Recall': recall,
            'AP': ap,
            'RR': rr,
            'nDCG': ndcg,
        }
        
    def evaluate_all(self, all_results, k=10):
        """
        تقييم جميع النتائج
        all_results: {query_id: [(doc_id, score), ...]}
        """
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
            
        # حساب المتوسطات
        final_metrics = {
            f'Mean P@{k}': metrics_sum['P@K'] / evaluated_queries,
            'Mean Recall': metrics_sum['Recall'] / evaluated_queries,
            'MAP': metrics_sum['AP'] / evaluated_queries,
            'MRR': metrics_sum['RR'] / evaluated_queries,
            f'nDCG@{k}': metrics_sum['nDCG'] / evaluated_queries,
            'Evaluated_Queries': evaluated_queries
        }
        
        logger.info("Evaluation completed.")
        for metric, value in final_metrics.items():
            if metric != 'Evaluated_Queries':
                logger.info(f"{metric}: {value:.4f}")
                
        return final_metrics
