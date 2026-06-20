import math
from collections import defaultdict
from config import TOP_K, BM25_K1, BM25_B
from utils import setup_logger

logger = setup_logger("Retrieval")

class SearchEngine:
    """فئة لاسترجاع الوثائق باستخدام نماذج مختلفة"""
    
    def __init__(self, index):
        self.index = index
        
    def score_tfidf(self, query_tokens):
        """
        حساب نقاط الوثائق باستخدام نموذج TF-IDF
        يعيد: قاموس {doc_id: score}
        """
        scores = defaultdict(float)
        
        # حساب تكرار المصطلحات في الاستعلام (Query TF)
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1
            
        for term, q_tf in query_tf.items():
            postings = self.index.get_term_postings(term)
            if not postings:
                continue
                
            idf = self.index.get_idf(term)
            
            # تحديث نقاط كل وثيقة تحتوي على المصطلح
            for doc_id, doc_tf in postings.items():
                # حساب TF-IDF للوثيقة
                doc_tfidf = (1 + math.log10(doc_tf)) * idf
                # حساب TF-IDF للاستعلام
                query_tfidf = (1 + math.log10(q_tf)) * idf
                
                # إضافة حاصل الضرب إلى النتيجة النهائية للوثيقة
                scores[doc_id] += doc_tfidf * query_tfidf
                
        return scores
        
    def score_bm25(self, query_tokens):
        """
        حساب نقاط الوثائق باستخدام نموذج BM25 (أفضل من TF-IDF عادةً)
        يعيد: قاموس {doc_id: score}
        """
        scores = defaultdict(float)
        k1 = BM25_K1
        b = BM25_B
        avg_dl = self.index.avg_doc_length
        N = self.index.total_docs
        
        for term in query_tokens:
            postings = self.index.get_term_postings(term)
            if not postings:
                continue
                
            df = self.index.get_df(term)
            # صيغة IDF الخاصة بـ BM25
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            for doc_id, tf in postings.items():
                dl = self.index.doc_lengths[doc_id]
                
                # حساب BM25 term
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
                
                scores[doc_id] += idf * (numerator / denominator)
                
        return scores
        
    def retrieve(self, query_tokens, model='bm25', top_k=TOP_K):
        """
        استرجاع أفضل الوثائق لاستعلام معين
        """
        if not query_tokens:
            return []
            
        if model.lower() == 'tfidf':
            scores = self.score_tfidf(query_tokens)
        elif model.lower() == 'bm25':
            scores = self.score_bm25(query_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        # ترتيب النتائج تنازلياً حسب النقاط
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # إرجاع أفضل K نتيجة فقط
        return ranked_results[:top_k]
        
    def retrieve_batch(self, processed_queries, model='bm25', top_k=TOP_K):
        """
        استرجاع النتائج لمجموعة من الاستعلامات
        يعيد: قاموس {query_id: [(doc_id, score), ...]}
        """
        logger.info(f"Retrieving results for {len(processed_queries)} queries using {model}...")
        results = {}
        
        for i, (query_id, tokens) in enumerate(processed_queries.items()):
            results[query_id] = self.retrieve(tokens, model, top_k)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Retrieved {i + 1} queries...")
                
        logger.info("Retrieval completed successfully.")
        return results


class DenseSearchEngine:
    """Dense retrieval using embedding cosine similarity via FAISS vector index."""

    def __init__(self, vector_index, embedding_model):
        self.vector_index = vector_index
        self.embedding_model = embedding_model

    def retrieve(self, query_text, top_k=TOP_K):
        """Retrieve top-K documents for a single raw query string."""
        prepared = self.embedding_model.prepare_text(query_text)
        if not prepared:
            return []

        query_vectors = self.embedding_model.encode_queries({"_": prepared})
        query_vector = query_vectors["_"]
        return self.vector_index.search(query_vector, top_k=top_k)

    def retrieve_batch(self, raw_queries, top_k=TOP_K):
        """
        Retrieve results for a batch of raw queries.

        Args:
            raw_queries: dict {query_id: raw_text}
            top_k: number of results per query

        Returns:
            dict {query_id: [(doc_id, score), ...]}
        """
        logger.info(
            f"Dense retrieval for {len(raw_queries)} queries (top_k={top_k})..."
        )
        query_vectors = self.embedding_model.encode_queries(raw_queries)
        results = {}

        for i, (query_id, query_vector) in enumerate(query_vectors.items()):
            results[query_id] = self.vector_index.search(query_vector, top_k=top_k)
            if (i + 1) % 100 == 0:
                logger.info(f"Retrieved {i + 1} queries...")

        logger.info("Dense retrieval completed successfully.")
        return results
