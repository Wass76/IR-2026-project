import math
from collections import defaultdict
from utils import setup_logger

logger = setup_logger("Indexing")

class InvertedIndex:
    """فئة لبناء الفهرس المعكوس (Inverted Index)"""
    
    def __init__(self):
        # index: {term: {doc_id: term_frequency}}
        # Use a picklable default factory (dict) instead of lambda.
        self.index = defaultdict(dict)
        # doc_lengths: {doc_id: length_of_doc}
        self.doc_lengths = {}
        # df: {term: document_frequency}
        self.df = defaultdict(int)
        # إجمالي عدد الوثائق
        self.total_docs = 0
        # متوسط طول الوثيقة
        self.avg_doc_length = 0
        
    def build(self, processed_docs):
        """
        بناء الفهرس من مجموعة وثائق معالجة
        processed_docs: {doc_id: [tokens]}
        """
        logger.info("Building inverted index...")
        self.total_docs = len(processed_docs)
        total_length = 0
        
        for doc_id, tokens in processed_docs.items():
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # حساب تكرار المصطلحات في الوثيقة
            term_counts = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
                
            # تحديث الفهرس و Document Frequency
            for term, count in term_counts.items():
                self.index[term][doc_id] = count
                self.df[term] += 1
                
        if self.total_docs > 0:
            self.avg_doc_length = total_length / self.total_docs
            
        logger.info(f"Inverted index built successfully. Unique terms: {len(self.index)}")
        
    def get_term_postings(self, term):
        """الحصول على قائمة الوثائق التي تحتوي على مصطلح معين"""
        return self.index.get(term, {})
        
    def get_df(self, term):
        """الحصول على Document Frequency لمصطلح معين"""
        return self.df.get(term, 0)
        
    def get_idf(self, term):
        """حساب Inverse Document Frequency لمصطلح معين"""
        df = self.get_df(term)
        if df == 0:
            return 0
        # استخدام صيغة IDF القياسية: log(N / df)
        return math.log10(self.total_docs / df)

if __name__ == "__main__":
    # اختبار سريع
    docs = {
        "d1": ["python", "is", "great"],
        "d2": ["python", "is", "fast", "and", "python", "is", "easy"],
        "d3": ["java", "is", "also", "good"]
    }
    index = InvertedIndex()
    index.build(docs)
    print(f"Index for 'python': {index.get_term_postings('python')}")
    print(f"IDF for 'python': {index.get_idf('python')}")
