import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from utils import setup_logger

logger = setup_logger("Preprocessing")

# تحميل أدوات NLTK المطلوبة
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """فئة لمعالجة وتنظيف النصوص"""
    
    def __init__(self, use_stemming=True, use_lemmatization=False):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
    def clean_text(self, text):
        """تنظيف النص الأساسي"""
        if not isinstance(text, str):
            return ""
            
        # تحويل الحروف إلى صغيرة
        text = text.lower()
        
        # إزالة علامات الترقيم والأرقام والرموز الخاصة
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def process_text(self, text):
        """المعالجة الكاملة للنص"""
        # 1. التنظيف
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return []
            
        # 2. التقطيع (Tokenization)
        tokens = word_tokenize(cleaned_text)
        
        # 3. إزالة كلمات التوقف (Stop words)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # 4. التجذير (Stemming) أو (Lemmatization)
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        return tokens
        
    def process_collection(self, collection):
        """
        معالجة مجموعة كاملة من النصوص (وثائق أو استعلامات)
        collection: قاموس {id: text}
        يعيد: قاموس {id: [tokens]}
        """
        logger.info(f"Processing {len(collection)} texts...")
        processed_collection = {}
        
        for i, (doc_id, text) in enumerate(collection.items()):
            processed_collection[doc_id] = self.process_text(text)
            
            if (i + 1) % 5000 == 0:
                logger.info(f"Processed {i + 1} texts...")
                
        logger.info("Preprocessing completed successfully.")
        return processed_collection

if __name__ == "__main__":
    # اختبار سريع
    preprocessor = TextPreprocessor()
    sample_text = "Python is a high-level programming language known for its simplicity!"
    print(f"Original: {sample_text}")
    print(f"Processed: {preprocessor.process_text(sample_text)}")
