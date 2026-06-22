import os
from pathlib import Path

# المسارات الأساسية للمشروع
# نفترض أن الكود يعمل من داخل مجلد code
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# مجلدات البيانات والنتائج
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# إنشاء المجلدات إذا لم تكن موجودة
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# إعدادات قاعدة البيانات (ir_datasets)
# تعيين مسار التحميل ليكون داخل مجلد المشروع بدلاً من المسار الافتراضي
os.environ['IR_DATASETS_HOME'] = str(DATA_DIR)
# cqadupstack في BEIR مقسّم إلى 12 موضوعاً فرعياً في ir_datasets.
# الاسم beir/cqadupstack وحده غير مسجّل — يجب تحديد الموضوع، مثل english.
#
# quora أيضاً مقسّمة: beir/quora (وثائق + استعلامات فقط، بدون qrels).
# للتقييم استخدم beir/quora/test (~10k استعلام + qrels) أو beir/quora/dev (~5k).
DATASET_NAME = 'beir/quora/test'

# إعدادات معالجة البيانات
BATCH_SIZE = 1000       # عدد الوثائق في كل دفعة (لتوفير الذاكرة)
MAX_DOCS = 200000      # Use None for full corpus; full baseline saved in results/baseline/run_full/

# إعدادات الفهرسة والاسترجاع
TOP_K = 10              # عدد النتائج المسترجعة لكل استعلام
INDEX_TYPE = 'bm25'     # نوع الفهرس: 'tfidf' أو 'bm25'

# إعدادات BM25
BM25_K1 = 1.5
BM25_B = 0.75
BM25_PARAMS_PATH = MODELS_DIR / "bm25_params.json"

# ضبط معاملات BM25 (Grid Search)
BM25_TUNING_DIR = RESULTS_DIR / "bm25_tuning"
BM25_K1_GRID = [0.6, 1.2, 1.5, 1.8, 2.0]
BM25_B_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BM25_TUNING_METRIC = "MAP"          # primary metric: MAP | nDCG@10 | Recall
BM25_TUNING_MAX_QUERIES = 500       # None = all qrel queries; int for faster tuning

# إعدادات التقييم والخط الأساسي (baseline)
MAX_EVAL_QUERIES = None          # None = all qrel queries; int for dev runs (e.g. 100)
EVAL_RANDOM_SEED = 42            # used only when MAX_EVAL_QUERIES is set
BASELINE_DIR = RESULTS_DIR / "baseline"
DENSE_DIR = RESULTS_DIR / "dense"
SAVE_RETRIEVAL_RESULTS = False   # skip large JSON on full-query runs; metrics only

# إعدادات التضمين (Embeddings) والفهرس المتجهي
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = BATCH_SIZE
NORMALIZE_EMBEDDINGS = True
REBUILD_VECTOR_INDEX = False
VECTOR_INDEX_PATH = MODELS_DIR / "vector_index.faiss"
VECTOR_DOC_IDS_PATH = MODELS_DIR / "vector_doc_ids.json"
EMBEDDING_METADATA_PATH = MODELS_DIR / "embedding_metadata.json"

# إعدادات الاسترجاع الهجين (Hybrid)
HYBRID_SERIAL_CANDIDATES = 100   # عدد مرشحي BM25 قبل إعادة الترتيب بالتضمين
HYBRID_PARALLEL_DEPTH = 100      # عمق كل مسار قبل الدمج في الوضع المتوازي
RRF_K = 60                       # ثابت Reciprocal Rank Fusion
HYBRID_FUSION_METHOD = "rrf"     # "rrf" أو "weighted"
HYBRID_BM25_WEIGHT = 0.5         # وزن BM25 عند استخدام weighted fusion

# إعدادات تحسين الاستعلام (Query Refinement)
REFINEMENT_DIR = RESULTS_DIR / "refinement"
PRF_TOP_DOCS = 5                 # وثائق PRF الأولية
PRF_EXPAND_TERMS = 8             # عدد المصطلحات المضافة من PRF
SPELL_MIN_TOKEN_LEN = 4          # أقل طول لتصحيح الإملاء
SPELL_CORRECTION_CUTOFF = 0.82   # عتبة التشابه لاقتراح التصحيح
HISTORY_MAX_SUGGESTIONS = 3      # اقتراحات من سجل البحث
HISTORY_MIN_OVERLAP = 1          # أقل تداخل رموز مع استعلام سابق
HISTORY_WINDOW = 50                # عدد الاستعلامات الأخيرة في السجل (للسرعة)

# إعدادات واجهة API (SOA Gateway)
API_HOST = "127.0.0.1"
API_PORT = 8000

BASELINE_DIR.mkdir(parents=True, exist_ok=True)
DENSE_DIR.mkdir(parents=True, exist_ok=True)
BM25_TUNING_DIR.mkdir(parents=True, exist_ok=True)
REFINEMENT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
