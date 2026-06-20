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

BASELINE_DIR.mkdir(parents=True, exist_ok=True)
DENSE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
