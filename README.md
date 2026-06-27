# IR Project 2026 — Information Retrieval System

**مشروع استرجاع المعلومات 2026**  
نظام بحث يدعم تمثيلات متعددة للوثائق والاستعلامات، مع تقييم كامل وواجهة ويب ومعمارية SOA.

---

## Overview | نظرة عامة

This project implements a complete **Information Retrieval (IR)** pipeline in Python, aligned with the **IR Project 2026** course requirements:

- Data loading and preprocessing
- **TF-IDF** (scikit-learn), **BM25** ([rank-bm25](https://github.com/dorianbrown/rank_bm25)), **Dense Embeddings**, **Hybrid (Serial & Parallel)**
- BM25 parameter tuning with justification
- Query refinement (spelling, PRF, history-based suggestions)
- Evaluation: **MAP**, **Recall**, **Precision@10**, **nDCG@10**
- **SQLite document persistence** (original + processed text)
- **Disk caching** for fast API startup (documents, indexes, BM25, TF-IDF, FAISS)
- **SOA** architecture with **REST API** and **Web UI**

**Dataset:** [beir/quora/test](https://ir-datasets.com/) via `ir_datasets` (~523K documents, 10K test queries with qrels).  
**Corpus used in experiments:** 200,000 documents (configurable in `code/config.py`).

---

## Project Structure | بنية المشروع

```
IR-2026-project/
├── code/                          # Python source code
│   ├── main.py                    # CLI entry point (all pipeline modes + API)
│   ├── config.py                  # Global configuration
│   ├── data_loader.py             # Load dataset, queries, qrels, corpus helpers
│   ├── document_store.py          # SQLite persistence (original + processed docs)
│   ├── preprocessing.py           # Preprocessing Service
│   ├── indexing.py                # Inverted index (lexical, for PRF/refinement)
│   ├── index_cache.py             # Disk cache: inverted index, BM25, TF-IDF
│   ├── lexical_scoring.py         # rank_bm25 + sklearn TF-IDF wrappers
│   ├── vector_index.py            # FAISS vector index
│   ├── embeddings.py              # Sentence-transformers encoder
│   ├── retrieval.py               # Search engines (lexical, dense, hybrid)
│   ├── query_refinement.py        # Query Refinement Service
│   ├── evaluation.py              # Evaluation metrics & comparison
│   ├── bm25_tuning.py             # BM25 grid search & justification
│   ├── api/app.py                 # FastAPI gateway (SOA)
│   └── services/ir_system.py      # Service orchestrator for API/UI
├── frontend/
│   └── index.html                 # Web search UI (English)
├── docs/
│   └── Architecture_Diagram.md    # SOA architecture diagrams (Mermaid)
├── data/                          # Dataset download + documents.db
├── models/                        # Cached indexes, BM25/TF-IDF, FAISS, tuned params
├── results/                       # Evaluation JSON outputs
├── logs/                          # Run logs
└── requirements.txt
```

---

## Requirements | المتطلبات

- **Python 3.10+** (tested on 3.13 / 3.14)
- Windows / Linux / macOS
- ~8 GB RAM recommended for 200K-document runs
- Internet on first run (dataset + embedding model download)

---

## Installation | التثبيت

```powershell
# Clone the repository
git clone <your-repo-url>
cd IR-2026-project

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> **Windows note:** If `python` is not on PATH, use the full path to your Python executable, e.g.  
> `C:\Users\<user>\anaconda3\python.exe`

---

## Configuration | الإعدادات

Edit `code/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATASET_NAME` | `beir/quora/test` | ir-datasets identifier |
| `MAX_DOCS` | `200000` | Document cap (`None` = full corpus) |
| `MAX_EVAL_QUERIES` | `None` | Eval query cap (`None` = all 10K) |
| `TOP_K` | `10` | Results per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Dense embedding model |
| `DOCUMENTS_DB_PATH` | `data/documents.db` | SQLite document store |
| `REBUILD_LEXICAL_INDEX` | `False` | Force rebuild inverted/BM25/TF-IDF caches |
| `REBUILD_VECTOR_INDEX` | `False` | Force rebuild FAISS embeddings |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Web API |

Tuned BM25 parameters are saved to `models/bm25_params.json` after tuning.

### Cached artifacts (`models/`)

| File | Contents |
|------|----------|
| `documents.db` (in `data/`) | Original document text + processed tokens |
| `inverted_index.pkl` | Inverted index (PRF / refinement) |
| `bm25_cache.pkl` | `rank_bm25.BM25Okapi` model |
| `tfidf_cache.pkl` | `TfidfVectorizer` + sparse matrix |
| `lexical_index_metadata.json` | Corpus fingerprint for cache validation |
| `vector_index.faiss` | Dense vector index |
| `embedding_metadata.json` | FAISS cache metadata |
| `bm25_params.json` | Tuned k1, b |

Caches invalidate automatically when the corpus, preprocessing settings, or BM25 parameters change.

---

## How to Run | طريقة التشغيل

All commands are run from the `code/` directory:

```powershell
cd code
python main.py --mode <mode> [options]
```

### CLI Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Baseline** | `python main.py --mode baseline` | TF-IDF + BM25 evaluation (before extra features) |
| **Dense** | `python main.py --mode dense` | Embedding retrieval only |
| **Hybrid** | `python main.py --mode hybrid` | Dense + Hybrid Serial + Hybrid Parallel |
| **Full** | `python main.py --mode full` | Baseline then dense+hybrid |
| **BM25 tuning** | `python main.py --mode tune-bm25` | Grid search for k1, b |
| **Refinement** | `python main.py --mode refinement` | Before/after query refinement eval |
| **API + UI** | `python main.py --mode api` | Start web server |

### Common Options

```powershell
python main.py --mode hybrid --max-docs 5000 --max-queries 100
python main.py --mode hybrid --hybrid-type both    # serial | parallel | both | none
```

### Examples

```powershell
# Quick dev test (5K docs, 100 queries)
python main.py --mode hybrid --max-docs 5000 --max-queries 100

# Full BM25 tuning on 200K docs
python main.py --mode tune-bm25 --max-docs 200000 --max-queries 500

# Query refinement before/after evaluation
python main.py --mode refinement --max-docs 200000

# Start API + Web UI
python main.py --mode api
```

---

## Web UI & REST API | الواجهة وواجهة البرمجة

### Start the server

```powershell
cd code
python main.py --mode api
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/ | Web search UI |
| http://127.0.0.1:8000/docs | Swagger API documentation |
| http://127.0.0.1:8000/health | System status + cache flags |

### UI workflow

1. Click **Load Index** (first run builds caches; later runs load from disk/SQLite)
2. Enter a query, e.g. `What causes a nightmare?`
3. Select model: **BM25**, **TF-IDF**, **Dense**, **Hybrid Serial**, or **Hybrid Parallel**
4. For **Hybrid Parallel**, adjust **BM25 Weight** / **Dense Weight** (weighted fusion)
5. Click **Search** — results show **Doc ID**, score, and **full original document text**
6. Optionally check **Use query refinement** or click **Preview Refinement**

### API examples

**Search**

```bash
POST http://127.0.0.1:8000/services/search
Content-Type: application/json

{
  "query": "What causes a nightmare?",
  "model": "hybrid_parallel",
  "top_k": 10,
  "use_refinement": false,
  "bm25_weight": 0.6,
  "dense_weight": 0.4
}
```

**Get original document by ID**

```bash
GET http://127.0.0.1:8000/services/documents/{doc_id}
```

Returns `doc_id`, `original_content`, `processed_tokens` (if cached), and `metadata`.

### SOA Services

| Service | REST Endpoint |
|---------|---------------|
| Indexing | `POST /services/index/load`, `GET /services/index/status` |
| Documents | `GET /services/documents/{doc_id}` |
| Preprocessing | `POST /services/preprocess` |
| Query Refinement | `POST /services/refine` |
| Retrieval | `POST /services/search` |
| Evaluation | `GET /services/evaluation/summary` |
| BM25 params | `GET /services/bm25/params` |
| Hybrid settings | `GET /services/hybrid/settings` |

Architecture diagrams: [`docs/Architecture_Diagram.md`](docs/Architecture_Diagram.md)

---

## Document Persistence | تخزين الوثائق

Original and processed documents are stored in **SQLite** (`data/documents.db`):

| Column | Description |
|--------|-------------|
| `doc_id` | Document identifier |
| `original_content` | Raw text from the corpus |
| `processed_tokens` | JSON token list after preprocessing |
| `metadata` | Dataset / corpus metadata |

- **`get_document(doc_id)`** — returns full original text (+ processed tokens when available)
- Search results include `full_content` loaded from the document store
- On startup, documents and processed tokens load from SQLite when the corpus matches (no re-read from ir_datasets, no re-preprocessing)

---

## Startup & Caching | التخزين المؤقت وسرعة التشغيل

Optimized API startup sequence:

1. Load cached **documents** + **processed tokens** from SQLite
2. Load cached **BM25** (`rank_bm25`) from `models/bm25_cache.pkl`
3. Load cached **TF-IDF** (scikit-learn) from `models/tfidf_cache.pkl`
4. Load cached **inverted index** from `models/inverted_index.pkl`
5. Load **embedding model** + cached **FAISS** index

The `/health` endpoint reports:

```json
{
  "documents_cached": true,
  "inverted_index_cached": true,
  "bm25_cached": true,
  "tfidf_cached": true,
  "vector_index_cached": true
}
```

Log messages on cache hit:

```
Processed corpus cache hit — loading N docs from SQLite...
Inverted index cache hit — loaded in X.XXs
BM25 cache hit — loaded in X.XXs
TF-IDF cache hit — loaded in X.XXs
Vector index cache hit
```

Set `REBUILD_LEXICAL_INDEX = True` or delete files under `models/` to force a full rebuild.

---

## Retrieval Models | نماذج الاسترجاع

| Model | Library / Method | Description |
|-------|------------------|-------------|
| **TF-IDF** | scikit-learn `TfidfVectorizer` | L2-normalized vectors, cosine similarity |
| **BM25** | `rank_bm25.BM25Okapi` | Probabilistic lexical retrieval (tuned k1, b) |
| **Embedding** | sentence-transformers + FAISS | Dense semantic retrieval |
| **Hybrid Serial** | BM25 → dense re-rank | Top-100 BM25 candidates re-ranked by embeddings |
| **Hybrid Parallel** | BM25 + dense → fusion | RRF (default) or **weighted fusion** via UI weights |

---

## Evaluation | التقييم

### Metrics (per project spec)

- **MAP** (Mean Average Precision)
- **Recall**
- **Precision@10**
- **nDCG@10**

### Output locations

| Run | Output folder |
|-----|---------------|
| Lexical baseline | `results/baseline/run_<docs>/` |
| Dense + hybrid | `results/dense/run_<docs>/full_comparison.json` |
| BM25 tuning | `results/bm25_tuning/run_<docs>/` |
| Query refinement | `results/refinement/run_<docs>/before_after_analysis.json` |

### Sample results (200K docs, 10K queries)

| Stage | MAP | nDCG@10 |
|-------|-----|---------|
| BM25 (before refinement) | 0.319 | 0.346 |
| BM25 (after refinement) | 0.162 | 0.209 |

> Query refinement is implemented and evaluated; on Quora it reduced MAP due to noisy PRF expansion — documented in `results/refinement/`.  
> Note: switching to `rank_bm25` / sklearn may slightly change lexical scores vs earlier manual implementations.

---

## Query Refinement | تحسين الاستعلام

When enabled, the pipeline applies:

1. **Spelling correction** — map OOV tokens to index vocabulary
2. **Pseudo-Relevance Feedback (PRF)** — expand query from top BM25 documents
3. **History-based suggestions** — terms from similar past queries in the session

Use **Preview Refinement** in the UI to inspect changes without searching.

---

## References | مراجع

- [ir-datasets](https://ir-datasets.com/)
- [BEIR Quora](https://github.com/beir-cellar/beir)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*
- Cormack, G. et al. (2009). *Reciprocal Rank Fusion*

---

## License

Academic project — IR Course 2026.
