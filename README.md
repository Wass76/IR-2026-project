# IR Project 2026 — Information Retrieval System

**مشروع استرجاع المعلومات 2026**  
نظام بحث يدعم تمثيلات متعددة للوثائق والاستعلامات، مع تقييم كامل وواجهة ويب ومعمارية SOA.

---

## Overview | نظرة عامة

This project implements a complete **Information Retrieval (IR)** pipeline in Python, aligned with the **IR Project 2026** course requirements:

- Data loading and preprocessing
- **TF-IDF**, **BM25**, **Dense Embeddings**, **Hybrid (Serial & Parallel)**
- BM25 parameter tuning with justification
- Query refinement (spelling, PRF, history-based suggestions)
- Evaluation: **MAP**, **Recall**, **Precision@10**, **nDCG@10**
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
│   ├── data_loader.py             # Load dataset, queries, qrels
│   ├── preprocessing.py           # Preprocessing Service
│   ├── indexing.py                # Inverted index (lexical)
│   ├── vector_index.py            # FAISS vector index
│   ├── embeddings.py              # Sentence-transformers encoder
│   ├── retrieval.py               # TF-IDF, BM25, Dense, Hybrid retrieval
│   ├── query_refinement.py        # Query Refinement Service
│   ├── evaluation.py              # Evaluation metrics & comparison
│   ├── bm25_tuning.py             # BM25 grid search & justification
│   ├── api/app.py                 # FastAPI gateway (SOA)
│   └── services/ir_system.py      # Service orchestrator for API/UI
├── frontend/
│   └── index.html                 # Web search UI
├── docs/
│   └── Architecture_Diagram.md    # SOA architecture diagrams (Mermaid)
├── data/                          # Downloaded dataset (auto-created)
├── models/                        # Cached indexes & tuned BM25 params
├── results/                       # Evaluation JSON outputs
├── logs/                          # Run logs
└── requirements.txt
```

---

## Requirements | المتطلبات

- **Python 3.10+** (tested on 3.13)
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
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Web API |

Tuned BM25 parameters are saved to `models/bm25_params.json` after tuning.

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
| http://127.0.0.1:8000/health | System status |

### UI workflow

1. Click **Load Index** (wait ~1–2 min for 200K docs if cache exists)
2. Enter a query, e.g. `What causes a nightmare?`
3. Select model: **BM25** (recommended)
4. Click **Search**
5. Optionally check **Use query refinement** to compare expanded queries

### API example

```bash
POST http://127.0.0.1:8000/services/search
Content-Type: application/json

{
  "query": "What causes a nightmare?",
  "model": "bm25",
  "top_k": 10,
  "use_refinement": false
}
```

### SOA Services

| Service | REST Endpoint |
|---------|---------------|
| Indexing | `POST /services/index/load`, `GET /services/index/status` |
| Preprocessing | `POST /services/preprocess` |
| Query Refinement | `POST /services/refine` |
| Retrieval | `POST /services/search` |
| Evaluation | `GET /services/evaluation/summary` |
| BM25 params | `GET /services/bm25/params` |

Architecture diagrams: [`docs/Architecture_Diagram.md`](docs/Architecture_Diagram.md)

---

## Retrieval Models | نماذج الاسترجاع

| Model | Description |
|-------|-------------|
| **TF-IDF** | Vector space model with TF-IDF weighting |
| **BM25** | Probabilistic lexical retrieval (tuned k1, b) |
| **Embedding** | Dense retrieval via sentence-transformers + FAISS |
| **Hybrid Serial** | BM25 candidates → dense re-ranking |
| **Hybrid Parallel** | BM25 + dense in parallel → RRF fusion |

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

---

## Query Refinement | تحسين الاستعلام

When enabled, the pipeline applies:

1. **Spelling correction** — map OOV tokens to index vocabulary
2. **Pseudo-Relevance Feedback (PRF)** — expand query from top BM25 documents
3. **History-based suggestions** — terms from similar past queries in the session

Use **Preview Refinement** in the UI to inspect changes without searching.

---

## Team | فريق العمل

| Member | Role |
|--------|------|
| _Name_ | _Responsibility_ |
| _Name_ | _Responsibility_ |

> Update this table with your team division document.

---

## Deliverables Checklist | قائمة التسليمات

- [x] Working IR pipeline (Python)
- [x] Single dataset with qrels (Quora)
- [x] TF-IDF, BM25, Embeddings, Hybrid
- [x] BM25 parameter tuning
- [x] Query refinement + before/after evaluation
- [x] SOA + REST API + Web UI
- [x] Architecture diagram (`docs/Architecture_Diagram.md`)
- [x] README (this file)
- [ ] Arabic technical report
- [ ] Demo videos
- [ ] Team work division document

---

## References | مراجع

- [ir-datasets](https://ir-datasets.com/)
- [BEIR Quora](https://github.com/beir-cellar/beir)
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*
- Cormack, G. et al. (2009). *Reciprocal Rank Fusion*

---

## License

Academic project — IR Course 2026.
