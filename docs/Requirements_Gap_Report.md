# IR Project 2026 — Requirements Gap Report

**Project:** Information Retrieval System  
**Date:** June 20, 2026 (updated)  
**Scope note:** One dataset is sufficient (requirement updated from the original two-dataset specification).

This document lists what is **done**, **partially done**, and **still missing** compared to the project specification (`IR Project 2026.pdf`).

---

## Executive Summary

| Area | Status | Completion |
|------|--------|------------|
| Core IR pipeline (load → preprocess → index → retrieve) | Done | ~90% |
| Dataset (single dataset with qrels) | Done | ~90% |
| Required representation models | Partial | ~65% |
| Evaluation metrics | Partial | ~85% |
| SOA architecture | Not started | ~5% |
| User interface | Not started | 0% |
| Deliverables (report, videos, README) | Not started | ~5% |
| **Overall project compliance** | **Working prototype (lexical + dense)** | **~45–50%** |

The codebase is a working **monolithic IR prototype** with a locked **before-additional-features lexical baseline** (TF-IDF + BM25, 10K queries) and a new **dense embedding path** (sentence-transformers + FAISS, evaluated on 200K docs). The specification still requires **hybrid retrieval**, **query refinement**, **SOA services**, a **web UI**, and **formal deliverables**.

**Recent progress (June 20, 2026):** Added `embeddings.py`, `vector_index.py`, `DenseSearchEngine`, `--mode dense` pipeline, FAISS index caching, and 200K-document embedding evaluation with side-by-side lexical comparison.

---

## 1. Dataset Requirements

### Specification (updated)

- One dataset from [ir-datasets.com](https://ir-datasets.com)
- More than 200K documents (recommended)
- Must include test queries and qrels
- Must **not** use the Antique dataset

### Current status: **Done**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Single dataset from ir-datasets | Done | `beir/quora/test` in `config.py` |
| Queries + qrels | Done | 10,000 queries with qrels loaded |
| > 200K documents | Done | Full Quora corpus indexed (~523K docs) for lexical baseline; dense run at 200K |
| No Antique dataset | Done | Not used |
| Full corpus indexing (lexical) | Done | Baseline run at `results/baseline/run_full/` |

**Action items:**
- [x] Index the full Quora corpus (`MAX_DOCS = None`) for lexical baseline
- [ ] Run dense embedding evaluation on full corpus (`MAX_DOCS = None`)
- [ ] Document dataset choice and statistics in the final Arabic report

---

## 2. Data Pre-Processing

### Specification

Apply preprocessing after loading: stemming, lemmatization, normalization, etc., suited to the dataset.

### Current status: **Partial**

| Feature | Status | Location |
|---------|--------|----------|
| Lowercasing | Done | `preprocessing.py` → `clean_text()` |
| Remove punctuation, numbers, symbols | Done | `preprocessing.py` |
| Whitespace normalization | Done | `preprocessing.py` |
| Tokenization (NLTK) | Done | `preprocessing.py` |
| Stop word removal | Done | `preprocessing.py` |
| Stemming (Porter) | Done | Enabled in `main.py` (lexical path) |
| Lemmatization | Partial | Implemented but disabled (`use_lemmatization=False`) |
| Dataset-specific tuning | Missing | Same pipeline for all text |
| Spelling correction | Missing | Not implemented |
| Dense-path text prep | Done | Minimal whitespace normalization in `embeddings.py` (no stemming/stopwords) |

**Action items:**
- [ ] Run experiments with lemmatization vs stemming and document the impact
- [ ] Describe preprocessing choices (lexical vs dense) in the Arabic report

---

## 3. Document Representation

### Specification

Index the dataset using **all** of the following methods:

1. VSM / TF-IDF  
2. Embedding (Word2Vec, BERT, etc.)  
3. BM25  
4. Hybrid representation:
   - **Serial** hybrid  
   - **Parallel** hybrid with **fusion methods** at the scoring level  

Additional conditions:
- Apply hybrid **twice** (serial and parallel) and choose the best approach for user search
- For parallel hybrid, use fusion methods (e.g. RRF, weighted score combination)
- For BM25, provide **theoretical justification** for parameter tuning (in report or UI)

### Current status: **Partial (3 of 4 methods; hybrid missing)**

| Method | Status | Location |
|--------|--------|----------|
| TF-IDF | Done | `retrieval.py` → `score_tfidf()` |
| BM25 | Done | `retrieval.py` → `score_bm25()` |
| Embedding (BERT via sentence-transformers) | **Done** | `embeddings.py`, `vector_index.py`, `DenseSearchEngine` in `retrieval.py` |
| Hybrid — Serial | **Missing** | No module |
| Hybrid — Parallel + Fusion | **Missing** | No module |
| BM25 parameter tuning | **Missing** | `BM25_K1`, `BM25_B` are fixed defaults only |
| BM25 tuning justification (report) | **Missing** | No documentation |

**Embedding implementation details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- Index: FAISS `IndexFlatIP` (cosine similarity via inner product)
- Pipeline: `python code/main.py --mode dense`
- Cached artifacts: `models/vector_index.faiss`, `models/vector_doc_ids.json`, `models/embedding_metadata.json`

**Action items:**
- [x] Add embedding module (`embeddings.py` with sentence-transformers)
- [x] Build a vector index for dense retrieval (`vector_index.py` with FAISS)
- [x] Add dense retrieval and evaluation pipeline (`run_dense_eval()` in `main.py`)
- [ ] Implement serial hybrid (e.g. lexical filter → re-rank with embeddings)
- [ ] Implement parallel hybrid with score-level fusion (RRF or weighted sum)
- [ ] Compare serial vs parallel hybrid in evaluation
- [ ] Tune BM25 `k1` and `b` and justify choices in the Arabic report
- [ ] Run dense evaluation on full corpus (currently 200K docs in `config.py`)

---

## 4. Indexing

### Specification

Build one or more indexes suitable for the dataset (e.g. inverted index) with good efficiency.

### Current status: **Partial (lexical + vector indexes built)**

| Feature | Status | Location |
|---------|--------|----------|
| Inverted index (`term → {doc_id: tf}`) | Done | `indexing.py` |
| Document frequency (DF) | Done | `indexing.py` |
| Document lengths + average length | Done | `indexing.py` (for BM25) |
| Inverted index persistence (pickle) | Done | `main.py`, `utils.py` |
| Load saved inverted index on re-run | Missing | Baseline always rebuilds from scratch |
| Vector / embedding index (FAISS) | **Done** | `vector_index.py` |
| Vector index persistence + metadata | **Done** | `models/vector_index.faiss`, `embedding_metadata.json` |
| Load cached vector index | **Done** | `_build_or_load_vector_index()` in `main.py` |
| `BATCH_SIZE` config usage | Partial | Used as `EMBEDDING_BATCH_SIZE`; not used for lexical indexing |

**Action items:**
- [x] Add vector index for embedding-based retrieval
- [x] Persist and reload vector index with metadata-based cache invalidation
- [ ] Optionally load existing pickle inverted index instead of rebuilding every baseline run
- [ ] Document index design and efficiency in the report

---

## 5. Query Processing

### Specification

Process queries with the same preprocessing and representation methods as documents.

### Current status: **Done (lexical); Done with intentional difference (dense)**

| Feature | Status | Location |
|---------|--------|----------|
| Same preprocessor for lexical docs and queries | Done | `main.py` → `run_baseline()` |
| Same tokenization pipeline (lexical) | Done | `TextPreprocessor.process_collection()` |
| Dense queries encoded like dense docs | Done | Raw text + minimal normalization in `embeddings.py` |

**Action items:**
- [ ] Ensure hybrid paths use consistent query processing across lexical and dense stages
- [ ] Document why dense path skips stemming/stopwords (standard practice for transformer encoders)

---

## 6. Query Refinement

### Specification

Query formulation assistance: expansion, suggestions, spelling correction, history-based refinement.

### Current status: **Missing**

| Feature | Status |
|---------|--------|
| Query expansion | Missing |
| Query suggestions | Missing |
| Spelling correction | Missing |
| Search history–based refinement | Missing |
| Query Refinement Service (SOA) | Missing |

**Action items:**
- [ ] Implement at least one refinement technique (e.g. pseudo-relevance feedback or spelling correction)
- [ ] Evaluate retrieval **before** and **after** refinement (required by spec)
- [ ] Expose refinement via API and optionally in the UI

---

## 7. Query Matching & Ranking

### Specification

Match query representation to document representation and rank by relevance (e.g. VSM, embedding cosine similarity).

### Current status: **Partial (lexical + dense done; hybrid missing)**

| Feature | Status | Location |
|---------|--------|----------|
| TF-IDF matching and ranking | Done | `retrieval.py` |
| BM25 matching and ranking | Done | `retrieval.py` |
| Top-K ranking | Done | `retrieve()`, `retrieve_batch()` |
| Embedding cosine similarity | **Done** | `DenseSearchEngine` + FAISS `IndexFlatIP` |
| Hybrid ranking pipeline | **Missing** | — |

**Action items:**
- [x] Add cosine similarity retrieval for dense vectors
- [ ] Integrate hybrid ranking (serial and parallel)

---

## 8. Service-Oriented Architecture (SOA)

### Specification

Design the system as SOA with separate, independently testable services and clear interfaces (REST API, message queue, or RPC).

### Required services

| Service | Status | Notes |
|---------|--------|-------|
| Preprocessing Service | Partial | Python class only; not a standalone service |
| Indexing Service | Partial | Python classes only (`InvertedIndex`, `VectorIndex`) |
| Retrieval Service | Partial | `SearchEngine`, `DenseSearchEngine` — not standalone |
| Ranking & Evaluation Service | Partial | Python class only |
| Query Refinement Service | **Missing** | — |
| UI / API Gateway | **Missing** | — |

### SOA requirements

| Requirement | Status |
|-------------|--------|
| Clear interfaces between services | Missing |
| REST API / Message Queue / RPC | Missing |
| Independent deployment per service | Missing |
| Design patterns documentation | Missing |
| Architecture diagram | Missing |
| Scalability / maintainability / loose coupling (bonus) | Missing |

**Current architecture:** single-script pipeline with two modes — not SOA.

```
main.py --mode baseline → data_loader → preprocessing → indexing → retrieval → evaluation
main.py --mode dense    → data_loader → embeddings → vector_index → dense retrieval → evaluation
```

**Action items:**
- [ ] Refactor into services (e.g. FastAPI microservices or modular service layer)
- [ ] Add REST endpoints for preprocess, index, search, evaluate, refine
- [ ] Draw SOA architecture diagram for the report
- [ ] Document communication protocol (REST recommended)

---

## 9. Evaluation

### Specification

Evaluate using standard IR metrics and compare all representation methods.

### Required metrics

| Metric | Required | Status | Location |
|--------|----------|--------|----------|
| MAP | Yes | Done | `evaluation.py` |
| Recall | Yes | Done | `evaluation.py` |
| Precision@10 | Yes | Done | `evaluation.py` |
| nDCG | Yes | Done | `evaluation.py` |
| MRR | Not in spec | Done (extra) | `evaluation.py` |

### Required evaluation workflow

| Requirement | Status |
|-------------|--------|
| Evaluate before additional features | Done | `results/baseline/run_full/` |
| Evaluate after additional features | Partial | Dense embedding eval done; hybrid + refinement pending |
| Compare TF-IDF, BM25, Embeddings, Hybrid | Partial (TF-IDF + BM25 + Embedding; no hybrid) |
| Analyze impact on retrieval quality | Partial (JSON artifacts; written analysis pending) |
| Analyze impact on retrieval speed | Partial (timing in metadata JSON files) |
| Justify model choices in report | Missing |
| Full query set evaluation | Done | 10,000 qrel-covered queries |

**Lexical baseline results (10,000 queries, full corpus ~523K docs):**

| Model | MAP | P@10 | Recall | nDCG@10 |
|-------|-----|------|--------|---------|
| BM25 | 0.727 | 0.118 | 0.876 | 0.776 |
| TF-IDF | 0.568 | 0.102 | 0.769 | 0.627 |

Artifacts: `results/baseline/run_full/` and `results/baseline/run_200000/` (200K validation run).

**Dense embedding results (10,000 queries, 200K docs — current `MAX_DOCS` setting):**

| Model | MAP | P@10 | Recall | nDCG@10 |
|-------|-----|------|--------|---------|
| Embedding (MiniLM-L6-v2) | 0.360 | 0.070 | 0.399 | 0.385 |
| BM25 (same 200K subset) | 0.316 | 0.061 | 0.371 | 0.343 |
| TF-IDF (same 200K subset) | 0.257 | 0.055 | 0.335 | 0.288 |

Artifacts: `results/dense/run_200000/` (includes side-by-side lexical baseline in `dense_comparison.json`).

**Note:** On the 200K subset, dense embeddings outperform lexical methods. Full-corpus dense numbers are not yet available.

**Action items:**
- [x] Implement nDCG@10 in `evaluation.py`
- [x] Evaluate on all 10,000 test queries
- [x] Add dense embedding evaluation pipeline and artifacts
- [ ] Run dense evaluation on full corpus (~523K docs)
- [ ] Run full comparison: TF-IDF, BM25, Embeddings, Serial Hybrid, Parallel Hybrid
- [x] Measure and report indexing/retrieval time per method (lexical + dense)
- [ ] Write before/after analysis for query refinement and hybrid add-ons

---

## 10. User Interface

### Specification

Web or mobile UI with dataset handling, search, and results display.

### Current status: **Missing**

| Requirement | Status |
|-------------|--------|
| Web or mobile application | Missing |
| Query input | Missing (CLI/script only) |
| Display retrieved results | Missing |
| Results in Arabic (per original spec) | Missing |
| Select representation model (TF-IDF, BM25, hybrid, etc.) | Missing |
| Optional advanced controls | Missing |

**Action items:**
- [ ] Build a web UI (e.g. React + FastAPI or simple HTML/JS)
- [ ] Connect UI to SOA / REST API
- [ ] Allow model selection from the interface
- [ ] Show ranked results with scores and snippets

---

## 11. Optional Bonus Features

The specification lists optional extensions (RAG, vector stores, multilingual retrieval, crawling, distributed IR, clustering, personalization, topic detection, agents, LTR).

**Status:** None implemented.

These are optional and depend on team size and supervisor approval. **Core requirements must be completed first.**

---

## 12. Deliverables Checklist

| Deliverable | Required | Status |
|-------------|----------|--------|
| Arabic technical report with references | Yes | **Missing** |
| Video: dataset used and description | Yes | **Missing** |
| Video: project steps and each service | Yes | **Missing** |
| System architecture diagram (SOA) | Yes | **Missing** |
| Evaluation reports (as specified) | Yes | Partial (lexical + dense JSON; hybrid/refinement pending) |
| Team work division document | Yes | **Missing** |
| Runnable demo + compatible UI | Yes | Partial (script runs; no UI) |
| Private GitHub repository + README | Yes | Partial (`requirements.txt` exists; no README yet) |

**Action items:**
- [ ] Write Arabic report (implementation, design, evaluation, references)
- [ ] Record dataset and system demo videos
- [ ] Create architecture diagram (SOA + data flow)
- [ ] Add README with setup, structure, and how to run (`baseline` and `dense` modes)
- [ ] Document team member responsibilities

---

## 13. Technical Constraints

| Constraint | Status |
|------------|--------|
| Python | Done |
| One dataset (updated requirement) | Done |
| No Antique dataset | Done |
| Dependencies documented | Partial | `requirements.txt` added |
| Deadline (spec: 7/3) | Confirm with supervisor |

---

## 14. What Is Already Implemented (Summary)

The following are in place and can be extended rather than rewritten:

1. **Data loading** — `ir_datasets`, documents, queries, qrels (`data_loader.py`)
2. **Preprocessing** — clean, tokenize, stop words, stem (`preprocessing.py`)
3. **Inverted index** — TF, DF, doc lengths (`indexing.py`)
4. **Lexical retrieval** — TF-IDF and BM25 with top-K ranking (`retrieval.py` → `SearchEngine`)
5. **Dense retrieval** — sentence-transformers encoding + FAISS search (`embeddings.py`, `vector_index.py`, `DenseSearchEngine`)
6. **Evaluation** — MAP, Recall, P@10, MRR, nDCG@10 (`evaluation.py`)
7. **Baseline workflow** — full-query lexical eval, timing, artifact export (`main.py` → `run_baseline()`)
8. **Dense workflow** — embed/index/eval with cache, side-by-side lexical comparison (`main.py` → `run_dense_eval()`)
9. **CLI modes** — `--mode baseline` and `--mode dense` (`main.py`)
10. **Query selection** — qrel-covered queries with optional sampling (`data_loader.py`)
11. **Configuration** — paths, BM25 params, embedding settings, eval settings (`config.py`)
12. **Utilities** — logging, JSON/pickle I/O (`utils.py`)
13. **Dependencies** — `requirements.txt` (ir_datasets, nltk, numpy, scipy, scikit-learn, sentence-transformers, faiss-cpu)

---

## 15. Recommended Implementation Order

### Phase 1 — Complete core IR (high priority)

1. ~~Full evaluation set (all queries) + nDCG metric~~ **Done**
2. ~~Embedding retrieval + vector index~~ **Done**
3. Serial and parallel hybrid with fusion  
4. BM25 parameter tuning + short justification  
5. Full-corpus dense evaluation (`MAX_DOCS = None`)

### Phase 2 — System architecture (high priority)

6. SOA refactor + REST API (FastAPI)  
7. Query refinement module  
8. Before/after evaluation for refinement  

### Phase 3 — User-facing (high priority)

9. Web UI with model selection and search  
10. End-to-end demo  

### Phase 4 — Deliverables (required for submission)

11. Arabic report + architecture diagram  
12. Evaluation comparison tables and analysis  
13. Demo videos + README + GitHub repository  
14. Team work division document  

---

## 16. File-Level Gap Map

| File / Module | Exists | Missing capability |
|---------------|--------|-------------------|
| `config.py` | Yes | API settings, hybrid/fusion settings |
| `data_loader.py` | Yes | — |
| `preprocessing.py` | Yes | Lemmatization experiments, spelling correction |
| `indexing.py` | Yes | Load inverted index from pickle on re-run |
| `retrieval.py` | Yes | Hybrid serial/parallel fusion |
| `embeddings.py` | **Yes** | — |
| `vector_index.py` | **Yes** | Approximate index (currently exact FlatIP only) |
| `evaluation.py` | Yes | Export formatted for Arabic report |
| `main.py` | Yes | SOA entry points, hybrid mode |
| `requirements.txt` | **Yes** | — |
| `services/` | **No** | REST API services |
| `hybrid.py` | **No** | Serial / parallel fusion |
| `query_refinement.py` | **No** | Query expansion / correction |
| `frontend/` | **No** | Web UI |
| `docs/` (report) | Partial | Arabic final report, diagrams |
| `README.md` | **No** | Project documentation |

---

## 17. Bottom Line

**Satisfied with one dataset:** Quora test split meets the updated dataset requirement with full corpus indexed (lexical) and 200K-document dense evaluation complete.

**Completed since last gap report (June 17 → June 20):**
- Dense embedding module and FAISS vector index with persistence/caching  
- `DenseSearchEngine` and `--mode dense` evaluation pipeline  
- 200K-document embedding evaluation on 10,000 queries with timing and comparison to lexical baseline  
- `requirements.txt` with sentence-transformers and faiss-cpu  

**Still required for a complete submission:**

- Serial + parallel hybrid retrieval with fusion  
- Query refinement + before/after evaluation  
- Full-corpus dense evaluation  
- BM25 parameter tuning + justification  
- SOA + REST API  
- Web UI  
- Arabic report, videos, README, and architecture documentation  

**Estimated remaining effort:** moderate-to-substantial — lexical and dense IR are working; hybrid, refinement, SOA, UI, and deliverables remain.

---

*Last updated from codebase review against IR Project 2026 specification (June 20, 2026). Dataset requirement adjusted to one dataset per project update.*
