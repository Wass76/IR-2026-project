# IR Project 2026 — System Diagrams

**Dataset:** `beir/quora/test` (~523K docs, 10K test queries)  
**Stack:** Python, FastAPI, SQLite, FAISS, rank-bm25, scikit-learn, sentence-transformers

> Render: open this file in VS Code Markdown preview, or paste Mermaid blocks into [mermaid.live](https://mermaid.live) and export PNG/SVG.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI["Web UI\nfrontend/index.html"]
        CLI["CLI\nmain.py"]
    end

    subgraph Gateway["API Gateway"]
        API["FastAPI\ncode/api/app.py"]
    end

    subgraph Orchestrator["Service Orchestrator"]
        IR["IRSystem\ncode/services/ir_system.py"]
    end

    subgraph Services["Core Services"]
        DL["Data Loader\ndata_loader.py"]
        DS["Document Store\ndocument_store.py"]
        PRE["Preprocessing\npreprocessing.py"]
        IC["Index Cache\nindex_cache.py"]
        RET["Retrieval\nretrieval.py"]
        REF["Query Refinement\nquery_refinement.py"]
        EV["Evaluation\nevaluation.py"]
    end

    subgraph Models["Retrieval Models"]
        BM25["BM25\nrank_bm25"]
        TFIDF["TF-IDF\nsklearn"]
        DEN["Dense\nFAISS + embeddings"]
        HYS["Hybrid Serial"]
        HYP["Hybrid Parallel\nRRF / weighted"]
    end

    subgraph Persistence["Persistence"]
        IRDS[("ir_datasets\nQuora download")]
        SQLITE[("SQLite\ndata/documents.db")]
        PICKLE[("Lexical cache\nmodels/*.pkl")]
        FAISS[("Vector index\nmodels/*.faiss")]
        RES[("Results\nresults/")]
    end

    UI --> API
    CLI --> DL
    CLI --> RET
    API --> IR

    IR --> DL
    IR --> DS
    IR --> PRE
    IR --> IC
    IR --> RET
    IR --> REF

    DL --> IRDS
    DL --> DS
    DS --> SQLITE
    IC --> PICKLE
    RET --> BM25
    RET --> TFIDF
    RET --> DEN
    RET --> HYS
    RET --> HYP
    DEN --> FAISS
    EV --> RES
```

---

## 2. SQLite Document Storage

```mermaid
erDiagram
    documents {
        TEXT doc_id PK
        TEXT original_content
        TEXT processed_tokens
        TEXT metadata
    }

    corpus_meta {
        TEXT key PK
        TEXT value
    }

    documents ||--o{ corpus_meta : "described by"
```

```mermaid
flowchart LR
    subgraph documents_table["Table: documents"]
        D1["doc_id: 188"]
        D2["original_content: What causes nightmares..."]
        D3["processed_tokens: caus, nightmar, seem, real"]
    end

    subgraph meta_table["Table: corpus_meta"]
        M1["dataset → beir/quora/test"]
        M2["max_docs → 200000 or full"]
        M3["doc_count → 200000"]
        M4["processed_cached → true"]
        M5["preprocessing_stemming → true"]
    end

    subgraph usage["Used for"]
        U1["Skip ir_datasets reload"]
        U2["Skip re-preprocessing"]
        U3["full_content in search UI"]
        U4["PRF token lookup"]
    end

    documents_table --> usage
    meta_table --> usage
```

**File:** `data/documents.db`  
**Not stored in DB:** BM25, TF-IDF, inverted index, FAISS (those live in `models/`)

---

## 3. Full Storage & Cache Layers

```mermaid
flowchart TB
    subgraph Source["Source Data"]
        ZIP["BEIR Quora zip\n~/.ir_datasets/"]
        CORPUS["corpus.jsonl\n523K questions"]
        QREL["test.qrels\n15,675 judgments"]
    end

    subgraph Layer1["Layer 1 — Document Cache"]
        DB[("documents.db\noriginal + tokens")]
    end

    subgraph Layer2["Layer 2 — Lexical Cache"]
        INV["inverted_index.pkl"]
        B25["bm25_cache.pkl"]
        TF["tfidf_cache.pkl"]
        META["lexical_index_metadata.json"]
    end

    subgraph Layer3["Layer 3 — Dense Cache"]
        FAISS["vector_index.faiss"]
        IDS["vector_doc_ids.json"]
        EMBM["embedding_metadata.json"]
    end

    subgraph Layer4["Layer 4 — Results"]
        BASE["results/baseline/"]
        DENSE["results/dense/"]
        REFR["results/refinement/"]
        TUNE["results/bm25_tuning/"]
        CHARTS["results/charts/"]
    end

    ZIP --> CORPUS
    CORPUS --> DB
    DB --> INV
    DB --> B25
    DB --> TF
    DB --> FAISS

    INV --> META
    FAISS --> IDS
    FAISS --> EMBM

    B25 --> DENSE
    TF --> DENSE
    FAISS --> DENSE
    QREL --> BASE
    QREL --> DENSE
    QREL --> REFR
    QREL --> TUNE
    DENSE --> CHARTS
```

---

## 4. Load Index Flow (API / UI)

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /services/index/load
    participant IR as IRSystem.load()
    participant DL as data_loader
    participant DB as documents.db
    participant IC as index_cache
    participant FAISS as vector_index

    User->>UI: Click Load Index
    UI->>API: POST /services/index/load
    API->>IR: load(max_docs)

    IR->>DL: load_or_build_processed_corpus()
    alt SQLite cache valid
        DL->>DB: load_corpus_bundle()
        DB-->>DL: originals + tokens
    else Cache miss
        DL->>DL: ir_datasets + preprocess
        DL->>DB: save_corpus_bundle()
    end

    IR->>IC: load_or_build_lexical_index()
    alt Pickle cache valid
        IC-->>IR: inverted + BM25 + TF-IDF
    else Cache miss
        IC->>IC: build + save pickles
    end

    IR->>FAISS: build or load FAISS
    IR->>IR: create SearchEngine, Dense, Hybrid, Refiner
    IR-->>API: status JSON
    API-->>UI: loaded=true, doc_count, cache flags
    UI-->>User: Index ready — search now
```

**Note:** Search also calls `load()` automatically if not loaded yet (`_ensure_loaded()`).

---

## 5. Search Request Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /services/search
    participant IR as IRSystem
    participant PRE as Preprocessing
    participant REF as QueryRefiner
    participant RET as Retrieval
    participant DB as documents.db

    User->>UI: Query + model + optional weights
    UI->>API: search payload
    API->>IR: search()

    IR->>IR: _ensure_loaded()

    opt use_refinement = true
        IR->>REF: refine(tokens)
        REF->>RET: BM25 top-5 for PRF
        REF-->>IR: expanded tokens
    end

    IR->>PRE: process_text(query)

    alt bm25 / tfidf
        IR->>RET: lexical retrieve
    else embedding
        IR->>RET: FAISS search
    else hybrid_serial
        IR->>RET: BM25 top-100 → dense rerank
    else hybrid_parallel
        IR->>RET: BM25 + dense → RRF or weighted fusion
    end

    RET-->>IR: ranked doc_ids + scores
    loop each result
        IR->>DB: get_document(doc_id)
        DB-->>IR: original_content
    end

    IR-->>API: results + full_content
    API-->>UI: JSON
    UI-->>User: ranked list with full text
```

---

## 6. Retrieval Models Comparison

```mermaid
flowchart TB
    Q["User Query"]

    subgraph Lexical["Lexical Path"]
        Q --> TOK["Tokenize + Stem"]
        TOK --> BM25["BM25\nrank_bm25"]
        TOK --> TFIDF["TF-IDF\nsklearn cosine"]
    end

    subgraph Dense["Dense Path"]
        Q --> ENC["Sentence-Transformers\nencode query"]
        ENC --> FAISS["FAISS\nnearest neighbors"]
    end

    subgraph Hybrid["Hybrid Path"]
        BM25 --> HS["Hybrid Serial\nBM25 candidates → dense rerank"]
        BM25 --> HP["Hybrid Parallel"]
        FAISS --> HP
        HP --> RRF["RRF fusion\ndefault"]
        HP --> WGT["Weighted fusion\nUI bm25_weight + dense_weight"]
    end

    BM25 --> TOPK["Top-K Results"]
    TFIDF --> TOPK
    HS --> TOPK
    RRF --> TOPK
    WGT --> TOPK
    FAISS --> TOPK
```

| Model | Method | Best for |
|-------|--------|----------|
| BM25 | Term matching + IDF | Exact word overlap |
| TF-IDF | Cosine on sparse vectors | Baseline lexical |
| Dense | Semantic embeddings | Paraphrases / synonyms |
| Hybrid Serial | BM25 filter → dense rerank | Precision at top |
| Hybrid Parallel | Both paths fused | Balance lexical + semantic |

---

## 7. Hybrid Parallel — Weighted Fusion

```mermaid
flowchart LR
    Q["Query"] --> B["BM25\n top-100"]
    Q --> D["Dense\n top-100"]

    B --> NB["Normalize BM25 scores\n0 to 1"]
    D --> ND["Normalize dense scores\n0 to 1"]

    NB --> FUSE["Fused score =\nbm25_weight × BM25_norm\n+ dense_weight × dense_norm"]
    ND --> FUSE

    FUSE --> SORT["Sort by fused score"]
    SORT --> TOP["Top-K results"]

    W["UI weights\ne.g. 0.7 / 0.3"] --> FUSE
```

**UI test:** same query with `1.0/0.0` vs `0.0/1.0` vs `0.5/0.5` — rank order should change.

---

## 8. Query Refinement Pipeline

```mermaid
flowchart TD
    Q["Original query tokens"] --> SPELL["Spelling correction\nOOV → nearest vocab term"]
    SPELL --> PRF["PRF expansion\nBM25 top-5 docs → add 8 terms"]
    PRF --> HIST["History expansion\nterms from similar past queries"]
    HIST --> RQ["Refined query tokens"]
    RQ --> BM25["BM25 retrieve again"]

    subgraph eval_note["On Quora eval"]
        N1["Queries already clean"]
        N2["Top-5 often not true duplicate"]
        N3["History adds noise in batch eval"]
        N4["MAP: 0.319 → 0.162"]
    end

    BM25 --> eval_note
```

---

## 9. CLI Evaluation Pipeline

```mermaid
flowchart TD
    START(["python main.py --mode MODE"]) --> MODE{Mode?}

    MODE -->|baseline| BL["BM25 + TF-IDF eval"]
    MODE -->|dense / hybrid| DE["All 5 models eval"]
    MODE -->|full| FULL["baseline then dense+hybrid"]
    MODE -->|tune-bm25| TU["Grid search k1 × b"]
    MODE -->|refinement| RF["BM25 before vs after refinement"]
    MODE -->|api| AP["Start FastAPI :8000"]

    BL --> OUT1[("results/baseline/run_*/")]
    DE --> OUT2[("results/dense/run_*/\nfull_comparison.json")]
    FULL --> OUT1
    FULL --> OUT2
    TU --> OUT3[("results/bm25_tuning/\nmodels/bm25_params.json")]
    RF --> OUT4[("results/refinement/\nbefore_after_analysis.json")]
    AP --> OUT5["http://127.0.0.1:8000"]
```

---

## 10. Dataset Structure (BEIR Quora)

```mermaid
flowchart TB
    subgraph BEIR["beir/quora/test"]
        DOCS["Corpus\n522,931 questions\ncorpus.jsonl"]
        QUERIES["Test queries\n10,000\nqueries.jsonl"]
        QRELS["Relevance judgments\n15,675 pairs\ntest.qrels"]
    end

    subgraph Task["Retrieval task"]
        T1["Given a query question"]
        T2["Find duplicate/near-duplicate questions in corpus"]
        T3["Score 1 = relevant in qrels"]
    end

    subgraph Example["Example"]
        EXQ["Query 187:\nWhat causes a nightmare?"]
        EXD["Doc 188:\nWhat causes nightmares that seem real?"]
        EXQ -->|"relevant"| EXD
    end

    QUERIES --> Task
    DOCS --> Task
    QRELS --> Task
    Task --> Example
```

---

## 11. Cache Invalidation — When to Rebuild

```mermaid
flowchart TD
    CHANGE["What changed?"] --> C1["MAX_DOCS\n200K → full"]
    CHANGE --> C2["Preprocessing\nstemming settings"]
    CHANGE --> C3["BM25 k1/b after tuning"]
    CHANGE --> C4["REBUILD_* flags\nin config.py"]
    CHANGE --> C5["Delete cache files"]

    C1 --> DEL["Delete documents.db\n+ models/*"]
    C2 --> DEL
    C3 --> DEL2["Delete bm25_cache.pkl\n+ metadata OR full models/"]
    C4 --> DEL
    C5 --> DEL

    DEL --> REBUILD["Next run rebuilds\nclean from ir_datasets"]
    DEL2 --> REBUILD
```

| Delete this | When |
|-------------|------|
| `data/documents.db` | Change doc count or preprocessing |
| `models/*.pkl` | Lexical cache invalid |
| `models/*.faiss` | Rebuild dense index |
| `data/beir/` | Force re-download (usually keep) |

---

## 12. End-to-End Data Lifecycle

```mermaid
flowchart LR
    A["1. Download\nir_datasets"] --> B["2. Preprocess\nNLTK"]
    B --> C["3. Save\nSQLite"]
    C --> D["4. Index\nBM25 TF-IDF FAISS"]
    D --> E["5. Search\nAPI or CLI"]
    E --> F["6. Evaluate\nvs qrels"]
    F --> G["7. Charts\nplot_results.py"]

    style A fill:#e8f4fc
    style C fill:#fff3cd
    style D fill:#d4edda
    style G fill:#f8d7da
```

---

## Export for Report

1. Open [mermaid.live](https://mermaid.live)
2. Paste any diagram block
3. Export **PNG** or **SVG**
4. Recommended slides:
   - **Section 1** — System architecture
   - **Section 3** — Storage layers
   - **Section 6** — Retrieval models
   - **Section 10** — Dataset

---

*Updated for IR Project 2026 — includes SQLite storage, disk caches, hybrid weights, and refinement.*
