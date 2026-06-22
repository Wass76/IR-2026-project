# IR Project 2026 — System Architecture Diagram

**Project:** Information Retrieval System  
**Architecture style:** Service-Oriented Architecture (SOA)  
**Dataset:** `beir/quora/test` (ir-datasets)  
**Stack:** Python, FastAPI, FAISS, sentence-transformers, NLTK

---

## 1. High-Level SOA Architecture

```mermaid
flowchart TB
    subgraph Client["Presentation Layer"]
        UI["Web UI\nfrontend/index.html"]
        CLI["CLI Pipeline\ncode/main.py"]
        Swagger["API Docs\n/docs"]
    end

    subgraph Gateway["API Gateway"]
        API["FastAPI Gateway\ncode/api/app.py"]
    end

    subgraph Services["Service Layer (SOA)"]
        IR["IRSystem Orchestrator\ncode/services/ir_system.py"]
        PRE["Preprocessing Service\npreprocessing.py"]
        IDX["Indexing Service\nindexing.py + vector_index.py"]
        RET["Retrieval Service\nretrieval.py"]
        REF["Query Refinement Service\nquery_refinement.py"]
        EVAL["Evaluation Service\nevaluation.py"]
        TUNE["BM25 Tuning Service\nbm25_tuning.py"]
    end

    subgraph Models["Representation & Ranking Models"]
        TFIDF["VSM / TF-IDF"]
        BM25["BM25\n(tuned k1, b)"]
        EMB["Dense Embeddings\nsentence-transformers"]
        HYB_S["Hybrid Serial\nBM25 → Dense re-rank"]
        HYB_P["Hybrid Parallel\nRRF fusion"]
    end

    subgraph Storage["Persistence Layer"]
        DATA[("ir-datasets\nQuora corpus")]
        INV[("Inverted Index\nin-memory")]
        FAISS[("Vector Index\nmodels/vector_index.faiss")]
        RES[("Results & Metrics\nresults/")]
        CFG[("Tuned Params\nmodels/bm25_params.json")]
    end

    UI --> API
    Swagger --> API
    CLI --> Services

    API --> IR
    IR --> PRE
    IR --> IDX
    IR --> RET
    IR --> REF
    IR --> EVAL
    IR --> TUNE

    RET --> TFIDF
    RET --> BM25
    RET --> EMB
    RET --> HYB_S
    RET --> HYB_P

    PRE --> INV
    IDX --> INV
    IDX --> FAISS
    TUNE --> CFG
    EVAL --> RES

    DATA --> IR
    INV --> RET
    FAISS --> EMB
    CFG --> BM25
```

---

## 2. Service Endpoints (REST API)

```mermaid
flowchart LR
    subgraph Gateway["FastAPI Gateway :8000"]
        E1["POST /services/index/load"]
        E2["GET  /services/index/status"]
        E3["POST /services/preprocess"]
        E4["POST /services/refine"]
        E5["POST /services/search"]
        E6["GET  /services/evaluation/summary"]
        E7["GET  /services/bm25/params"]
    end

    E1 --> IDX["Indexing Service"]
    E2 --> IDX
    E3 --> PRE["Preprocessing Service"]
    E4 --> REF["Query Refinement Service"]
    E5 --> RET["Retrieval Service"]
    E6 --> EVAL["Evaluation Service"]
    E7 --> TUNE["BM25 Tuning Service"]
```

| Service | REST endpoint | Module |
|---------|---------------|--------|
| API Gateway | `/`, `/health`, `/docs` | `code/api/app.py` |
| Indexing | `/services/index/load`, `/status` | `indexing.py`, `vector_index.py` |
| Preprocessing | `/services/preprocess` | `preprocessing.py` |
| Query Refinement | `/services/refine` | `query_refinement.py` |
| Retrieval | `/services/search` | `retrieval.py` |
| Evaluation | `/services/evaluation/summary` | `evaluation.py` |
| BM25 Tuning | `/services/bm25/params` | `bm25_tuning.py` |

---

## 3. Search Request Flow (Sequence)

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as API Gateway
    participant IR as IRSystem
    participant REF as Query Refinement
    participant PRE as Preprocessing
    participant RET as Retrieval
    participant IDX as Indexes

    User->>UI: Enter query + select model
    UI->>API: POST /services/search
    API->>IR: search(query, model, use_refinement)

    alt use_refinement = true
        IR->>REF: refine(query)
        REF->>RET: initial BM25 (PRF)
        REF-->>IR: expanded query tokens
    end

    IR->>PRE: tokenize / stem query
    PRE-->>IR: query tokens

    alt model = bm25 / tfidf
        IR->>RET: lexical retrieve
        RET->>IDX: inverted index lookup
    else model = embedding
        IR->>RET: dense retrieve
        RET->>IDX: FAISS vector search
    else model = hybrid
        IR->>RET: serial or parallel hybrid
        RET->>IDX: inverted + vector index
    end

    IDX-->>RET: ranked doc IDs + scores
    RET-->>IR: top-K results
    IR-->>API: results + snippets
    API-->>UI: JSON response
    UI-->>User: display ranked results
```

---

## 4. Offline Evaluation Pipeline (CLI)

```mermaid
flowchart TD
    START(["python main.py --mode ..."]) --> MODE{Mode?}

    MODE -->|baseline| B1["Load dataset"]
    MODE -->|dense / hybrid| D1["Load + index corpus"]
    MODE -->|tune-bm25| T1["Grid search k1, b"]
    MODE -->|refinement| R1["Before/after BM25 eval"]
    MODE -->|api| A1["Start FastAPI server"]
    MODE -->|full| F1["baseline → dense+hybrid"]

    B1 --> B2["Preprocess docs & queries"]
    B2 --> B3["Build inverted index"]
    B3 --> B4["Retrieve: BM25 + TF-IDF"]
    B4 --> B5["Evaluate: MAP, Recall, P@10, nDCG"]
    B5 --> OUT1[("results/baseline/")]

    D1 --> D2["Build / load FAISS index"]
    D2 --> D3["Retrieve: embedding + hybrid"]
    D3 --> D4["Compare all models"]
    D4 --> OUT2[("results/dense/")]

    T1 --> OUT3[("results/bm25_tuning/\nmodels/bm25_params.json")]

    R1 --> OUT4[("results/refinement/")]

    A1 --> OUT5["http://127.0.0.1:8000"]
```

---

## 5. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        Q["User Query"]
        D["Documents\n(Quora pairs)"]
    end

    subgraph Processing
        P1["Clean & Tokenize"]
        P2["Stem / Normalize"]
        P3["Encode Embeddings"]
    end

    subgraph Indexes
        I1["Inverted Index\nterm → doc:tf"]
        I2["Vector Index\nFAISS IndexFlatIP"]
    end

    subgraph Retrieval
        L["Lexical\nTF-IDF / BM25"]
        V["Dense\nCosine similarity"]
        H["Hybrid\nSerial / Parallel RRF"]
    end

    subgraph Output
        RK["Ranked Results"]
        MT["Evaluation Metrics"]
    end

    D --> P1 --> P2 --> I1
    D --> P3 --> I2
    Q --> P1
    Q --> P3

    I1 --> L
    I2 --> V
    L --> H
    V --> H

    L --> RK
    V --> RK
    H --> RK
    RK --> MT
```

---

## 6. Component Map (Codebase)

```
IR-2026-project/
├── code/
│   ├── main.py              # CLI orchestrator (baseline, dense, hybrid, tuning, refinement, api)
│   ├── config.py            # Global configuration
│   ├── data_loader.py       # ir-datasets loader
│   ├── preprocessing.py     # Preprocessing Service
│   ├── indexing.py          # Inverted Index Service
│   ├── vector_index.py      # Vector Index Service (FAISS)
│   ├── embeddings.py        # Sentence-transformers encoder
│   ├── retrieval.py         # Retrieval Service (lexical, dense, hybrid)
│   ├── query_refinement.py  # Query Refinement Service
│   ├── evaluation.py        # Evaluation Service
│   ├── bm25_tuning.py       # BM25 parameter tuning
│   ├── api/app.py           # API Gateway (FastAPI)
│   └── services/ir_system.py # SOA service orchestrator
├── frontend/index.html        # Web search UI
├── models/                    # Cached indexes & tuned params
├── results/                   # Evaluation artifacts
└── data/                      # Downloaded dataset (ir_datasets)
```

---

## 7. Design Principles (SOA)

| Principle | How it is applied |
|-----------|-------------------|
| **Loose coupling** | UI and CLI talk to services only through REST API or `IRSystem` interface |
| **Reusability** | Core modules (`retrieval`, `evaluation`, `preprocessing`) shared by CLI and API |
| **Separation of concerns** | Each service has a single responsibility |
| **Independent testing** | Each module can be run/tested separately (`main.py --mode ...`) |
| **Scalability** | Vector index cached on disk; API loads indexes once at startup |

---

## 8. How to render this diagram

- **GitHub / VS Code:** Mermaid blocks render automatically in Markdown preview.
- **Report (Word/PDF):** Paste diagrams into [mermaid.live](https://mermaid.live) and export as PNG/SVG.
- **Presentation:** Use the exported PNG from section 1 (High-Level SOA) as the main architecture slide.

---

*Generated for IR Project 2026 — matches the current codebase structure.*
