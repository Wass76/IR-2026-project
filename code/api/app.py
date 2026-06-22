"""
REST API gateway (SOA) for the IR system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import TOP_K, PROJECT_ROOT
from services.ir_system import IRSystem

FRONTEND_DIR = PROJECT_ROOT / "frontend"


class PreprocessRequest(BaseModel):
    text: str


class RefineRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    model: str = Field(
        default="bm25",
        description="bm25 | tfidf | embedding | hybrid_serial | hybrid_parallel",
    )
    top_k: int = Field(default=TOP_K, ge=1, le=100)
    use_refinement: bool = False


def create_app(ir_system=None):
    app = FastAPI(
        title="IR Project 2026 API",
        description="Service-oriented retrieval API: preprocess, index, search, refine, evaluate",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    system = ir_system or IRSystem()

    @app.get("/")
    def root():
        ui = FRONTEND_DIR / "index.html"
        if ui.exists():
            return FileResponse(ui)
        return {
            "message": "IR Project 2026 API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health():
        return {"status": "ok", **system.status()}

    @app.post("/services/index/load")
    def load_index(max_docs: int | None = None):
        try:
            return system.load(max_docs=max_docs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/services/index/status")
    def index_status():
        return system.status()

    @app.post("/services/preprocess")
    def preprocess(req: PreprocessRequest):
        try:
            system._ensure_loaded()
            return system.preprocess(req.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/services/refine")
    def refine(req: RefineRequest):
        try:
            return system.refine(req.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/services/search")
    def search(req: SearchRequest):
        try:
            return system.search(
                query=req.query,
                model=req.model,
                top_k=req.top_k,
                use_refinement=req.use_refinement,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/services/evaluation/summary")
    def evaluation_summary():
        return IRSystem.latest_metrics()

    @app.get("/services/bm25/params")
    def bm25_params():
        from config import BM25_PARAMS_PATH, BM25_K1, BM25_B
        from utils import load_json

        if BM25_PARAMS_PATH.exists():
            return load_json(BM25_PARAMS_PATH)
        return {"k1": BM25_K1, "b": BM25_B, "tuned": False}

    return app


app = create_app()
