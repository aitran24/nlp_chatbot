from contextlib import asynccontextmanager
import json
from urllib.request import urlopen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_MODEL
from backend.rag_service import RagService


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RagService()
    app.state.rag = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="DUT RAG Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    service: RagService = app.state.rag
    return {
        "ok": True,
        "graph_enabled": service.neo4j_driver is not None,
        "reranker_enabled": service.reranker is not None,
    }


@app.get("/models")
def models():
    # if LLM_PROVIDER == "ollama":
    #     print ("Fetching model list from Ollama API...")
    #     try:
    #         with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
    #             payload = json.loads(resp.read().decode("utf-8"))
    #         names = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
    #         return {"provider": LLM_PROVIDER, "default_model": OLLAMA_MODEL, "models": names}
    #     except Exception:
    #         return {"provider": LLM_PROVIDER, "default_model": OLLAMA_MODEL, "models": [OLLAMA_MODEL]}

    if LLM_PROVIDER == "groq":
        # Groq client is used via cloud/API key; we don't have a local list endpoint here.
        return {"provider": LLM_PROVIDER, "default_model": GROQ_MODEL, "models": [GROQ_MODEL]}

    # Fallback
    return {"provider": LLM_PROVIDER, "default_model": GROQ_MODEL, "models": [GROQ_MODEL]}

@app.post("/plan")
def plan(req: QueryRequest):
    service: RagService = app.state.rag
    retrieval = service.retrieve(query=req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "model": req.model,
        "entities": retrieval["entities"].__dict__,
        "plan": retrieval["plan"].__dict__,
        "graph_match": None if retrieval["graph"] is None else retrieval["graph"].__dict__,
        "vector_preview": retrieval["vector_hits"][:3],
    }


@app.post("/query")
def query(req: QueryRequest):
    service: RagService = app.state.rag
    return service.answer(query=req.query, top_k=req.top_k, model=req.model)
