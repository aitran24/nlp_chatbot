from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.rag_service import RagService


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RagService()
    app.state.rag = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="DUT RAG Backend", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    service: RagService = app.state.rag
    return {
        "ok": True,
        "graph_enabled": service.neo4j_driver is not None,
        "reranker_enabled": service.reranker is not None,
    }


@app.post("/plan")
def plan(req: QueryRequest):
    service: RagService = app.state.rag
    retrieval = service.retrieve(query=req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "entities": retrieval["entities"].__dict__,
        "plan": retrieval["plan"].__dict__,
        "graph_match": None if retrieval["graph"] is None else retrieval["graph"].__dict__,
        "vector_preview": retrieval["vector_hits"][:3],
    }


@app.post("/query")
def query(req: QueryRequest):
    service: RagService = app.state.rag
    return service.answer(query=req.query, top_k=req.top_k)
