# app/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from .ingest import ingest_text, ingest_path
from .rag import RAGPipeline, embedding_dim
from .vector import ensure_collection
from orion_llm_client import OrionLLMClient
import os, traceback

app = FastAPI(title="RAG Service (Phase II)")
rag = RAGPipeline()

# Initialize shared LLM client once at startup
llm_client = OrionLLMClient(
    base_url=os.getenv("BRAIN_URL", "http://orion-brain-service:8088"),
    model=os.getenv("MODEL", "mistral:instruct")
)

class IngestBody(BaseModel):
    path: str | None = None
    text: str | None = None

class AskBody(BaseModel):
    question: str
    k: int = 4

@app.on_event("startup")
async def startup_event():
    # Make sure Qdrant collection exists before first ingestion
    ensure_collection(embedding_dim())

@app.get("/health")
def health():
    return {"ok": True, "backends": llm_client.base_url, "model": llm_client.model}

@app.post("/ingest")
def ingest(b: IngestBody):
    try:
        if not b.path and not b.text:
            return {"ok": False, "error": "Provide 'path' or 'text'."}
        n = ingest_text(b.text, source="inline") if b.text else ingest_path(b.path)
        return {"ok": True, "chunks": n}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

@app.post("/ask")
def ask(b: AskBody):
    try:
        if b.k and b.k != rag.top_k:
            rag.top_k = b.k
        answer, ctx = rag.answer(b.question)  # rag.answer now uses OrionLLMClient internally
        return {"ok": True, "answer": answer, "context": ctx}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
