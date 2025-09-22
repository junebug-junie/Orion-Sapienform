# app/ingest.py
import os
from .rag import chunk_text, embed_texts, embedding_dim
from .vector import ensure_collection, upsert_embeddings

# optional pdf support
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Install pypdf to parse PDFs")
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    raise RuntimeError(f"Unsupported file type: {ext}")

def ingest_text(text: str, source: str = "inline", chunk_size=600, overlap=120) -> int:
    ensure_collection(embedding_dim())
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    vecs = embed_texts(chunks)
    payloads = [{"text": c, "source": source} for c in chunks]
    upsert_embeddings(vecs, payloads)
    return len(chunks)

def ingest_path(path: str, **kw) -> int:
    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")
    text = read_file(path)
    return ingest_text(text, source=os.path.abspath(path), **kw)
