# app/rag.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
from fastembed import TextEmbedding
from orion_llm_client import OrionLLMClient

# ---- chunking ----
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split raw text into overlapping word chunks for embedding."""
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += step
    return chunks

# ---- embeddings (FastEmbed) ----
_EMBED = None
_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")

def get_embedder():
    global _EMBED
    if _EMBED is None:
        cache = os.getenv("FASTEMBED_CACHE_PATH")
        _EMBED = TextEmbedding(model_name=_MODEL, cache_dir=cache)
    return _EMBED

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert a batch of texts into dense embeddings."""
    emb = get_embedder()
    return [v.tolist() for v in emb.embed(texts)]

def embedding_dim() -> int:
    """Return the dimensionality of the embeddings (needed for Qdrant schema)."""
    return len(embed_texts(["x"])[0])

# ---- RAG System Prompt ----
RAG_SYSTEM_TEMPLATE = (
    "You are a concise assistant. Use ONLY the provided context to answer. "
    "If the context is insufficient, say you don't know.\n\nContext:\n{context}"
)

@dataclass
class RAGPipeline:
    top_k: int = 4

    def answer(self, question: str) -> tuple[str, list[dict]]:
        from .vector import search
        qv = embed_texts([question])[0]
        hits = search(qv, k=self.top_k)

        chosen, ctx_parts = [], []
        for score, payload in hits:
            txt = payload["text"]
            chosen.append({"score": float(score), "source": payload.get("source"), "text": txt})
            ctx_parts.append(f"[score={score:.3f}] {txt}")

        context = "\n\n".join(ctx_parts)

        # Use shared OrionLLMClient
        cli = OrionLLMClient(base_url=os.getenv("BRAIN_URL"), model=os.getenv("MODEL"))
        response = cli.generate(
            f"Answer the question using the following context:\n{context}",
            options={"temperature": 0.3, "num_predict": int(os.getenv("MAX_TOKENS","256"))},
            system="You are a concise assistant. Use only the context provided."
        )
        cli.close()
        return response, chosen
