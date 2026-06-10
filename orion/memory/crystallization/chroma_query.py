from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def query_chroma_collection(
    *,
    host: str,
    port: int,
    collection: str,
    query_embedding: list[float],
    n_results: int = 8,
) -> list[dict[str, Any]]:
    """Query Chroma HTTP API for semantic hits (Postgres remains canonical)."""
    if not query_embedding:
        return []
    try:
        import chromadb  # type: ignore
    except ImportError:
        logger.warning("chromadb not installed; skipping chroma query")
        return []

    try:
        client = chromadb.HttpClient(host=host, port=port)
        coll = client.get_or_create_collection(collection)
        result = coll.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "documents", "distances"])
    except Exception as exc:
        logger.warning("chroma_query_failed collection=%s error=%s", collection, exc)
        return []

    hits: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for i, doc_id in enumerate(ids):
        hits.append({
            "doc_id": doc_id,
            "text": docs[i] if i < len(docs) else None,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return hits
