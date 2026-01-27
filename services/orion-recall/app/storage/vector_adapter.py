from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

import httpx

try:
    from chromadb import HttpClient  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    HttpClient = None
    ChromaSettings = None

from app.settings import settings
from orion.schemas.vector.schemas import EmbeddingGenerateV1


def _parse_meta_ts(meta: Dict[str, Any]) -> Optional[float]:
    if not isinstance(meta, dict):
        return None

    for k in ("ts", "time", "epoch"):
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass

    for k in ("timestamp", "created_at", "createdAt", "time_iso"):
        v = meta.get(k)
        if not v:
            continue
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return datetime.strptime(str(v)[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
            except Exception:
                continue
    return None


def _recent_enough(meta: Dict[str, Any], since_ts: float) -> bool:
    ts = _parse_meta_ts(meta)
    return ts is not None and ts >= since_ts


def _get_client() -> Optional[HttpClient]:
    url = settings.RECALL_VECTOR_BASE_URL
    if not url:
        return None
    if HttpClient is None or ChromaSettings is None:
        return None
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _embedding_url() -> Optional[str]:
    url = settings.RECALL_VECTOR_EMBEDDING_URL
    if not url:
        return None
    return str(url).strip() or None


def _embed_query_text(text: str) -> Optional[List[float]]:
    url = _embedding_url()
    if not url:
        return None
    payload = EmbeddingGenerateV1(
        doc_id=str(uuid4()),
        text=text,
        embedding_profile="default",
        include_latent=False,
    )
    try:
        resp = httpx.post(
            url,
            json=payload.model_dump(mode="json"),
            timeout=settings.RECALL_VECTOR_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict):
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
    return None


def _parse_collections(val: str) -> List[str]:
    if not val:
        return []
    return [c.strip() for c in str(val).split(",") if c.strip()]


def fetch_vector_fragments(
    *,
    query_text: str,
    time_window_days: int,
    max_items: int,
    session_id: Optional[str] = None,
    node_id: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Lightweight wrapper around the existing Chroma client used by the legacy recall pipeline.
    Returns a list of dicts that can be converted to MemoryItems downstream.
    """
    if not query_text:
        return []

    try:
        client = _get_client()
    except Exception:
        client = None
    if client is None:
        return []

    collections = _parse_collections(settings.RECALL_VECTOR_COLLECTIONS)
    if not collections:
        return []

    since_ts = (datetime.utcnow() - timedelta(days=max(1, time_window_days))).timestamp()
    frags: List[Dict[str, Any]] = []

    query_embedding = _embed_query_text(query_text)
    if not query_embedding:
        return []

    logger = logging.getLogger("orion-recall.vector")
    has_session_filter = bool(session_id)
    logger.debug(
        "vector recall: collections=%s session_filter=%s session_id=%s",
        collections,
        has_session_filter,
        session_id,
    )

    for coll_name in collections:
        try:
            coll = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        try:
            where: Optional[Dict[str, Any]] = None
            if metadata_filters or session_id or node_id:
                where = dict(metadata_filters or {})
                if session_id:
                    where["session_id"] = session_id
                    if coll_name == "orion_chat":
                        where["source_channel"] = "orion:chat:history:log"
                if node_id:
                    where["source_node"] = node_id

            res = coll.query(
                query_embeddings=[query_embedding],
                n_results=max_items * 2,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except Exception:
            continue

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        k = min(len(ids), len(docs), len(metas), len(dists) or len(ids))
        for i in range(k):
            nid = ids[i]
            ntext = docs[i] or ""
            meta = metas[i] or {}
            dist = dists[i] if isinstance(dists, list) and i < len(dists) else None

            if not _recent_enough(meta, since_ts):
                continue

            sim = None
            if isinstance(dist, (int, float)) and not math.isnan(dist):
                sim = max(0.0, 1.0 - float(dist))

            frags.append(
                {
                    "id": str(nid),
                    "source": "vector",
                    "source_ref": coll_name,
                    "text": str(ntext)[:1200],
                    "ts": _parse_meta_ts(meta) or since_ts,
                    "tags": [
                        "vector-assoc",
                        f"collection:{coll_name}",
                    ]
                    + ([str(meta.get("source"))] if meta.get("source") else []),
                    "score": sim or 0.0,
                    "meta": meta,
                }
            )

    frags.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return frags[:max_items]
