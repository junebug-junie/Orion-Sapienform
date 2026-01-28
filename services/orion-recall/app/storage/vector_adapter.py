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
    profile_name: Optional[str] = None,
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

    for coll_name in collections:
        try:
            coll = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        try:
            base_where: Optional[Dict[str, Any]] = None
            if metadata_filters or node_id:
                base_where = dict(metadata_filters or {})
                if node_id:
                    base_where["source_node"] = node_id

            def _query(where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
                return coll.query(
                    query_embeddings=[query_embedding],
                    n_results=max_items * 2,
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )

            res = _query(base_where)
        except Exception:
            continue

        def _append_results(
            result: Dict[str, Any],
            extra_tags: Optional[List[str]] = None,
            score_multiplier: float = 1.0,
            skip_ids: Optional[set[str]] = None,
        ) -> int:
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]

            count = 0
            k = min(len(ids), len(docs), len(metas), len(dists) or len(ids))
            for i in range(k):
                nid = ids[i]
                nid_str = str(nid)
                if skip_ids and nid_str in skip_ids:
                    continue
                ntext = docs[i] or ""
                meta = metas[i] or {}
                if not isinstance(meta, dict):
                    meta = {}
                dist = dists[i] if isinstance(dists, list) and i < len(dists) else None

                if settings.RECALL_EXCLUDE_REJECTED and str(meta.get("memory_status") or "").lower() == "rejected":
                    continue
                if settings.RECALL_DURABLE_ONLY and str(meta.get("memory_tier") or "").lower() != "durable":
                    continue
                if not _recent_enough(meta, since_ts):
                    continue

                sim = None
                if isinstance(dist, (int, float)) and not math.isnan(dist):
                    sim = max(0.0, 1.0 - float(dist)) * score_multiplier

                canonical_id = str(meta.get("correlation_id") or nid_str)
                tags = [
                    "vector-assoc",
                    f"collection:{coll_name}",
                ] + ([str(meta.get("source"))] if meta.get("source") else [])
                if canonical_id != nid_str:
                    meta = dict(meta)
                    meta["vector_doc_id"] = nid_str
                    tags.append("canon:correlation_id")
                if extra_tags:
                    tags.extend(extra_tags)

                frags.append(
                    {
                        "id": canonical_id,
                        "source": "vector",
                        "source_ref": coll_name,
                        "text": str(ntext)[:1200],
                        "ts": _parse_meta_ts(meta) or since_ts,
                        "tags": tags,
                        "score": sim or 0.0,
                        "meta": meta,
                    }
                )
                count += 1
            return count

        _append_results(res)

    frags.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    logger.debug("vector recall: collections=%s hits=%s", collections, len(frags))
    return frags[:max_items]


def fetch_vector_exact_matches(
    *,
    tokens: List[str],
    max_items: int,
    session_id: Optional[str] = None,
    profile_name: Optional[str] = None,
    node_id: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    client = _get_client()
    if client is None:
        return []
    collections = _parse_collections(settings.RECALL_VECTOR_COLLECTIONS)
    if not collections:
        return []

    results: List[Dict[str, Any]] = []
    for coll_name in collections:
        try:
            coll = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        base_where: Optional[Dict[str, Any]] = None
        if metadata_filters or node_id:
            base_where = dict(metadata_filters or {})
            if node_id:
                base_where["source_node"] = node_id

        for token in tokens:
            try:
                res = coll.get(
                    where=base_where,
                    where_document={"$contains": token},
                    include=["documents", "metadatas"],
                    limit=max_items,
                )
            except Exception:
                continue
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []

            for idx, nid in enumerate(ids):
                if len(results) >= max_items:
                    return results
                nid_str = str(nid)
                doc = docs[idx] if idx < len(docs) else ""
                meta = metas[idx] if idx < len(metas) else {}
                if not isinstance(meta, dict):
                    meta = {}
                if settings.RECALL_EXCLUDE_REJECTED and str(meta.get("memory_status") or "").lower() == "rejected":
                    continue
                if settings.RECALL_DURABLE_ONLY and str(meta.get("memory_tier") or "").lower() != "durable":
                    continue
                tags = [
                    "vector-exact",
                    f"collection:{coll_name}",
                ]
                if meta.get("source"):
                    tags.append(str(meta.get("source")))
                canonical_id = str(meta.get("correlation_id") or nid_str)
                if canonical_id != nid_str:
                    meta = dict(meta)
                    meta["vector_doc_id"] = nid_str
                    tags.append("canon:correlation_id")
                results.append(
                    {
                        "id": canonical_id,
                        "source": "vector",
                        "source_ref": coll_name,
                        "text": str(doc)[:1200],
                        "ts": _parse_meta_ts(meta) or 0.0,
                        "tags": tags,
                        "score": 0.95,
                        "meta": meta,
                    }
                )
    return results[:max_items]
