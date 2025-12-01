# app/storage/vector_adapter.py
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from chromadb import HttpClient
from chromadb.config import Settings as ChromaSettings

from app.settings import settings
from app.types import Fragment


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
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _parse_collections(val: str) -> List[str]:
    if not val:
        return []
    return [c.strip() for c in str(val).split(",") if c.strip()]


def fetch_vector_fragments(
    *,
    query_text: str,
    time_window_days: int,
    max_items: int,
) -> List[Fragment]:
    if not query_text:
        return []

    client = _get_client()
    if client is None:
        return []

    collections = _parse_collections(settings.RECALL_VECTOR_COLLECTIONS)
    if not collections:
        return []

    since_ts = (datetime.utcnow() - timedelta(days=max(1, time_window_days))).timestamp()
    frags: List[Fragment] = []

    for coll_name in collections:
        try:
            coll = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        try:
            res = coll.query(
                query_texts=[query_text],
                n_results=max_items * 2,
                include=["documents", "metadatas", "distances", "ids"],
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

            sal = (sim or 0.2) * 0.9

            frags.append(
                Fragment(
                    id=str(nid),
                    kind="association",
                    source="vector",
                    text=str(ntext)[:1200],
                    ts=_parse_meta_ts(meta) or since_ts,
                    tags=[
                        "vector-assoc",
                        f"collection:{coll_name}",
                    ]
                    + ([str(meta.get("source"))] if meta.get("source") else []),
                    salience=sal,
                    meta=meta,
                )
            )

    frags.sort(key=lambda x: x.salience, reverse=True)
    return frags[:max_items]
