from __future__ import annotations
from datetime import datetime, timedelta
from typing import Iterable, Optional, Dict, Any, List

import math
from chromadb import HttpClient
from app.settings import settings
from app.utils import Fragment, clean_text


def _parse_meta_ts(meta: Dict[str, Any]) -> Optional[float]:
    """
    Accepts multiple timestamp styles:
    - meta["ts"] as epoch seconds (float/int)
    - meta["timestamp"] / meta["created_at"] as ISO8601
    Returns epoch seconds or None.
    """
    if not isinstance(meta, dict):
        return None

    # epoch-like
    for k in ("ts", "time", "epoch"):
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass

    # ISO-like
    for k in ("timestamp", "created_at", "createdAt", "time_iso"):
        v = meta.get(k)
        if not v:
            continue
        try:
            # allow strings like "2025-10-22T16:15:11+00:00"
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except Exception:
            # try a looser parse
            try:
                return datetime.strptime(str(v)[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
            except Exception:
                continue
    return None


def _recent_enough(meta: Dict[str, Any], since_ts: float) -> bool:
    ts = _parse_meta_ts(meta)
    return ts is not None and ts >= since_ts


def enrich_from_chroma(
    frags: List[Fragment],
    hours: int = 24,
    recent_ids: Optional[Iterable[str]] = None,
    n_results: int = 8,
) -> List[Fragment]:
    """
    For each input fragment (collapse/chat), retrieve vector neighbors from Chroma,
    then filter those neighbors to keep only:
      - items with metadata timestamp within the last `hours`, OR
      - items whose metadata 'collapse_id' or 'id' is in `recent_ids` (fallback)
    Returns a new list including the original frags + association fragments.
    """
    if not frags:
        return frags

    try:
        client = HttpClient(host=settings.VECTOR_DB_HOST, port=settings.VECTOR_DB_PORT)
        coll = client.get_or_create_collection(name=settings.VECTOR_DB_COLLECTION)
    except Exception:
        # Chroma unavailable — pass through
        return frags

    since_ts = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    recent_id_set = set(recent_ids or [])
    out = list(frags)

    for f in frags:
        # only enrich textual kinds
        if f.kind not in ("collapse", "chat"):
            continue

        q = clean_text(getattr(f, "text", "") or "")[:700]
        if not q:
            continue

        try:
            # ask for metadata so we can filter by recency
            res = coll.query(
                query_texts=[q],
                n_results=n_results * 3,  # overfetch so we can filter down
                include=["documents", "metadatas", "distances", "ids"],
            )
        except Exception:
            continue

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        # Defensive: align lengths
        k = min(len(ids), len(docs), len(metas), len(dists) or len(ids))
        candidates = []
        for i in range(k):
            nid = ids[i]
            ntext = docs[i] or ""
            meta = metas[i] or {}
            dist = dists[i] if isinstance(dists, list) and i < len(dists) else None

            # recency check via metadata timestamp
            keep = _recent_enough(meta, since_ts)
            # fallback: if no timestamps, allow if collapse_id/id is among recent SQL IDs
            if not keep and recent_id_set:
                cid = meta.get("collapse_id") or meta.get("id")
                keep = cid in recent_id_set

            if not keep:
                continue

            # similarity → salience bump (if we got distances)
            sim = None
            if isinstance(dist, (int, float)) and not math.isnan(dist):
                # Chroma returns cosine distance by default: sim ≈ 1 - dist
                sim = max(0.0, 1.0 - float(dist))

            candidates.append((nid, ntext, meta, sim))

        # Trim to n_results after filtering
        candidates = candidates[:n_results]

        for nid, ntext, meta, sim in candidates:
            assoc_id = meta.get("collapse_id") or meta.get("id") or nid
            out.append(
                Fragment(
                    id=f"{f.id}__vec__{nid}",
                    kind="association",
                    text=clean_text(ntext)[:1200],
                    tags=["vector-assoc"]
                    + ([str(meta.get("source"))] if meta.get("source") else [])
                    + ([f"assoc:{assoc_id}"] if assoc_id else []),
                    salience=max(getattr(f, "salience", 0.1), (sim or 0.2) * 0.8),
                )
            )

    return out
