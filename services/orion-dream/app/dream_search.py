"""Search dreams by meaning. Index text per kind; Chroma is never the record.

Only rows from app.introspect_dreams.index_rows reach the index, so a
never-offered hypothesis is never embedded (blind experiment).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.introspect_dreams import doc_id
from orion.introspect.semantic_index import (
    IndexPass,
    SearchConfig,
    embed,
    index_docs,
    nearest,
)
from orion.schemas.introspect import clip_text

CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
_DOC_PREFIX = "dream-search"
_HASH_KEYS = ("kind", "occurred_ts")


def _epoch(value: datetime) -> float:
    """UTC epoch seconds; a naive timestamp is UTC (see introspect_dreams)."""
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).timestamp()


def document_text(kind: str, row: dict[str, Any]) -> str | None:
    if kind == "narrative":
        raw = row.get("themes") if isinstance(row.get("themes"), list) else []
        themes = [str(t).strip() for t in raw if str(t).strip()]
        parts = [
            str(row.get("tldr") or "").strip(),
            f"Themes: {', '.join(themes)}" if themes else "",
            str(row.get("narrative") or "").strip(),
        ]
    else:
        claim, why = str(row.get("claim") or "").strip(), str(row.get("why") or "").strip()
        if not claim:
            return None
        return clip_text(f"{claim}\nWhy: {why}" if why else claim, INDEX_TEXT_CHARS)[0]
    body = "\n\n".join(p for p in parts if p)
    return clip_text(body, INDEX_TEXT_CHARS)[0] if body else None


async def index_missing(
    pairs: list[tuple[str, dict[str, Any]]],
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    bus: Any,
    source: Any,
    batch: int | None = None,
) -> IndexPass:
    """Embed and upsert dreams the index lacks or holds stale text for."""
    docs = []
    for kind, row in pairs:
        text = document_text(kind, row)
        if text:
            occurred = row.get("occurred_at")
            meta: dict[str, Any] = {"kind": kind, "occurred_at": occurred.isoformat() if occurred else ""}
            if occurred:
                meta["occurred_ts"] = _epoch(occurred)
            docs.append((doc_id(kind, row), text, meta))
    # A re-offered hypothesis keeps its text but moves offered_at, so the
    # filtered fields are hashed too.
    return await index_docs(
        docs, cfg, client=client, bus=bus, source=source, doc_prefix=_DOC_PREFIX, batch=batch,
        hash_keys=_HASH_KEYS,
    )


def search_filter(kind: str | None, since: datetime | None) -> dict[str, Any] | None:
    """Chroma where-clause for kind/since, so filtering happens before the top-N cut."""
    clauses: list[dict[str, Any]] = []
    if kind is not None:
        clauses.append({"kind": kind})
    if since is not None:
        clauses.append({"occurred_ts": {"$gte": _epoch(since)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


async def rank(
    client: httpx.AsyncClient, cfg: SearchConfig, query: str,
    *, kind: str | None = None, since: datetime | None = None,
) -> list[tuple[str, float]]:
    """Embed the query once; (doc_id, similarity) at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query, doc_prefix=_DOC_PREFIX)
    hits = await nearest(client, cfg, vector, CANDIDATES, where=search_filter(kind, since))
    return [s for s in hits if s[1] >= cfg.min_similarity]
