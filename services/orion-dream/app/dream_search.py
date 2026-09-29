"""Search dreams by meaning. Index text per kind; Chroma is never the record.

Only rows from app.introspect_dreams.index_rows reach the index, so a
never-offered hypothesis is never embedded (blind experiment).
"""
from __future__ import annotations

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
            docs.append((doc_id(kind, row), text, {
                "kind": kind, "occurred_at": occurred.isoformat() if occurred else "",
            }))
    return await index_docs(
        docs, cfg, client=client, bus=bus, source=source, doc_prefix=_DOC_PREFIX, batch=batch,
    )


async def rank(client: httpx.AsyncClient, cfg: SearchConfig, query: str) -> list[tuple[str, float]]:
    """Embed the query once; (doc_id, similarity) at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query, doc_prefix=_DOC_PREFIX)
    return [s for s in await nearest(client, cfg, vector, CANDIDATES) if s[1] >= cfg.min_similarity]
