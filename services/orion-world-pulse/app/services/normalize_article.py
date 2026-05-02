from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Mapping

from orion.schemas.world_pulse import ArticleRecordV1, WorldPulseSourceV1


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_article(*, run_id: str, source: WorldPulseSourceV1, item: Mapping[str, Any], fetched_at: datetime) -> ArticleRecordV1:
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or "").strip()
    excerpt = str(item.get("summary") or item.get("excerpt") or "").strip()
    normalized_blob = f"{title.lower()}|{url.lower()}|{excerpt[:256].lower()}"
    content_hash = _sha(normalized_blob)
    article_id = f"article:{content_hash[:16]}"
    return ArticleRecordV1(
        article_id=article_id,
        run_id=run_id,
        source_id=source.source_id,
        source_name=source.name,
        url=url,
        canonical_url=url,
        title=title,
        author=item.get("author"),
        published_at=item.get("published_at"),
        fetched_at=fetched_at,
        text_excerpt=excerpt[:1000] if excerpt else None,
        normalized_text_hash=_sha(f"{title.lower()}|{excerpt.lower()}"),
        content_hash=content_hash,
        categories=list(source.categories),
        region_scope=source.region_scope,
        source_trust_tier=source.trust_tier,
        allowed_uses=source.allowed_uses,
        dedupe_key=f"{source.source_id}:{content_hash[:24]}",
        extraction_status="normalized",
        provenance={"source_url": source.url, "strategy": source.strategy},
        raw_metadata=dict(item.get("metadata") or {}),
    )
