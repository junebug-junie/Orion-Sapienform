from __future__ import annotations

from orion.schemas.world_pulse import ArticleRecordV1


def dedupe_articles(items: list[ArticleRecordV1]) -> list[ArticleRecordV1]:
    seen: set[str] = set()
    out: list[ArticleRecordV1] = []
    for item in items:
        if item.dedupe_key in seen:
            continue
        seen.add(item.dedupe_key)
        out.append(item)
    return out
