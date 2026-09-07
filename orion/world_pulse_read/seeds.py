from __future__ import annotations

import hashlib
from typing import Any

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def make_seed_id(
    *,
    kind: str,
    run_id: str,
    url: str,
    item_id: str | None = None,
) -> str:
    h = _url_hash(url)
    if kind == "digest_item":
        if not item_id:
            raise ValueError("digest_item seed_id requires item_id")
        return f"digest_item:{run_id}:{item_id}:{h}"
    if kind == "finding":
        return f"finding:{run_id}:{h}"
    raise ValueError(f"unknown seed kind: {kind}")


def _http_url(value: str) -> str | None:
    v = (value or "").strip()
    if v.startswith("http://") or v.startswith("https://"):
        return v
    return None


def seeds_from_digest_payload(
    payload: dict[str, Any],
    *,
    article_urls: dict[str, str] | None = None,
) -> list[WorldPulseReadSeedV1]:
    """Findings first, then digest items. Skip rows with no resolvable URL."""
    article_urls = article_urls or {}
    run_id = str(payload.get("run_id") or "")
    out: list[WorldPulseReadSeedV1] = []

    for followup in payload.get("curiosity_followups") or []:
        section = str(followup.get("section") or "")
        for art in followup.get("articles") or []:
            url = _http_url(str(art.get("url") or ""))
            if not url or not run_id:
                continue
            out.append(
                WorldPulseReadSeedV1(
                    seed_id=make_seed_id(kind="finding", run_id=run_id, url=url),
                    kind="finding",
                    run_id=run_id,
                    url=url,
                    title=str(art.get("title") or ""),
                    section=section,
                )
            )

    for item in payload.get("items") or []:
        item_id = str(item.get("item_id") or "")
        if not item_id or not run_id:
            continue
        url = None
        for wr in item.get("worth_reading") or []:
            url = _http_url(str(wr))
            if url:
                break
        if url is None:
            for aid in item.get("article_ids") or []:
                mapped = article_urls.get(str(aid))
                if mapped:
                    url = _http_url(mapped)
                    if url:
                        break
        if not url:
            continue
        out.append(
            WorldPulseReadSeedV1(
                seed_id=make_seed_id(
                    kind="digest_item", run_id=run_id, url=url, item_id=item_id
                ),
                kind="digest_item",
                run_id=run_id,
                url=url,
                title=str(item.get("title") or ""),
                section=str(item.get("category") or ""),
                item_id=item_id,
            )
        )
    return out
