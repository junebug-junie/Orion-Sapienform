from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import asyncpg

from orion.core.contracts.memory_cards import visibility_allows_card

logger = logging.getLogger("orion-recall.cards")

_EXCLUDED_STATUS = frozenset({"rejected", "archived", "deprecated"})
_NEIGHBOR_TYPES = frozenset({"relates_to", "child_of", "supports"})


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower()) if t}


def _neighbor_confidence_ok(meta: Dict[str, Any]) -> bool:
    raw = (meta or {}).get("confidence") if isinstance(meta, dict) else None
    if raw is None:
        return True
    order = ("uncertain", "possible", "likely", "certain")
    try:
        return order.index(str(raw).lower()) >= order.index("likely")
    except ValueError:
        return True


async def fetch_card_fragments(
    pool: asyncpg.Pool,
    fragment: str,
    profile: Dict[str, Any],
    *,
    lane: Optional[str],
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    """Score memory cards for fusion; returns candidate dicts (source=cards)."""
    top_k = int(profile.get("cards_top_k", 0) or 0)
    if top_k <= 0:
        return []

    q_tokens = _tokens(fragment)
    if not q_tokens:
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT card_id, slug, types, anchor_class, status, title, summary, tags, anchors,
                   visibility_scope, evidence, updated_at
            FROM memory_cards
            WHERE NOT (status = ANY($1::text[]))
            """,
            list(_EXCLUDED_STATUS),
        )

    scored: List[tuple[float, asyncpg.Record]] = []
    for row in rows:
        scopes = list(row["visibility_scope"] or [])
        if not visibility_allows_card(scopes, lane):
            continue
        title = str(row["title"] or "")
        summary = str(row["summary"] or "")
        tags = [str(t).lower() for t in (row["tags"] or []) if t]
        anchors = [str(a).lower() for a in (row["anchors"] or []) if a]
        ac = str(row["anchor_class"] or "").lower()
        types = [str(t).lower() for t in (row["types"] or [])]

        score = 0.0
        for tok in anchors:
            if tok in q_tokens:
                score += 2.0
        for tok in _tokens(title):
            if tok in q_tokens:
                score += 1.0
        for tok in _tokens(summary):
            if tok in q_tokens:
                score += 0.5
        for tag in tags:
            if tag in q_tokens:
                score += 0.3
        if ac and ac in q_tokens:
            score += 2.0
        for t in types:
            if t in q_tokens:
                score += 0.5

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: (-x[0], str(x[1]["updated_at"] or "")))
    top = scored[:top_k]

    out: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    async with pool.acquire() as conn:
        for base_score, row in top:
            cid = row["card_id"]
            cid_str = str(cid)
            if cid_str in seen_ids:
                continue
            seen_ids.add(cid_str)
            type0 = (row["types"] or ["fact"])[0]
            anchor_label = row["anchor_class"] or type0
            text = f"[card:{anchor_label}] {row['title']} — {row['summary']}"
            out.append(
                {
                    "id": f"card:{cid_str}",
                    "source": "cards",
                    "source_ref": str(row["slug"]),
                    "text": text,
                    "snippet": text,
                    "ts": row["updated_at"].timestamp() if row["updated_at"] else None,
                    "tags": ["memory_card", f"slug:{row['slug']}"],
                    "score": min(1.0, base_score / 8.0),
                    "meta": {"card_id": cid_str},
                }
            )

            n_rows = await conn.fetch(
                """
                SELECT e.edge_type, e.metadata, e.to_card_id, e.from_card_id,
                       c.card_id, c.slug, c.types, c.anchor_class, c.title, c.summary,
                       c.tags, c.anchors, c.visibility_scope, c.status, c.updated_at
                FROM memory_card_edges e
                JOIN memory_cards c
                  ON (e.from_card_id = $1 AND c.card_id = e.to_card_id)
                  OR (e.to_card_id = $1 AND c.card_id = e.from_card_id)
                WHERE (e.from_card_id = $1 OR e.to_card_id = $1)
                  AND e.edge_type = ANY($2::text[])
                  AND c.card_id <> $1
                  AND NOT (c.status = ANY($3::text[]))
                """,
                cid,
                list(_NEIGHBOR_TYPES),
                list(_EXCLUDED_STATUS),
            )

            n_added = 0
            for nr in n_rows:
                if n_added >= max_neighbors:
                    break
                meta = nr["metadata"] if isinstance(nr["metadata"], dict) else {}
                if not _neighbor_confidence_ok(meta):
                    continue
                scopes = list(nr["visibility_scope"] or [])
                if not visibility_allows_card(scopes, lane):
                    continue
                nid = str(nr["card_id"])
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                type0n = (nr["types"] or ["fact"])[0]
                anchor_label_n = nr["anchor_class"] or type0n
                ntext = f"[card:{anchor_label_n}] {nr['title']} — {nr['summary']}"
                out.append(
                    {
                        "id": f"card:{nid}",
                        "source": "cards",
                        "source_ref": str(nr["slug"]),
                        "text": ntext,
                        "snippet": ntext,
                        "ts": nr["updated_at"].timestamp() if nr["updated_at"] else None,
                        "tags": ["memory_card", "neighbor", f"edge:{nr['edge_type']}"],
                        "score": min(1.0, (base_score * 0.5) / 8.0),
                        "meta": {"card_id": nid, "neighbor_of": cid_str},
                    }
                )
                n_added += 1

    return out


async def fetch_card_fragments_guarded(
    pool: Optional[asyncpg.Pool],
    fragment: str,
    profile: Dict[str, Any],
    *,
    lane: Optional[str],
    timeout_sec: float,
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    if pool is None:
        return []
    try:
        return await asyncio.wait_for(
            fetch_card_fragments(pool, fragment, profile, lane=lane, max_neighbors=max_neighbors),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("cards backend timeout after %ss", timeout_sec)
        return []
    except Exception as exc:
        logger.warning("cards backend error: %s", exc)
        return []
