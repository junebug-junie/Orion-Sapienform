from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from orion.core.contracts.memory_cards import visibility_allows_card

logger = logging.getLogger("orion-recall.cards")

_NEIGHBOR_TYPES = frozenset({"relates_to", "child_of", "supports"})

# always_inject cards are a small, foundational, zero-latency tier -- they
# bypass relevance scoring entirely (see fetch_card_fragments below). This is
# a sanity guard, not a truncation-by-design the way the old
# RECALL_CARDS_CANDIDATE_LIMIT was: these rows are meant to be few.
_ALWAYS_INJECT_LIMIT = 10

# ts_rank_cd's weight array is ordered {D,C,B,A} (see Postgres docs on
# ts_rank_cd), and Postgres rejects any weight outside [0,1] ("weight out of
# range", confirmed live). Mapped from the original 2026-05-01 design's
# per-field ratios (anchor +2.0, title +1.0, summary +0.5, tag +0.3),
# normalized to fit that range by dividing by the max (2.0): A=anchor(1.0),
# B=title(0.5), C=summary(0.25), D=tag(0.15). Uniformly rescaling every
# weight by the same constant does not change relative ranking between rows
# (ts_rank_cd's score is linear in the weight array), so this preserves the
# original design's 2.0:1.0:0.5:0.3 ratio exactly while satisfying Postgres's
# constraint. Matches the search_vector generated column's setweight()
# labels in sql/memory_cards.sql.
_TS_RANK_WEIGHTS = "{0.15,0.25,0.5,1.0}"

_CONFIDENCE_MULTIPLIER_SQL = """
    CASE confidence
        WHEN 'certain' THEN 1.0
        WHEN 'likely' THEN 0.8
        WHEN 'possible' THEN 0.5
        WHEN 'uncertain' THEN 0.3
        ELSE 0.5
    END
"""

# Excludes cards whose era_bound time_horizon.end has passed -- a gate, not a
# down-rank (a superseded fact should be gone from "current fact" results,
# not sitting at #4). Written as a CASE (not a WHERE-clause AND chain)
# because CASE branches are guaranteed to evaluate in order and stop at the
# first match, so the timestamptz cast on the ELSE branch only ever runs
# once kind/end/format have all been confirmed safe -- a malformed
# time_horizon.end value can never reach the cast and 500 the whole query.
_TEMPORAL_VALIDITY_SQL = """
    NOT (
        CASE
            WHEN time_horizon IS NULL THEN false
            WHEN time_horizon ->> 'kind' IS DISTINCT FROM 'era_bound' THEN false
            WHEN (time_horizon ->> 'end') IS NULL THEN false
            WHEN (time_horizon ->> 'end') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN false
            ELSE (time_horizon ->> 'end')::timestamptz < now()
        END
    )
"""

# Same guard, qualified for the neighbor-expansion query below (table alias
# `c`) -- an expired era_bound neighbor must be excluded there too, not just
# from the primary always_inject/ranked queries. Derived by substitution
# rather than duplicated so the two guards can never drift out of sync.
_TEMPORAL_VALIDITY_SQL_QUALIFIED = _TEMPORAL_VALIDITY_SQL.replace("time_horizon", "c.time_horizon")

_CARD_COLUMNS = """
    card_id, slug, types, anchor_class, status, title, summary, tags, anchors,
    visibility_scope, evidence, subschema, updated_at, confidence, priority, time_horizon
"""


def _neighbor_confidence_ok(meta: Dict[str, Any]) -> bool:
    raw = (meta or {}).get("confidence") if isinstance(meta, dict) else None
    if raw is None:
        return True
    order = ("uncertain", "possible", "likely", "certain")
    try:
        return order.index(str(raw).lower()) >= order.index("likely")
    except ValueError:
        return True


async def _fetch_always_inject_cards(pool: asyncpg.Pool) -> List[asyncpg.Record]:
    """always_inject cards: no relevance scoring, no candidate window --
    every call fetches the (small) live set of foundational always-on cards."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            f"""
            SELECT {_CARD_COLUMNS}
            FROM memory_cards
            WHERE priority = 'always_inject' AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT {_ALWAYS_INJECT_LIMIT}
            """
        )


async def _fetch_ranked_cards(
    pool: asyncpg.Pool, query_text: str, top_k: int
) -> List[Tuple[float, asyncpg.Record]]:
    """Structured, weighted full-text scoring (Item 1c). No candidate-window
    LIMIT before scoring -- ranking runs over the GIN-indexed search_vector
    column (scales with the number of MATCHING rows, not corpus size); the
    only LIMIT is on the final ranked output, applied after ORDER BY score
    DESC."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {_CARD_COLUMNS},
                   ts_rank_cd('{_TS_RANK_WEIGHTS}'::float4[], search_vector, query)
                   * ({_CONFIDENCE_MULTIPLIER_SQL}) AS score
            FROM memory_cards, plainto_tsquery('english', $1) AS query
            WHERE status = 'active'
              AND search_vector @@ query
              AND {_TEMPORAL_VALIDITY_SQL}
            ORDER BY score DESC
            LIMIT $2
            """,
            query_text,
            top_k,
        )
    return [(float(r["score"]), r) for r in rows]


async def fetch_card_fragments(
    pool: asyncpg.Pool,
    fragment: str,
    profile: Dict[str, Any],
    *,
    lane: Optional[str],
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    """Score memory cards for fusion via structured, graded, temporally-honest
    retrieval (source=cards) -- see docs/superpowers/specs/
    2026-07-21-memory-cards-self-authored-substrate-spec.md Item 1c. Replaces
    the live-embedding cosine path (cards_embedding.py, deleted) entirely:
    always_inject cards bypass scoring, everything else is ranked by
    Postgres full-text search over a weighted tsvector times a confidence
    multiplier, gated by temporal validity."""
    top_k = int(profile.get("cards_top_k", 0) or 0)
    if top_k <= 0:
        return []

    query_text = (fragment or "").strip()
    if not query_text:
        return []

    always_inject_rows = await _fetch_always_inject_cards(pool)
    ranked = await _fetch_ranked_cards(pool, query_text, top_k)

    visible_top: List[Tuple[float, asyncpg.Record]] = []
    seen_ids: set[str] = set()

    for row in always_inject_rows:
        if not visibility_allows_card(lane=lane, visibility_scope=list(row["visibility_scope"] or [])):
            continue
        cid_str = str(row["card_id"])
        if cid_str in seen_ids:
            continue
        seen_ids.add(cid_str)
        # always_inject cards are not competing for rank -- score them at the
        # ceiling so they never lose a fusion tie-break against a merely
        # highly-relevant ranked card.
        visible_top.append((1.0, row))

    for score, row in ranked:
        if not visibility_allows_card(lane=lane, visibility_scope=list(row["visibility_scope"] or [])):
            continue
        cid_str = str(row["card_id"])
        if cid_str in seen_ids:
            continue
        seen_ids.add(cid_str)
        visible_top.append((score, row))

    out: List[Dict[str, Any]] = []

    async with pool.acquire() as conn:
        for base_score, row in visible_top:
            cid = row["card_id"]
            cid_str = str(cid)
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
                    "score": min(1.0, float(base_score)),
                    "meta": {
                        "card_id": cid_str,
                        "relevance_score": float(base_score),
                        "confidence": row["confidence"],
                        "priority": row["priority"],
                    },
                }
            )

            n_rows = await conn.fetch(
                f"""
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
                  AND c.status = 'active'
                  AND {_TEMPORAL_VALIDITY_SQL_QUALIFIED}
                """,
                cid,
                list(_NEIGHBOR_TYPES),
            )

            n_added = 0
            for nr in n_rows:
                if n_added >= max_neighbors:
                    break
                meta = nr["metadata"] if isinstance(nr["metadata"], dict) else {}
                if not _neighbor_confidence_ok(meta):
                    continue
                scopes = list(nr["visibility_scope"] or [])
                if not visibility_allows_card(lane=lane, visibility_scope=scopes):
                    continue
                nid = str(nr["card_id"])
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                type0n = (nr["types"] or ["fact"])[0]
                anchor_label_n = nr["anchor_class"] or type0n
                ntext = f"[card:{anchor_label_n}] {nr['title']} — {nr['summary']}"
                neighbor_sim = float(base_score) * 0.5
                out.append(
                    {
                        "id": f"card:{nid}",
                        "source": "cards",
                        "source_ref": str(nr["slug"]),
                        "text": ntext,
                        "snippet": ntext,
                        "ts": nr["updated_at"].timestamp() if nr["updated_at"] else None,
                        "tags": ["memory_card", "neighbor", f"edge:{nr['edge_type']}"],
                        "score": min(1.0, neighbor_sim),
                        "meta": {"card_id": nid, "neighbor_of": cid_str, "similarity": neighbor_sim},
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
