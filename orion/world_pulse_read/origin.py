"""How a read started: the chain of reads that led to it, from the queue's own rows.

A read's ``request_json.parent_request_id`` names the read whose Stage 2 asked for it
(``need_stage1_urls``, checked against that read's own links by ``hop_refusal_reason``); the
chain ends at a read with no parent -- a world-pulse digest item or finding, a Juniper request,
or a curiosity run's request. The reading-finding prompt shows this chain so Orion describes how
a read began from the record (live 2026-10-26: a prompt that said "nobody asked you to read it"
produced "not because a process scheduled it" about a read a scheduled digest had started).

Read-only SELECT. Producer and consumer both run in Hub.
"""
from __future__ import annotations

from typing import Any

from orion.schemas.reading_finding import ReadingOriginStepV1

# Re-entry is capped per read (HUB_WORLD_PULSE_READ_MAX_ROUND_TRIPS), but a chain can hop many
# times; the cap only stops a malformed cycle from recursing without end.
MAX_ORIGIN_DEPTH = 12

ORIGIN_SQL = """
WITH RECURSIVE chain AS (
    SELECT s.seed_id AS leaf, 0 AS depth, s.seed_id, s.kind, s.url, s.title, s.request_json
    FROM world_pulse_read_seed s
    WHERE s.seed_id = ANY($1::text[])
  UNION ALL
    SELECT c.leaf, c.depth + 1, p.seed_id, p.kind, p.url, p.title, p.request_json
    FROM chain c
    JOIN world_pulse_read_seed p ON p.request_id = (c.request_json->>'parent_request_id')::uuid
    WHERE c.depth < $2
)
SELECT leaf, depth, seed_id, kind, url, title,
       request_json->>'requested_by' AS requested_by,
       request_json->>'invocation_context' AS invocation_context,
       request_json->>'why_now' AS why_now,
       request_json->>'parent_request_id' AS parent_request_id
FROM chain
ORDER BY leaf, depth DESC
"""


def origins_from_rows(rows: list[Any]) -> dict[str, list[ReadingOriginStepV1]]:
    """Rows of ``ORIGIN_SQL`` -> per read, its chain oldest first, ending with the read. Pure."""
    out: dict[str, list[ReadingOriginStepV1]] = {}
    for r in rows:
        out.setdefault(r["leaf"], []).append(ReadingOriginStepV1(
            seed_id=r["seed_id"], kind=r["kind"], url=r["url"] or "", title=r["title"] or "",
            requested_by=r["requested_by"] or "", invocation_context=r["invocation_context"] or "",
            why_now=" ".join((r["why_now"] or "").split()),
            parent_request_id=r["parent_request_id"] or "",
        ))
    return out


async def read_origins(conn: Any, seed_ids: list[str]) -> dict[str, list[ReadingOriginStepV1]]:
    if not seed_ids:
        return {}
    return origins_from_rows(await conn.fetch(ORIGIN_SQL, sorted(set(seed_ids)), MAX_ORIGIN_DEPTH))
