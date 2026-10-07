"""Read-only queries the situation graph runs against ``episode_memory`` (no graph reads, no LLM).

Both run on the service's existing psycopg pool (autocommit, dict rows) and are bounded: facts are
only rows whose window can still cover now, priming joins on referent keys only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

FACTS_SQL = """
SELECT m.memory_id::text AS memory_id, m.purpose, m.statement, m.occurred_at, m.created_at,
       m.expires_at, m.voice, m.confirmation_state,
       COALESCE(json_agg(json_build_object('key', r.referent_key, 'role', r.role))
                FILTER (WHERE r.referent_key IS NOT NULL), '[]') AS referents
FROM episode_memory m
LEFT JOIN episode_memory_referent r USING (memory_id)
WHERE m.status = 'active'
  AND m.confirmation_state NOT IN ('rejected', 'corrected')
  AND (m.expires_at > %(now)s
       OR (m.expires_at IS NULL AND m.purpose = 'happened'
           AND COALESCE(m.occurred_at, m.created_at) > %(since)s))
GROUP BY m.memory_id
ORDER BY COALESCE(m.occurred_at, m.created_at) DESC
LIMIT 200
"""

KNOWN_KEYS_SQL = "SELECT DISTINCT referent_key FROM episode_memory_referent"

PRIME_SQL = """
SELECT m.memory_id::text AS memory_id, m.statement, m.voice, m.confirmation_state, m.strength,
       m.half_life_days, m.last_reinforced_at, m.created_at,
       array_agg(DISTINCT r.referent_key) AS referent_keys
FROM episode_memory m
JOIN episode_memory_referent r USING (memory_id)
WHERE r.referent_key = ANY(%(cues)s)
  AND m.status = 'active'
  AND m.confirmation_state NOT IN ('rejected', 'corrected')
  AND NOT (m.memory_id::text = ANY(%(exclude)s))
GROUP BY m.memory_id
ORDER BY max(m.strength) DESC, max(m.last_reinforced_at) DESC
LIMIT %(limit)s
"""


async def load_facts(pool: Any, now: datetime, default_ttl: timedelta) -> dict:
    async with pool.connection() as conn:
        rows = await (await conn.execute(FACTS_SQL, {"now": now, "since": now - default_ttl})).fetchall()
        keys = await (await conn.execute(KNOWN_KEYS_SQL)).fetchall()
    return {"rows": [dict(r) for r in rows], "known_keys": [r["referent_key"] for r in keys]}


async def prime(pool: Any, cues: list[str], exclude: list[str], limit: int) -> list[dict]:
    async with pool.connection() as conn:
        rows = await (await conn.execute(PRIME_SQL, {"cues": list(cues), "exclude": list(exclude), "limit": int(limit)})).fetchall()
    return [dict(r) for r in rows]
