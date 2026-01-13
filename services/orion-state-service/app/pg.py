from __future__ import annotations

import re
from typing import Any, Dict, List

import asyncpg


_SAFE_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
def _safe_ident(name: str) -> str:
    if not _SAFE_IDENT.match(name or ""):
        raise ValueError(f"unsafe_identifier:{name}")
    return name


async def fetch_recent_spark_telemetry_metadata(
    *,
    postgres_uri: str,
    table: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch recent spark_telemetry rows (metadata JSON) newest-first."""
    t = _safe_ident(table)
    sql = f"SELECT metadata FROM {t} ORDER BY timestamp DESC NULLS LAST LIMIT $1"

    conn = await asyncpg.connect(dsn=postgres_uri)
    try:
        rows = await conn.fetch(sql, int(limit))
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = r.get("metadata") or {}
            if isinstance(meta, dict):
                out.append(meta)
        return out
    finally:
        await conn.close()
