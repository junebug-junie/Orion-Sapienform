"""Phase F store — read the compaction-request queue, persist staged deltas.

Two seams only:
  - `load_pending_requests`: drains recent Phase-E compaction asks (read-only);
  - `persist_compaction_delta`: writes the *staged* delta to `dream_compaction_delta`.

Both are best-effort and never raise. Critically, this module writes to exactly
one table (`dream_compaction_delta`, a staging table) and touches **no canonical
memory** — the "applies nothing" invariant of Phase F lives here by construction.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.settings import settings

logger = logging.getLogger("orion-dream.rem_store")

if TYPE_CHECKING:
    from orion.schemas.compaction import MemoryCompactionDeltaV1

_engine = None

# The only table this store is permitted to write. Kept as a named constant so a
# test can assert Phase F never widens its write surface into canonical memory.
STAGING_WRITE_TABLE = "dream_compaction_delta"


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(settings.POSTGRES_URI, pool_pre_ping=True)
    return _engine


def load_pending_requests(limit: int) -> list[dict]:
    """Recent un-consumed compaction requests (Phase-E queue). [] on any miss.

    Read-only. Returns raw request_json dicts so the caller owns interpretation.
    """
    limit = max(0, int(limit))
    if limit == 0:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT request_json FROM dream_compaction_request_queue
                        WHERE consumed_at IS NULL
                        ORDER BY created_at DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        out: list[dict] = []
        for r in rows:
            payload = r["request_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                out.append(payload)
        return out
    except Exception as exc:
        logger.debug("rem: pending request load failed: %s", exc)
        return []


def persist_compaction_delta(delta: "MemoryCompactionDeltaV1") -> bool:
    """Insert one staged delta. Never raises; idempotent on delta_id.

    Writes ONLY the staging table — no canonical memory is touched in Phase F.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dream_compaction_delta
                        (delta_id, dream_id, created_at, cards_out, edges_downscaled,
                         rows_pruned, bytes_reclaimed_est, proposal_marked, delta_json)
                    VALUES
                        (:delta_id, :dream_id, :created_at, :cards_out, :edges_downscaled,
                         :rows_pruned, :bytes_reclaimed_est, :proposal_marked,
                         CAST(:delta_json AS jsonb))
                    ON CONFLICT (delta_id) DO NOTHING
                    """
                ),
                {
                    "delta_id": delta.delta_id,
                    "dream_id": delta.dream_id,
                    "created_at": delta.created_at,
                    "cards_out": delta.metrics.cards_out,
                    "edges_downscaled": delta.metrics.edges_downscaled,
                    "rows_pruned": delta.metrics.rows_pruned,
                    "bytes_reclaimed_est": delta.metrics.bytes_reclaimed_est,
                    "proposal_marked": bool(delta.proposal_marked),
                    "delta_json": json.dumps(delta.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("rem: delta persist failed id=%s err=%s", delta.delta_id, exc)
        return False
