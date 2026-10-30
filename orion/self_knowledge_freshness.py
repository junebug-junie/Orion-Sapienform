"""Freshness queries two or more services independently need to agree on.

``orion-cortex-exec`` (``app/self_study_refresh.py``) reads
``NEWEST_SELF_KNOWLEDGE_ITEM_SQL`` to decide whether its own daily fact-scan
is due; ``orion-hub`` (``scripts/self_atlas_cluster_history.py``) reads the
same constant to refuse re-clustering a table Self Atlas has already trained
on. Both are the same real check (2026-09-19), so the SQL and timestamp
normalization live here, once -- each service keeps its own engine/connection
(different DSN resolution and pooling per CLAUDE.md's service-boundary rule;
this module never opens a connection itself, only tells a caller's own
connection what to run and how to read the result back).

The Layer 3 reflection freshness query that lived here was retired on 2026-10-30
with the reflect loop itself (fold into curiosity).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

NEWEST_SELF_KNOWLEDGE_ITEM_SQL = "SELECT MAX(created_at) AS newest FROM self_knowledge_items"

def normalize_newest(newest: Optional[datetime]) -> Optional[datetime]:
    """Stamp a naive timestamp as UTC; pass a tz-aware one or ``None`` through."""
    if newest is None:
        return None
    if newest.tzinfo is None:
        return newest.replace(tzinfo=timezone.utc)
    return newest
