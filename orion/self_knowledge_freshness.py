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

``NEWEST_SELF_CONCEPT_REFLECTION_SQL`` (2026-09-20) is the same pattern for
``self_concept_history`` rows with ``produced_by='layer3_reflect'`` --
currently only ``orion-cortex-exec``'s Layer 3 reflect timer reads it, but it
lives here rather than inlined in that service so a second reader (e.g.
orion-hub, if Self Atlas's own freshness guard ever needs to distinguish
mechanical clustering from real reflection) does not have to re-derive it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

NEWEST_SELF_KNOWLEDGE_ITEM_SQL = "SELECT MAX(created_at) AS newest FROM self_knowledge_items"

NEWEST_SELF_CONCEPT_REFLECTION_SQL = (
    "SELECT MAX(created_at) AS newest FROM self_concept_history WHERE produced_by = 'layer3_reflect'"
)


def normalize_newest(newest: Optional[datetime]) -> Optional[datetime]:
    """Stamp a naive timestamp as UTC; pass a tz-aware one or ``None`` through."""
    if newest is None:
        return None
    if newest.tzinfo is None:
        return newest.replace(tzinfo=timezone.utc)
    return newest
