"""Fresh endogenous curiosity candidates for Hub readers.

This module used to also prepend a "[curiosity focus]" hint line to the
context-exec Agent lane's prompt; that lane and orion-context-exec were retired
2026-10-10, so only the candidate reader remains. Its live consumer is
endogenous_outreach.py (``_fetch_fresh_candidates`` / ``usable_candidates``).
"""

from __future__ import annotations

import json
import os
from typing import Any

_MAX_AGE_SEC = 120.0


def _fetch_fresh_candidates(*, max_age_sec: float = _MAX_AGE_SEC) -> list[dict[str, Any]]:
    """Latest curiosity candidate set newer than ``max_age_sec``, else [].

    Callers on a slower cadence (e.g. endogenous_outreach) widen ``max_age_sec``
    rather than duplicating this query.
    """
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return []
    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT candidates_json FROM substrate_endogenous_curiosity_candidates
                        WHERE generated_at >= now() - make_interval(secs => :max_age)
                        ORDER BY generated_at DESC LIMIT 1
                        """
                    ),
                    {"max_age": float(max_age_sec)},
                )
                .mappings()
                .first()
            )
    finally:
        engine.dispose()
    if not row:
        return []
    return usable_candidates(row["candidates_json"])


def usable_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Stored candidates_json -> the dict candidates a hint/outreach may show.

    Unscored event seeds (an accepted reading link, note strength:unscored_event)
    are not "gaps" and have no strength to rank by: they stay out of the agent
    hint and outreach topics. Orion still sees them in raw self-inquiry rows.
    """
    from orion.core.schemas.frontier_curiosity import is_unscored_event

    if isinstance(candidates, str):
        candidates = json.loads(candidates)
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, dict) and not is_unscored_event(c.get("notes"))]
