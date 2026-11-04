"""Phase D — episode grounding for reverie (read-only).

Grounds a spontaneous thought in more than *now*: recent rung-4 episode summaries
(episode ids). Layer 11 consolidation motifs were the other half until the
consolidation rung was killed (2026-11-04). This is a quality
lever, strictly read-only — reverie reads these different-kind artifacts and
never writes them (type asymmetry: it can't feed cognition its own kind).

Every loader degrades to an empty list on absent/unavailable input — never
raises. Refs are capped (§ cap-all-collections).
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("orion-thought.grounding")

DEFAULT_GROUNDING_CAP = 5

RefLoader = Callable[[int], list[str]]


def collect_grounding(*, episode_loader: RefLoader, cap: int = DEFAULT_GROUNDING_CAP) -> list[str]:
    """Return episode_summary_refs, capped and read-only ([] on any loader error)."""
    cap = max(0, int(cap))
    try:
        episodes = list(episode_loader(cap))[:cap]
    except Exception:
        episodes = []
    return [str(e) for e in episodes]


def _select_ids(sql: str, limit: int) -> list[str]:
    from .store import _get_engine  # local: DB is optional at read time

    from sqlalchemy import text

    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
    return [str(next(iter(r.values()))) for r in rows]


def default_episode_loader(limit: int) -> list[str]:
    """Recent rung-4 episode ids (best-effort, [] on error)."""
    try:
        return _select_ids(
            "SELECT episode_id FROM substrate_episode_summaries "
            "ORDER BY window_end DESC LIMIT :limit",
            limit,
        )
    except Exception as exc:
        logger.debug("episode grounding load failed: %s", exc)
        return []
