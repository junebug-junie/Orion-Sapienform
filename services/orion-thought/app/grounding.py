"""Phase D — episode + motif grounding for reverie (read-only).

Grounds a spontaneous thought in more than *now*: recent Layer 11 consolidation
motifs (frame ids) and rung-4 episode summaries (episode ids). This is a quality
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

# (motif_refs, episode_summary_refs)
RefLoader = Callable[[int], list[str]]


def collect_grounding(
    *,
    motif_loader: RefLoader,
    episode_loader: RefLoader,
    cap: int = DEFAULT_GROUNDING_CAP,
) -> tuple[list[str], list[str]]:
    """Return (motif_refs, episode_summary_refs), each capped and read-only."""
    cap = max(0, int(cap))
    try:
        motifs = list(motif_loader(cap))[:cap]
    except Exception:
        motifs = []
    try:
        episodes = list(episode_loader(cap))[:cap]
    except Exception:
        episodes = []
    return [str(m) for m in motifs], [str(e) for e in episodes]


def _select_ids(sql: str, limit: int) -> list[str]:
    from .store import _get_engine  # local: DB is optional at read time

    from sqlalchemy import text

    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
    return [str(next(iter(r.values()))) for r in rows]


def default_motif_loader(limit: int) -> list[str]:
    """Recent Layer 11 consolidation frame ids (best-effort, [] on error)."""
    try:
        return _select_ids(
            "SELECT frame_id FROM substrate_consolidation_frames "
            "ORDER BY generated_at DESC LIMIT :limit",
            limit,
        )
    except Exception as exc:
        logger.debug("motif grounding load failed: %s", exc)
        return []


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
