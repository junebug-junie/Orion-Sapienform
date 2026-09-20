"""Compose hire role-teach progress_lines from independent advisory sources.

Each source fails open on its own: a bad hop-note read must not drop the
queue line, and a missing FieldState score must not drop denials/budget.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def build_role_teach_progress_lines(
    *,
    hop_note_texts: Sequence[str] = (),
    peer_brief_status: str | None = None,
    peer_brief_next_hop_n: int | None = None,
    queue_score: float | None = None,
    queue_driver: str | None = None,
) -> list[str]:
    """refusal + budget + queue progress lines (order fixed)."""
    lines: list[str] = []

    try:
        from orion.curiosity.access_refusals import count_access_refusals
        from orion.curiosity.role_teach_disclosure import format_access_refusal_progress

        lines.extend(
            format_access_refusal_progress(count_access_refusals(hop_note_texts))
        )
    except Exception:  # noqa: BLE001 — fail-open per source
        logger.debug("role_teach_refusal_progress_failed", exc_info=True)

    try:
        from orion.curiosity.role_teach_disclosure import format_budget_spent_progress

        if peer_brief_status is not None:
            lines.extend(
                format_budget_spent_progress(
                    status=str(peer_brief_status),
                    next_hop_n=peer_brief_next_hop_n,
                )
            )
    except Exception:  # noqa: BLE001 — fail-open per source
        logger.debug("role_teach_budget_progress_failed", exc_info=True)

    try:
        from orion.curiosity.queue_contention_disclosure import (
            format_queue_contention_progress,
        )

        lines.extend(format_queue_contention_progress(queue_score, queue_driver))
    except Exception:  # noqa: BLE001 — fail-open per source
        logger.debug("role_teach_queue_progress_failed", exc_info=True)

    return lines
