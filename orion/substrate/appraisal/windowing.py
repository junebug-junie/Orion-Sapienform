"""Select the recent chat-window molecules to feed the repair appraiser."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from orion.substrate.molecules import SubstrateMoleculeV1


def select_recent_chat_molecules(
    molecules: Iterable[SubstrateMoleculeV1],
    *,
    source_id: str | None = None,
    max_age_seconds: int = 300,
    max_count: int = 20,
) -> list[SubstrateMoleculeV1]:
    """Return at most ``max_count`` molecules in newest-first order.

    Rules (spec §12):
    - If ``source_id`` is provided, keep only molecules whose
      ``provenance['source_id']`` matches.
    - Drop molecules older than ``max_age_seconds``.
    - Sort by ``created_at`` descending.
    - Cap at ``max_count``.
    """

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=max_age_seconds)
    keep: list[SubstrateMoleculeV1] = []
    for m in molecules:
        if source_id is not None and m.provenance.get("source_id") != source_id:
            continue
        if m.created_at < cutoff:
            continue
        keep.append(m)

    keep.sort(key=lambda m: m.created_at, reverse=True)
    return keep[:max_count]
