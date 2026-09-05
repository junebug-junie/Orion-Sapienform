"""Durable per-turn log of chat_stance's real belief computation.

`services/orion-cortex-exec/app/chat_stance.py` computes a real
`UnifiedRelationalBeliefSetV1` every chat turn (anchors, degraded producers,
shift/repair lineage) and, until this schema existed, discarded it after the
turn -- confirmed live 2026-09-05, only a 4-key compact summary survived 30
minutes in an in-process cache (`executor.py`'s `_PRIOR_STANCE_CACHE`).

This is the durable write shape: one append-only row per turn. Feeds
self_study's Layer 1 `_behavioral_items()` (services/orion-cortex-exec/app/
self_study.py) -- Orion's own conversational behavior as a real self-fact,
not a redacted/aggregated stub. Per Juniper's explicit 2026-09-05 call, this
carries real content (she is the sole user and companion; the only guard
kept is the existing source_ref-allowlist pattern that stops personal
content leaking into an unrelated downstream consumer, not redaction from
her).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ChatStanceShiftKind = Literal["NONE", "TOPIC", "STANCE", "REPAIR"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatStanceBeliefLogV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    correlation_id: str | None = None
    session_id: str | None = None
    shift_kind: ChatStanceShiftKind | None = None
    # Real content, capped for storage sanity, not redacted -- see module
    # docstring. Which anchors/concepts were live this turn, in plain text.
    anchor_summary: str | None = None
    degraded_producers: list[str] = Field(default_factory=list)
    lineage_summary: str | None = None
