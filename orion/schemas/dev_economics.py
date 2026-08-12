"""Bus payload schema for the dev-economics domain's
``orion:substrate:dev_economics_ledger`` channel
(docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md).

One event per real producer tick, aggregating the real GROWTH in every
Claude Code transcript's (top-level session or subagent dispatch)
cumulative totals since the last tick -- real token counts, real word
counts, and a real $ cost estimate priced against
``orion/dev_economics/pricing.py``'s explicitly versioned rate table.

Deliberately NOT window-membership filtering by session start time (the
first draft's approach, and ``JuniperAffectiveStateV1``'s own convention) --
a ``SessionUsageRecord`` is a cumulative snapshot of a whole transcript
file, not a discrete per-message event, so window-membership filtering
would silently under-report every session that outlives one poll tick (the
common case). See ``services/orion-cocreation-signals/app/producers/
dev_economics.py``'s ``dev_economics_loop`` docstring for the full story
(caught by code review 2026-08-12 before this ever shipped).
``window_since``/``window_until`` are both set to ``observed_at`` --
bookkeeping fields only, not a real window boundary, kept for schema shape
consistency with the other windowed signals in this program.

``total_estimated_cost_usd`` is ``None`` only when literally nothing in
this tick's real delta could be priced (no Anthropic-rate-table model
appears, or a date before that model's earliest known rate window) -- never
a fabricated $0.00 standing in for "some real spend happened, we just
couldn't price it." ``unpriced_session_count`` discloses exactly how much
of this tick's real activity that partial total excludes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DevEconomicsLedgerV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dev_economics_ledger.v1"] = "dev_economics_ledger.v1"
    # Real, stable per-event id for durable persistence (added when this
    # signal got its first real consumer -- orion-sql-writer's
    # DevEconomicsLedgerSQL table, docs/superpowers/pr-reports/2026-08-12-
    # dev-economics-hub-observability.md). Same convention as
    # DominanceStreakTickV1.tick_telemetry_id: a default_factory here means
    # every event already carries its own idempotency key, so a redelivered
    # message upserts safely instead of duplicating a row. Non-breaking
    # addition -- has a default, so any already-published event without it
    # still parses.
    event_id: str = Field(default_factory=lambda: f"dev-economics-{uuid4()}")
    observed_at: datetime
    window_since: datetime
    window_until: datetime

    session_count: int
    subagent_transcript_count: int

    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_input_tokens: int
    total_cache_read_input_tokens: int
    total_tokens: int

    total_assistant_word_count: int
    total_human_word_count: int

    total_estimated_cost_usd: float | None = None
    priced_session_count: int = 0
    unpriced_session_count: int = 0

    # Real model->count histogram for this window (e.g.
    # {"claude-sonnet-5": 3, "claude-haiku-4-5-20251001": 1}) -- never raw
    # transcript text, same privacy discipline as every other signal in
    # this program.
    model_mix: dict[str, int] = {}
