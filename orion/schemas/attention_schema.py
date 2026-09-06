"""AttentionSchemaV1 -- one thin, shared shape for "what is Orion attending
to right now, and why", emitted by every process in Orion that attends and
selects.

Design: docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md
(PR #2092). Attention Schema Theory's claim is that the attention schema is a
*lossy, simplified cartoon* of attention, not an accurate readout -- so this
model is deliberately narrow, and its narrowness is the theory, not a
compromise.

Four producers, one consumer, one table:

    orion-substrate-runtime  process="substrate_attention"  (every ~30s tick)
    orion-thought            process="reverie"              (per chain)
    orion-hub                process="curiosity"            (per investigation run)
    orion-cortex-exec        process="cortex_turn"          (per unified turn, human or self-initiated)
        -> bus channel ATTENTION_SCHEMA_CHANNEL
        -> orion-sql-writer -> table substrate_attention_schema

Each producer is a pure projection function ("adapter") that reads that
process's own existing artifact and returns one of these. No base class, no
plugin registry, no producer interface -- three or four functions that return
a value. See the design doc's "Adapters, not inheritance".

`attention_reason` IS NOT A SHARED ENUM AND MUST NEVER BECOME ONE. Each
process keeps its own vocabulary (substrate's `top_down_override`, reverie's
`coalition_broadcast`, curiosity's `tested_held_prior`, cortex's
`selected:ask`). The surface does not normalize, map, or reconcile them.
Cross-process comparison happens at rating time, by a rater reading the
narratives, not at schema time by a lookup table. Normalizing real
vocabularies into one taxonomy is the keyword-cathedral move the sentience
program exists to prevent.

`narrative_kind` exists because of the design doc's Missing Question 2: a
narrative *computed by code from data* (substrate, cortex, curiosity) and one
*written by Orion inside an LLM turn* (reverie's `interpretation`) are not the
same kind of object, and the blind-rater acceptance check needs to
stratify on that or it will score prose style instead of attention content.
Declared here rather than discovered later.

Write-only measurement surface. Nothing routes, gates, or budgets off it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

ATTENTION_SCHEMA_CHANNEL = "orion:attention:schema"
ATTENTION_SCHEMA_KIND = "attention.schema.v1"

AttentionSchemaProcessV1 = Literal[
    "substrate_attention",
    "reverie",
    "curiosity",
    "cortex_turn",
]

NarrativeKindV1 = Literal["computed", "self_report"]

MAX_LABEL_CHARS = 300
MAX_NARRATIVE_CHARS = 4000
MAX_PREDICTED_NEXT_CHARS = 1000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clip(text: object, limit: int) -> str:
    """Whitespace-normalised, hard-capped string for the bounded fields below.

    Adapters call this so a long upstream artifact (a reverie interpretation,
    a curiosity continue_note) degrades to a truncated row rather than a
    validation error that drops the row entirely.
    """
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


class AttentionSchemaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["attention.schema.v1"] = "attention.schema.v1"
    # Process-owned, deterministic per attention event (a substrate tick, a
    # reverie chain, a curiosity run, a chat turn) so a republish is an
    # ON CONFLICT no-op in the table rather than a duplicate row.
    entry_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=_utc_now)
    process: AttentionSchemaProcessV1
    correlation_id: str | None = None

    # --- what won -----------------------------------------------------------
    # None is a real, named state ("nothing won this tick"), distinct from
    # the row being absent. `attention_reason` says which kind of nothing.
    attended_id: str | None = None
    attended_label: str = Field(default="", max_length=MAX_LABEL_CHARS)

    # --- why it won -- process-owned vocabulary, never normalised ----------
    attention_reason: str = Field(min_length=1)
    reason_narrative: str = Field(default="", max_length=MAX_NARRATIVE_CHARS)
    narrative_kind: NarrativeKindV1 = "computed"

    # --- how confident, from a real signal, never invented -----------------
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_basis: str | None = None

    # --- what it expects to attend to next ---------------------------------
    predicted_next: str | None = Field(default=None, max_length=MAX_PREDICTED_NEXT_CHARS)


def bind_correlation(row: AttentionSchemaV1) -> tuple[AttentionSchemaV1, UUID]:
    """Return (row, envelope correlation id) with the two guaranteed equal.

    Review finding 2026-09-06: orion-sql-writer stamps every persisted row's
    `correlation_id` column from the *envelope* (`extra_sql_fields`, applied
    after payload validation), and `BaseEnvelope.correlation_id` defaults to
    a fresh uuid4. A producer that builds the envelope without passing one
    therefore gets a random id in the table, silently replacing the row's own
    (reverie thought, cortex turn, curiosity run) -- every join back to the
    originating artifact dead on arrival. Same reason
    orion/substrate/chat_stance_belief_bus.py passes `correlation_id=` on
    its envelope. Every producer of this schema calls this and passes the
    returned UUID on the envelope. A row with no usable id gets one minted
    here so payload and column still agree.
    """
    raw = row.correlation_id
    try:
        corr = UUID(str(raw)) if raw else uuid4()
    except (ValueError, TypeError):
        corr = uuid4()
    if str(corr) != raw:
        row = row.model_copy(update={"correlation_id": str(corr)})
    return row, corr
