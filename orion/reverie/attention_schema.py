"""Reverie -> AttentionSchemaV1 projection (write-only adapter).

Adapt, do not migrate: `ReverieChainV1` / `SpontaneousThoughtV1` and their
table and consumers are untouched. This reads them and emits the shared
shape alongside (docs/superpowers/specs/2026-09-04-attention-schema-surface-
design.md, "Adapters, not inheritance").

What is real in the live data, checked 2026-09-06 against
`substrate_reverie_chain` (24,140 rows) and `substrate_reverie_thought`:

- `ReverieChainV1.trigger` -- the field the design doc pointed at as reverie's
  "why" -- is NULL on every one of the 24,140 chains. `chain.py` never sets it.
  So the honest why-it-won for a reverie chain is what actually starts one:
  the substrate broadcast coalition it inherits (`theme_key_for(coalition)`).
  Vocabulary here is `coalition_broadcast` / `no_coalition`, with
  `trigger:<pressure_kind>` reserved for the day a chain carries a trigger.
- `SpontaneousThoughtV1.interpretation` is a real, varying, LLM-written
  paragraph. It is Orion's *self-report* of what the coalition is about, so
  it is emitted as `narrative_kind="self_report"` (Missing Question 2). The
  chain's `ema_summary` is the computed fallback when no thought has one.
- `next_focus` / `drift` are empty on the sampled live rows; `predicted_next`
  is populated when they are not, and honestly None otherwise.
"""

from __future__ import annotations

from typing import Sequence

from orion.schemas.attention_schema import (
    MAX_LABEL_CHARS,
    MAX_NARRATIVE_CHARS,
    MAX_PREDICTED_NEXT_CHARS,
    AttentionSchemaV1,
    clip,
)
from orion.schemas.reverie import ReverieChainV1, SpontaneousThoughtV1


def to_attention_schema(
    chain: ReverieChainV1, thoughts: Sequence[SpontaneousThoughtV1]
) -> AttentionSchemaV1:
    """One row per chain. Pure; never touches I/O."""
    first = next((t for t in thoughts if not t.hollow), thoughts[0] if thoughts else None)
    coalition = first.coalition if first is not None else None
    theme = chain.theme_key if chain.theme_key and chain.theme_key != "unknown" else None
    attended_id = (coalition.selected_open_loop_id if coalition is not None else None) or theme

    trigger = chain.trigger
    if trigger is not None and trigger.pressure_kind and trigger.pressure_kind != "unspecified":
        reason = f"trigger:{trigger.pressure_kind}"
    elif attended_id is not None:
        reason = "coalition_broadcast"
    else:
        reason = "no_coalition"

    interpretation = clip(first.interpretation if first is not None else "", MAX_NARRATIVE_CHARS)
    if interpretation:
        narrative, kind = interpretation, "self_report"
    else:
        narrative, kind = clip(chain.ema_summary, MAX_NARRATIVE_CHARS), "computed"

    last = thoughts[-1] if thoughts else None
    predicted = None
    if last is not None:
        predicted = clip(last.next_focus or last.expectation or "", MAX_PREDICTED_NEXT_CHARS) or None

    return AttentionSchemaV1(
        entry_id=f"reverie-{chain.chain_id}",
        generated_at=chain.created_at,
        process="reverie",
        correlation_id=first.correlation_id if first is not None else None,
        attended_id=attended_id,
        attended_label=clip(theme or "", MAX_LABEL_CHARS),
        attention_reason=reason,
        reason_narrative=narrative,
        narrative_kind=kind,
        confidence=float(chain.ema_salience),
        confidence_basis=(
            f"ema_salience over {len(thoughts)} thoughts; chain ended {chain.terminal_reason}"
        ),
        predicted_next=predicted,
    )
