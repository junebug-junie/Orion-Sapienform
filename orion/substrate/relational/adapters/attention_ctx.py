"""Attention-broadcast adapter — folds the workspace winner into beliefs.

GWT consumer rung: maps the rung-3 continuous-broadcast projection
(``AttentionBroadcastProjectionV1`` — the coalition that won the substrate
workspace competition) into a single belief node, so the unified belief set
contains a belief about *what Orion is currently attending to*. This is the
broadcast's first audience: without a consumer, the workspace winner is a
log row, not a broadcast.

ctx-sourced, pure (no network, no DB): reads ``ctx['attention_broadcast']``
as a dict or JSON string (the raw projection_json hydrated by the felt-state
reader), and degrades to ``None`` when absent, unparseable, or when nothing
is attended — never raises.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.attention_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived workspace state, refreshed every tick
_ATTENDED_NODE_CAP = 8


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="attention_broadcast",
        source_channel="substrate.attention.broadcast",
        producer="attention_broadcast_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> dict[str, Any] | None:
    try:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.debug("attention_broadcast_adapter_parse_failed error=%s", exc)
    return None


def map_attention_broadcast_ctx_to_substrate(
    ctx: dict[str, Any],
) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['attention_broadcast']`` → one ``attending:current_focus`` node."""
    ctx = ctx if isinstance(ctx, dict) else {}
    payload = _coerce(ctx.get("attention_broadcast"))
    if payload is None:
        return None

    selected_action_type = str(payload.get("selected_action_type") or "none")
    attended_raw = payload.get("attended_node_ids") or []
    attended = [str(x) for x in attended_raw if x][:_ATTENDED_NODE_CAP]
    if selected_action_type == "none" and not attended:
        # Workspace competition produced no winner — nothing is attended,
        # so there is no belief to assert.
        return None

    description = payload.get("selected_description")
    node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="attending:current_focus",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=_make_prov(),
        signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.6),
        metadata={
            "source_kind": "attention_broadcast",
            "selected_action_type": selected_action_type,
            "selected_open_loop_id": payload.get("selected_open_loop_id"),
            "selected_description": str(description) if description else None,
            "attended_node_ids": attended,
        },
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
