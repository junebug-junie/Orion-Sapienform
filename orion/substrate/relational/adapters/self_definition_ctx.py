"""Self-definition adapter -- graphdb_durable tier.

Orion's own answer to "what am I", written as a `:SelfDefinition` node during
a self-inquiry curiosity run and mirrored by Hub into `self_concept_history`
(`concept_id="self:definition"`, `produced_by="curiosity_self_inquiry"`;
see orion/curiosity/self_inquiry.py). The felt-state reader's
`orion_self_definition` lane (orion/substrate/felt_state_reader.py) hydrates
the latest row into ctx; this adapter turns it into one
``StateSnapshotNodeV1`` anchored to ``orion`` so
``chat_stance.py:_project_identity_from_beliefs`` can put Orion's own words
into the identity kernel next to the operator-authored card.

Same shape and same rule as ``identity_yaml.py``: one snapshot, no concept
nodes, source is ctx and never disk. Tier 2 rather than 1 because it is
Orion-authored and revisable, not operator config -- the authored card still
wins a collision, but nothing here collides with it: the snapshot has its own
node_id and its own ``snapshot_source``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)

from orion.substrate.adapters._common import make_temporal

CTX_KEY = "orion_self_definition"
SNAPSHOT_SOURCE = "self_definition"
NODE_ID = "sub-self-definition-orion"
_TIER_RANK = 2  # graphdb_durable


def _provenance() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="self_concept_history.self_definition",
        source_channel="self_definition_ctx.adapter",
        producer="self_definition_ctx_adapter",
        tier_rank=_TIER_RANK,
    )


def _parse_created_at(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def map_self_definition_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """``ctx["orion_self_definition"]`` -> one snapshot node, or None.

    The ctx value is the felt-state lane's payload: a dict with ``content``,
    ``version``, ``evidence_refs``, ``entry_id`` and ``created_at``. Empty
    content is None (nothing this turn), not an empty snapshot -- a blank
    self-definition must not displace the authored card's lines.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    raw = ctx.get(CTX_KEY)
    if not isinstance(raw, dict):
        return None
    content = str(raw.get("content") or "").strip()
    if not content:
        return None
    evidence = [str(v).strip() for v in (raw.get("evidence_refs") or []) if str(v).strip()]
    try:
        version = int(raw.get("version") or 1)
    except (TypeError, ValueError):
        version = 1
    observed_at = _parse_created_at(raw.get("created_at")) or datetime.now(timezone.utc)

    snapshot = StateSnapshotNodeV1(
        node_id=NODE_ID,
        anchor_scope="orion",
        temporal=make_temporal(observed_at=observed_at),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.8),
        snapshot_source=SNAPSHOT_SOURCE,
        dimensions={"identity_weight": 0.8, "evidence_count": float(len(evidence))},
        metadata={
            "content": content,
            "version": version,
            "evidence_refs": evidence,
            "entry_id": str(raw.get("entry_id") or ""),
            "created_at": observed_at.isoformat(),
        },
        promotion_state="canonical",
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[snapshot])
