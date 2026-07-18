"""Identity YAML adapter — operator_static tier.

Converts operator-authored identity facts from ctx into a single
``StateSnapshotNodeV1`` anchored to ``orion``.  Hub must populate
``orion_identity_summary``, ``juniper_relationship_summary``, and
``response_policy_summary`` in ctx before the first ``beliefs_for_stance`` call;
there is no disk-reading path in this adapter.  Results are written to the
durable substrate store and protected from overwrite by lower tiers.

Does NOT emit ``ConceptNodeV1`` nodes. ``orion_identity_summary`` is a flat
merge of ``orion_identity.yaml``'s ``nature``, ``core_drives``,
``self_permissions``, and ``anti_patterns`` blocks (see
``orion/cognition/personality/identity_context.py``) — operator-authored
identity facts and response-style directives ("speak as a real ongoing
presence rather than customer-support software") with no distinction between
them. Concept graph nodes are supposed to represent things Orion actually
induced, not operator config reified verbatim; a prior version of this
adapter turned every line into a ``concept``-kind graph node and polluted the
substrate concept graph with policy prose. The snapshot's metadata still
carries the full lists for stance-brief prompt assembly
(``chat_stance.py:_project_identity_from_beliefs`` reads it by
``snapshot_source``), which is unaffected by this change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)

from orion.substrate.adapters._common import make_temporal

_TIER_RANK = 1  # operator_static


def _make_op_static_provenance(*, source_kind: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="user_asserted",
        source_kind=source_kind,
        source_channel="identity_yaml.adapter",
        producer="identity_yaml_adapter",
        tier_rank=_TIER_RANK,
    )


def map_identity_yaml_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map identity ctx keys into substrate nodes (operator_static, anchor=orion).

    Reads ``orion_identity_summary``, ``juniper_relationship_summary``, and
    ``response_policy_summary`` from ctx.  Returns None if ctx is empty or None.
    """
    ctx = ctx if isinstance(ctx, dict) else {}

    orion_summary: list[str] = [str(v).strip() for v in (ctx.get("orion_identity_summary") or []) if str(v).strip()]
    juniper_summary: list[str] = [str(v).strip() for v in (ctx.get("juniper_relationship_summary") or []) if str(v).strip()]
    response_policy: list[str] = [str(v).strip() for v in (ctx.get("response_policy_summary") or []) if str(v).strip()]

    if not any([orion_summary, juniper_summary, response_policy]):
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    high_signals = SubstrateSignalBundleV1(confidence=0.95, salience=0.8)

    # Single StateSnapshotNodeV1 carrying all three summary lists in metadata —
    # the projection helper (chat_stance.py:_project_identity_from_beliefs) reads
    # these back by snapshot_source for stance-brief prompt assembly. No
    # ConceptNodeV1 nodes are emitted — see module docstring.
    snapshot = StateSnapshotNodeV1(
        node_id="sub-identity-snapshot-orion",
        anchor_scope="orion",
        temporal=temporal,
        provenance=_make_op_static_provenance(source_kind="identity_yaml.snapshot"),
        signals=high_signals,
        snapshot_source="identity_yaml",
        dimensions={"identity_weight": 1.0},
        metadata={
            "orion_identity_summary": orion_summary,
            "juniper_relationship_summary": juniper_summary,
            "response_policy_summary": response_policy,
        },
        promotion_state="canonical",
    )

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[snapshot])
