"""Recall bundle adapter — snapshot_ephemeral tier.

Maps ``ctx["recall_bundle"]`` fragments into substrate nodes.  No network call;
always-fresh from ctx.  pull_on_cold=False (registered with TTL=0).

Fragment source tags:
  journal / metacog → ConceptNodeV1  (label = snippet)
  tension            → TensionNodeV1
  dream              → EventNodeV1
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EventNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    TensionNodeV1,
)

from orion.substrate.adapters._common import make_temporal

_TIER_RANK = 4  # snapshot_ephemeral
_ANCHOR_DEFAULT = "orion"


def _make_prov(*, source_tag: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind=f"recall.{source_tag}",
        source_channel="ctx.recall_bundle",
        producer="recall_adapter",
        tier_rank=_TIER_RANK,
    )


def _anchor_for_fragment(frag: dict[str, Any]) -> str:
    subject = str(frag.get("subject") or frag.get("anchor") or "").lower()
    if "relationship" in subject:
        return "relationship"
    if "juniper" in subject:
        return "juniper"
    return _ANCHOR_DEFAULT


def map_recall_bundle_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ctx recall_bundle fragments → substrate nodes (snapshot_ephemeral)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    recall_bundle = ctx.get("recall_bundle") or {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle, dict) else []

    if not isinstance(fragments, list) or not fragments:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    nodes: list[Any] = []

    for frag in fragments[:24]:
        if not isinstance(frag, dict):
            continue
        source = str(frag.get("source") or "").lower()
        snippet = str(frag.get("snippet") or "").strip()[:200]
        if not snippet:
            continue

        anchor = _anchor_for_fragment(frag)

        if "journal" in source or "metacog" in source:
            nodes.append(
                ConceptNodeV1(
                    anchor_scope=anchor,
                    label=snippet,
                    temporal=temporal,
                    provenance=_make_prov(source_tag="journal" if "journal" in source else "metacog"),
                    signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.4),
                    metadata={"recall_source": source, "snippet": snippet},
                )
            )
        elif "tension" in source:
            nodes.append(
                TensionNodeV1(
                    anchor_scope=anchor,
                    tension_kind="recall",
                    intensity=0.5,
                    temporal=temporal,
                    provenance=_make_prov(source_tag="tension"),
                    signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.5),
                    metadata={"recall_source": source, "label": snippet, "snippet": snippet},
                )
            )
        elif "dream" in source:
            nodes.append(
                EventNodeV1(
                    anchor_scope=anchor,
                    event_type="dream",
                    summary=snippet,
                    temporal=temporal,
                    provenance=_make_prov(source_tag="dream"),
                    signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.3),
                    metadata={"recall_source": source},
                )
            )

    return SubstrateGraphRecordV1(anchor_scope=_ANCHOR_DEFAULT, nodes=nodes) if nodes else None
