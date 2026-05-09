"""Producer registry contracts for the Cognitive Unification Layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1


@dataclass(frozen=True)
class TrustTierV1:
    name: str           # "operator_static" | "graphdb_durable" | "concept_induced" | "snapshot_ephemeral"
    rank: int           # 1–4; lower = higher authority
    write_through: bool # whether to persist to GraphDB / durable store


# Canonical tier singletons
OPERATOR_STATIC = TrustTierV1(name="operator_static", rank=1, write_through=True)
GRAPHDB_DURABLE = TrustTierV1(name="graphdb_durable", rank=2, write_through=True)
CONCEPT_INDUCED = TrustTierV1(name="concept_induced", rank=3, write_through=True)
SNAPSHOT_EPHEMERAL = TrustTierV1(name="snapshot_ephemeral", rank=4, write_through=False)

TIER_BY_NAME: dict[str, TrustTierV1] = {
    t.name: t for t in (OPERATOR_STATIC, GRAPHDB_DURABLE, CONCEPT_INDUCED, SNAPSHOT_EPHEMERAL)
}


@dataclass(frozen=True)
class ProducerEntryV1:
    """Descriptor for a single data producer wired into the unification layer."""

    producer_id: str
    trust_tier: TrustTierV1
    anchor_scopes: tuple[str, ...]     # anchors this producer covers
    freshness_ttl_sec: int             # staleness threshold in seconds
    pull_on_cold: bool                 # fan out when anchor is cold/stale
    adapter_fn: Callable[[dict[str, Any]], SubstrateGraphRecordV1 | None]
    # adapter_fn receives the ctx dict; network-based adapters (autonomy, self_study,
    # orionmem) ignore ctx and make their own calls; ctx-based adapters (recall,
    # social, identity_yaml) read from ctx directly.


@dataclass
class ProducerRegistryV1:
    """Ordered list of producer entries; constructed once at process startup."""

    producers: list[ProducerEntryV1] = field(default_factory=list)

    def producers_for_anchor(self, anchor: str) -> list[ProducerEntryV1]:
        return [p for p in self.producers if anchor in p.anchor_scopes]

    def cold_producers_for_anchor(self, anchor: str) -> list[ProducerEntryV1]:
        return [p for p in self.producers if anchor in p.anchor_scopes and p.pull_on_cold]

    def ephemeral_ctx_producers_for_anchor(self, anchor: str) -> list[ProducerEntryV1]:
        """Producers that are always-fresh ctx-based (pull_on_cold=False, snapshot_ephemeral)."""
        return [
            p for p in self.producers
            if anchor in p.anchor_scopes
            and not p.pull_on_cold
            and p.trust_tier.name == SNAPSHOT_EPHEMERAL.name
        ]
