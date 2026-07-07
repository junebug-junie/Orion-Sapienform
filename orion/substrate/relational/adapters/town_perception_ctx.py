"""Town-perception adapter — binds Orion's embodied world contact into the substrate.

Higher-order rung: maps ``WorldPerceptionV1`` (Orion's live view of who is near
its AI Town body) into substrate proximity/social belief nodes so the unified
belief set contains beliefs about *where Orion is and who it is next to* — the
"I am embodied near X" grounding — not only abstract self/world state.

Each nearby player becomes a bounded-salience ``town:near:{name}`` concept node
anchored on Orion, where salience is a proximity function ``1/(1+distance)`` so
closer players contribute more pressure. Contributions are bounded and the
adapter never raises; the substrate perception cadence is slower than the drive
tick, keeping this from turning into a feedback loop.

ctx-sourced, no network: reads ``ctx['perception']`` as a ``WorldPerceptionV1``,
a dict, or a JSON string.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.schemas.embodiment import WorldPerceptionV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.town_perception_ctx")

_TIER_RANK = 2  # graphdb_durable-equivalent: town perception is a trusted internal lane


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="town_perception",
        source_channel="orion:embodiment:perception",
        producer="town_perception_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> WorldPerceptionV1 | None:
    try:
        if isinstance(raw, WorldPerceptionV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return WorldPerceptionV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return WorldPerceptionV1.model_validate(raw)
    except Exception as exc:
        logger.debug("town_perception_adapter_parse_failed error=%s", exc)
    return None


def _proximity_from_distance(distance: float) -> float:
    """Bounded proximity: 1 at distance 0, decaying toward 0 as distance grows."""
    try:
        d = max(0.0, float(distance))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + d)))


def map_town_perception_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['perception']`` → substrate proximity belief nodes (anchor=orion)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    perception = _coerce(ctx.get("perception") or ctx.get("perception_json"))
    if perception is None:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()
    nodes: list[Any] = []

    for entry in perception.nearby_players or []:
        if not isinstance(entry, dict):
            continue
        player_id = str(entry.get("player_id") or "").strip()
        name = str(entry.get("name") or player_id or "unknown").strip()
        distance = entry.get("distance")
        if distance is None and isinstance(entry.get("position"), dict):
            try:
                px = float(entry["position"].get("x", 0.0))
                py = float(entry["position"].get("y", 0.0))
                ox = float(perception.position.get("x", 0.0))
                oy = float(perception.position.get("y", 0.0))
                distance = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
            except (TypeError, ValueError):
                distance = None
        proximity = _proximity_from_distance(distance if distance is not None else 1.0)
        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"town:near:{name}",
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.7, salience=proximity),
                metadata={
                    "source_kind": "town_perception",
                    "player_id": player_id,
                    "name": name,
                    "distance": round(float(distance), 6) if distance is not None else None,
                    "proximity": round(proximity, 6),
                },
            )
        )

    if not nodes:
        return None

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)
