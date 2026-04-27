from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.world_pulse import SourceTrustAssessmentV1, WorldPulseSourceV1


def assess_source(source: WorldPulseSourceV1) -> SourceTrustAssessmentV1:
    confidence = max(0.1, min(1.0, 1.05 - (source.trust_tier * 0.15)))
    reasons = [f"tier:{source.trust_tier}", f"approved:{source.approved}", f"enabled:{source.enabled}"]
    return SourceTrustAssessmentV1(
        source_id=source.source_id,
        trust_tier=source.trust_tier,
        allowed_uses=source.allowed_uses,
        requires_corroboration=source.requires_corroboration,
        politics_allowed=source.politics_allowed,
        factuality=source.factuality,
        bias_profile=source.bias_profile,
        confidence_weight=confidence,
        reasons=reasons,
        assessed_at=source.updated_at or source.created_at or datetime.now(timezone.utc),
    )


def source_allowed_for_fetch(*, source: WorldPulseSourceV1, requested_by: str) -> bool:
    if not source.enabled:
        return False
    if requested_by == "scheduler" and not source.approved:
        return False
    if requested_by != "scheduler" and not source.approved:
        return False
    return True


def source_allowed_for_graph(*, source: WorldPulseSourceV1) -> bool:
    return bool(source.allowed_uses.graph_write) and int(source.trust_tier) <= 3
