from __future__ import annotations

from datetime import datetime, timezone

from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01
from orion.attention.field_attention.selectors import (
    select_capability_targets,
    select_node_targets,
    select_system_targets,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1


def stable_frame_id(*, tick_id: str, policy_id: str) -> str:
    return f"attention.frame:{tick_id}:{policy_id}"


def build_attention_frame(
    *,
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None = None,
    now: datetime | None = None,
) -> FieldAttentionFrameV1:
    generated_at = now or datetime.now(timezone.utc)

    node_targets = select_node_targets(field, policy, previous_frame)
    capability_targets = select_capability_targets(field, policy, previous_frame)
    system_targets = select_system_targets(field, policy)

    all_targets = node_targets + capability_targets + system_targets
    all_targets.sort(key=lambda t: t.salience_score, reverse=True)

    active: list[FieldAttentionTargetV1] = []
    suppressed: list[FieldAttentionTargetV1] = []
    for t in all_targets:
        if t.salience_score < policy.thresholds.suppress_below:
            suppressed.append(t)
        elif t.salience_score >= policy.thresholds.min_salience:
            active.append(t)
        else:
            suppressed.append(t)

    nodes = [t for t in active if t.target_kind == "node"][: policy.limits.max_node_targets]
    caps = [t for t in active if t.target_kind == "capability"][: policy.limits.max_capability_targets]
    systems = [t for t in active if t.target_kind == "system"][: policy.limits.max_system_targets]
    capped = (nodes + caps + systems)[: policy.limits.max_targets_total]
    capped.sort(key=lambda t: t.salience_score, reverse=True)

    overall = clamp01(max((t.salience_score for t in capped), default=0.0))

    return FieldAttentionFrameV1(
        frame_id=stable_frame_id(tick_id=field.tick_id, policy_id=policy.policy_id),
        generated_at=generated_at,
        source_field_tick_id=field.tick_id,
        source_field_generated_at=field.generated_at,
        attention_policy_id=policy.policy_id,
        overall_salience=overall,
        dominant_targets=capped,
        node_targets=nodes,
        capability_targets=caps,
        system_targets=systems,
        suppressed_targets=suppressed,
        recent_perturbations=list(field.recent_perturbations),
        warnings=[],
    )
