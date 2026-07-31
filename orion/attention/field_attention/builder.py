from __future__ import annotations

from datetime import datetime, timezone

from orion.attention.field_attention.candidate_precision_weighted import PrecisionEwmaBaseline
from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01
from orion.attention.field_attention.selectors import (
    select_capability_targets,
    select_host_targets,
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
    prediction_error_baselines: dict[str, PrecisionEwmaBaseline] | None = None,
    previous_frame: FieldAttentionFrameV1 | None = None,
    now: datetime | None = None,
) -> FieldAttentionFrameV1:
    """2026-07-30: `previous_frame` is used by `select_host_targets`/
    `select_capability_targets` (Candidate B's `novelty_scorer()`, real
    theory-grounded coverage for targets Candidate A's precision-weighting
    can't reach -- no real prediction-error history exists for physical
    hosts or capabilities). NOT used by `select_node_targets` (Candidate A's
    precision-weighting already accounts for "how surprising is this
    relative to its own history" as its core theory; a second, hand-tuned
    novelty layer on top of it would reintroduce exactly the disease this
    patch removes -- deliberate asymmetry, not an oversight).

    `prediction_error_baselines` is Candidate A's real input: {node_id:
    PrecisionEwmaBaseline}, a persisted, incrementally-updated running
    baseline per target, caller-fetched/advanced
    (`AttentionRuntimeStore.advance_node_prediction_error_baseline`) so this
    stays a pure function -- see `select_node_targets`'s own docstring.
    2026-07-30 fix (Sentience Striving Program officer review, `orion/
    sentience_striving_program/README.md` §12): was `prediction_error_
    histories: dict[str, list[float]]`, a raw ASC-by-time error history
    re-fetched fresh from a ~30-minute rolling retention window every tick
    -- replaced with a persisted baseline whose observation count survives
    that window's own pruning, per that section's full incident record.
    """
    generated_at = now or datetime.now(timezone.utc)

    node_targets = select_node_targets(
        field, policy, prediction_error_baselines or {}
    ) + select_host_targets(field, policy, previous_frame)
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
