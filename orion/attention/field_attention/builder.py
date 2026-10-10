from __future__ import annotations

from datetime import datetime, timezone

from orion.attention.field_attention.candidate_precision_weighted import PrecisionEwmaBaseline
from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01
from orion.attention.field_attention.selectors import (
    mark_observed_not_attended,
    select_capability_targets,
    select_host_targets,
    select_node_targets,
    select_system_targets,
    world_first_targets,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1


_OVER_CAP_REASON = (
    "over the per-kind target cap this tick: observed, not attended "
    "(kept here so the next tick's novelty diff has a real prior)"
)


def stable_frame_id(*, tick_id: str, policy_id: str) -> str:
    return f"attention.frame:{tick_id}:{policy_id}"


def build_attention_frame(
    *,
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    prediction_error_baselines: dict[str, PrecisionEwmaBaseline] | None = None,
    previous_frame: FieldAttentionFrameV1 | None = None,
    now: datetime | None = None,
    previous_field: FieldStateV1 | None = None,
    world_first_candidates: list | None = None,
) -> FieldAttentionFrameV1:
    """``world_first_candidates`` (a list of ``AttentionCandidateV1``; the
    worker passes it when ``ATTENTION_WORLD_FIRST_ENABLED`` is on) switches
    this frame to world-first ranking (``_build_world_first_frame``): the
    world by default, the body only when unusual for itself, and no winner
    on a calm tick. None keeps the previous ranking below, byte for byte.

    2026-07-30: `previous_frame` is used by `select_host_targets`/
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

    # #2534 decision 2: novelty skips channels that went dark / came back.
    # Needs the field tick the previous frame was built from; any other tick
    # would compare against the wrong vectors, so a mismatch is ignored.
    if (
        previous_field is not None
        and (previous_frame is None or previous_field.tick_id != previous_frame.source_field_tick_id)
    ):
        previous_field = None

    if world_first_candidates is not None:
        return _build_world_first_frame(
            field=field,
            policy=policy,
            previous_frame=previous_frame,
            previous_field=previous_field,
            generated_at=generated_at,
            candidates=world_first_candidates,
        )

    node_targets = select_node_targets(
        field, policy, prediction_error_baselines or {}, now=generated_at
    ) + select_host_targets(field, policy, previous_frame, previous_field)
    capability_targets = select_capability_targets(field, policy, previous_frame, previous_field)
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
    # Active targets past a per-kind cap are not attended this tick, but they
    # were observed: record them with the suppressed ones. Dropping them left
    # no entry for next tick's novelty diff to find, so the same steady target
    # read its whole pressure as fresh novelty on the following tick
    # (2026-09-25, D1 in docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md).
    kept = {id(t) for t in (*nodes, *caps, *systems)}
    suppressed.extend(
        t.model_copy(update={"reasons": [*t.reasons, _OVER_CAP_REASON]})
        for t in active
        if id(t) not in kept
    )
    # Same order as every other bucket: strongest first (stable, so ties keep
    # the pre-cap order). Unsorted, the over-cap targets -- the strongest
    # ones here -- trailed the below-threshold ones.
    suppressed.sort(key=lambda t: t.salience_score, reverse=True)
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


def _build_world_first_frame(
    *,
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
    previous_field: FieldStateV1 | None,
    generated_at: datetime,
    candidates: list,
) -> FieldAttentionFrameV1:
    """World-first frame. Attended targets are only the eligible candidates
    (world fresh and busier than usual; body high/unusual in its bad
    direction), best first by percentile against their own history.
    Host/capability/system novelty targets are still computed so the next
    tick's novelty diff has a prior, but are recorded as observed, not
    attended. No eligible candidate -> empty ``dominant_targets`` and
    ``overall_salience`` 0.0: an explicit no-winner frame."""
    attended, observed = world_first_targets(field, policy, candidates)
    candidate_ids = {getattr(c, "source_id", None) for c in candidates}
    host_field = field.model_copy(
        update={
            "node_vectors": {
                k: v for k, v in field.node_vectors.items() if k not in candidate_ids
            }
        }
    )
    body_novelty = (
        select_host_targets(host_field, policy, previous_frame, previous_field)
        + select_capability_targets(field, policy, previous_frame, previous_field)
        + select_system_targets(field, policy)
    )
    suppressed = observed + mark_observed_not_attended(body_novelty)
    suppressed.sort(key=lambda t: t.salience_score, reverse=True)

    capped = attended[: policy.limits.max_targets_total]
    over = attended[policy.limits.max_targets_total:]
    suppressed = [
        *(t.model_copy(update={"reasons": [*t.reasons, _OVER_CAP_REASON]}) for t in over),
        *suppressed,
    ]
    nodes = [t for t in capped if t.target_kind == "node"]
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
        capability_targets=[],
        system_targets=[],
        suppressed_targets=suppressed,
        recent_perturbations=list(field.recent_perturbations),
        warnings=[] if capped else ["world_first_no_winner"],
    )
