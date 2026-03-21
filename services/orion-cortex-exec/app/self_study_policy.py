from __future__ import annotations

from typing import Any, Sequence

from orion.schemas.self_study import (
    SelfStudyConsumerContextV1,
    SelfStudyConsumerPolicyDecisionV1,
    SelfStudyRetrieveFiltersV1,
    SelfStudyRetrieveRequestV1,
    SelfStudyRetrieveResultV1,
)

PLANNING_OUTPUT_MODES = frozenset({"project_planning", "comparative_analysis", "decision_support"})
REFLECTIVE_OUTPUT_MODES = frozenset({"reflective_depth"})
_MODE_ORDER = {"factual": 0, "conceptual": 1, "reflective": 2}
_MODE_TRUST = {
    "factual": ["authoritative"],
    "conceptual": ["authoritative", "induced"],
    "reflective": ["authoritative", "induced", "reflective"],
}


def _normalize_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _downgrade_mode(requested_mode: str | None, max_mode: str) -> tuple[str, bool]:
    if not requested_mode:
        return max_mode, False
    if requested_mode not in _MODE_ORDER:
        return max_mode, True
    if _MODE_ORDER[requested_mode] <= _MODE_ORDER[max_mode]:
        return requested_mode, False
    return max_mode, True


def resolve_self_study_consumer_policy(
    *,
    consumer_name: str,
    output_mode: str | None,
    config: dict[str, Any] | None,
) -> SelfStudyConsumerPolicyDecisionV1:
    cfg = config or {}
    explicit = bool(cfg)
    enabled = _normalize_bool(cfg.get("enabled"), default=False)
    requested_mode = cfg.get("retrieval_mode")

    if consumer_name == "legacy.plan":
        if output_mode in REFLECTIVE_OUTPUT_MODES:
            consumer_kind = "metacog_self"
            max_mode = "reflective"
        elif output_mode in PLANNING_OUTPUT_MODES:
            consumer_kind = "planning_architecture"
            max_mode = "conceptual"
        else:
            consumer_kind = "delivery_debug"
            max_mode = "factual"
    elif consumer_name == "actions.respond_to_juniper_collapse_mirror.v1":
        consumer_kind = "metacog_self"
        max_mode = "reflective"
    else:
        return SelfStudyConsumerPolicyDecisionV1(
            consumer_name=consumer_name,
            consumer_kind="delivery_debug",
            explicit=explicit,
            enabled=False,
            retrieval_mode=None,
            max_mode=None,
            allowed_trust_tiers=[],
            policy_reason="unknown_consumer",
        )

    if not enabled:
        return SelfStudyConsumerPolicyDecisionV1(
            consumer_name=consumer_name,
            consumer_kind=consumer_kind,
            explicit=explicit,
            enabled=False,
            retrieval_mode=None,
            max_mode=max_mode,
            allowed_trust_tiers=[],
            policy_reason="self_study_not_enabled",
        )

    retrieval_mode, downgraded = _downgrade_mode(str(requested_mode) if requested_mode else None, max_mode)
    reason = "policy_allowed"
    if downgraded and requested_mode:
        reason = f"policy_downgraded_to_{retrieval_mode}"

    return SelfStudyConsumerPolicyDecisionV1(
        consumer_name=consumer_name,
        consumer_kind=consumer_kind,
        explicit=explicit,
        enabled=True,
        retrieval_mode=retrieval_mode,
        max_mode=max_mode,
        allowed_trust_tiers=list(_MODE_TRUST[retrieval_mode]),
        policy_reason=reason,
        downgraded=downgraded,
    )


def build_self_study_consumer_request(
    decision: SelfStudyConsumerPolicyDecisionV1,
    config: dict[str, Any] | None,
) -> SelfStudyRetrieveRequestV1:
    cfg = config or {}
    filters = cfg.get("filters") if isinstance(cfg.get("filters"), dict) else {}
    return SelfStudyRetrieveRequestV1(
        retrieval_mode=decision.retrieval_mode or "factual",
        filters=SelfStudyRetrieveFiltersV1.model_validate(filters),
    )


def build_self_study_consumer_context(
    decision: SelfStudyConsumerPolicyDecisionV1,
    *,
    result: SelfStudyRetrieveResultV1 | None,
    notes: Sequence[str] | None = None,
) -> SelfStudyConsumerContextV1:
    return SelfStudyConsumerContextV1(
        consumer_name=decision.consumer_name,
        consumer_kind=decision.consumer_kind,
        retrieval_mode=decision.retrieval_mode or "factual",
        policy_reason=decision.policy_reason,
        used=result is not None,
        result=result,
        notes=list(notes or []),
    )


def render_self_study_consumer_context(context: SelfStudyConsumerContextV1, *, max_items: int = 6) -> str:
    if context.result is None:
        return f"SELF-STUDY CONTEXT unavailable ({context.policy_reason})."
    lines = [
        f"SELF-STUDY CONTEXT mode={context.retrieval_mode} consumer={context.consumer_name}",
        f"Counts: total={context.result.counts.total}, authoritative={context.result.counts.authoritative}, induced={context.result.counts.induced}, reflective={context.result.counts.reflective}",
    ]
    shown = 0
    for group in context.result.groups:
        for item in group.items:
            lines.append(
                f"- [{item.trust_tier}|{item.record_type}] {item.title} :: {item.content_preview} (id={item.stable_id}, source={item.source_kind})"
            )
            shown += 1
            if shown >= max_items:
                return "\n".join(lines)
    return "\n".join(lines)
