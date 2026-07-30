from __future__ import annotations

from typing import Literal, cast

ProposalKind = Literal[
    "observe",
    "inspect",
    "summarize",
    "stabilize",
    "defer",
    "request_policy_review",
    "prepare_action",
]

TargetKind = Literal[
    "node",
    "capability",
    "field",
    "self_state",
    "service",
    "system",
]

ProposedEffect = Literal[
    "increase_observability",
    "reduce_pressure",
    "preserve_stability",
    "increase_coherence",
    "defer_until_policy",
    "prepare_for_policy_gate",
    "no_effect",
]

PolicyGate = Literal[
    "none",
    "read_only",
    "operator_review",
    "autonomy_policy",
    "execution_policy",
]


_TEMPLATE_COPY: dict[str, tuple[str, str, list[str]]] = {
    "inspect_execution_pressure": (
        "Inspect orchestration execution pressure",
        "Execution pressure is elevated on capability:orchestration; inspect supporting field and attention evidence.",
        ["execution_pressure_elevated", "orchestration_inspect"],
    ),
    "summarize_loaded_state": (
        "Summarize loaded operating condition",
        "Resource pressure is loaded; summarize for downstream review.",
        ["loaded_operating_condition", "downstream_review"],
    ),
    "watch_reliability": (
        "Observe orchestration reliability signals",
        "Reliability pressure warrants continued observation of capability:orchestration without action.",
        ["reliability_watch", "preserve_stability"],
    ),
    "request_policy_review_for_action": (
        "Prepare policy review for possible action",
        "Execution pressure is sufficient to consider a policy-gated action proposal.",
        ["policy_gate_required", "not_approval", "not_execution"],
    ),
    "defer_due_to_low_readiness": (
        "Defer action until readiness improves",
        "Reliability pressure suggests deferring possible action until policy can evaluate later.",
        ["defer_stability", "low_readiness"],
    ),
    "inspect_transport_status": (
        "Inspect transport capability status",
        "Reliability pressure is elevated; inspect capability:transport evidence.",
        ["transport_inspect", "read_only"],
    ),
    # 2026-07-30: these 3 transport templates no longer score against any real
    # dimension (contract_pressure was never produced by field_pressures() --
    # see config/proposals/proposal_policy.v1.yaml's own comment on each).
    # Descriptions no longer claim a "pressure signal" reasoning that doesn't
    # exist; they surface on base_priority + the real all-4-core-dimensions
    # urgency fallback alone, same read-only bounded action either way.
    "inspect_bus_channel_catalog": (
        "Inspect bus channel catalog alignment",
        "Configured observer streams may be uncataloged; inspect orion/bus/channels.yaml read-only.",
        ["transport_catalog_inspect", "read_only"],
    ),
    "summarize_transport_contract_drift": (
        "Summarize transport contract drift",
        "Periodic bounded summary of capability:transport for review.",
        ["transport_contract_drift", "read_only"],
    ),
    "watch_transport_backpressure": (
        "Watch transport backpressure",
        "Periodic observation of transport signals without bus mutation.",
        ["transport_backpressure_watch", "read_only"],
    ),
}

TRANSPORT_PROPOSAL_TEMPLATE_KEYS = frozenset(
    {
        "inspect_transport_status",
        "inspect_bus_channel_catalog",
        "summarize_transport_contract_drift",
        "watch_transport_backpressure",
    }
)

FORBIDDEN_TRANSPORT_PROPOSAL_KEYS = frozenset(
    {
        "restart_bus",
        "purge_stream",
        "replay_stream",
        "change_catalog",
        "change_bus_config",
    }
)


def template_title_description(
    template_key: str,
    *,
    target_id: str,
) -> tuple[str, str, list[str]]:
    title, description, reasons = _TEMPLATE_COPY.get(
        template_key,
        (
            f"Proposal for {target_id}",
            # 2026-07-30: was "...matched current self-state dimensions" --
            # stale since the 2026-07-22 SelfStateV1 burn (builder.py now
            # scores directly off FieldStateV1's field_pressures(), not
            # self-state).
            f"Template {template_key} matched current field pressure dimensions.",
            [f"template:{template_key}"],
        ),
    )
    return title, description, list(reasons)


def cast_proposal_kind(kind: str) -> ProposalKind:
    return cast(ProposalKind, kind)


def cast_target_kind(kind: str) -> TargetKind:
    return cast(TargetKind, kind)


def cast_proposed_effect(effect: str) -> ProposedEffect:
    return cast(ProposedEffect, effect)


def cast_policy_gate(gate: str) -> PolicyGate:
    return cast(PolicyGate, gate)
