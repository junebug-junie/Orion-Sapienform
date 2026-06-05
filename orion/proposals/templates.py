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
        "Self-state is loaded with elevated field/resource/execution pressure; summarize for downstream review.",
        ["loaded_operating_condition", "downstream_review"],
    ),
    "watch_reliability": (
        "Observe orchestration reliability signals",
        "Reliability pressure warrants continued observation of capability:orchestration without action.",
        ["reliability_watch", "preserve_stability"],
    ),
    "request_policy_review_for_action": (
        "Prepare policy review for possible action",
        "Agency readiness and execution pressure are sufficient to consider a policy-gated action proposal.",
        ["policy_gate_required", "not_approval", "not_execution"],
    ),
    "defer_due_to_low_readiness": (
        "Defer action until readiness improves",
        "Uncertainty or reliability pressure suggests deferring possible action until policy can evaluate later.",
        ["defer_stability", "low_readiness"],
    ),
    "inspect_transport_status": (
        "Inspect transport capability status",
        "Transport contract or reliability pressure is elevated; inspect capability:transport evidence.",
        ["transport_inspect", "read_only"],
    ),
    "inspect_bus_channel_catalog": (
        "Inspect bus channel catalog alignment",
        "Configured observer streams may be uncataloged; inspect orion/bus/channels.yaml read-only.",
        ["transport_catalog_inspect", "read_only"],
    ),
    "summarize_transport_contract_drift": (
        "Summarize transport contract drift",
        "Catalog drift pressure on capability:transport warrants a bounded summary for review.",
        ["transport_contract_drift", "read_only"],
    ),
    "watch_transport_backpressure": (
        "Watch transport backpressure",
        "Transport pressure signals warrant observation without bus mutation.",
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
            f"Template {template_key} matched current self-state dimensions.",
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
