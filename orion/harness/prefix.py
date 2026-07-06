from __future__ import annotations

from orion.harness.operator_brief import HARNESS_REPO_OPERATOR_BRIEF, HARNESS_RUNTIME_OPERATOR_BRIEF
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1

_TECH_TASK_MODES = frozenset({"technical_collaboration", "triage"})
_TECH_FRAMES = frozenset({"technical", "planning"})


def _needs_repo_operator_brief(
    thought: ThoughtEventV1,
    answer_contract: AnswerContract | None,
) -> bool:
    if answer_contract is not None and answer_contract.requires_repo_grounding:
        return True
    sl = thought.stance_harness_slice
    return sl.task_mode in _TECH_TASK_MODES or sl.conversation_frame in _TECH_FRAMES


def _needs_runtime_operator_brief(
    thought: ThoughtEventV1,
    answer_contract: AnswerContract | None,
) -> bool:
    if answer_contract is not None and answer_contract.requires_runtime_grounding:
        return True
    sl = thought.stance_harness_slice
    return sl.task_mode == "triage" and sl.conversation_frame in _TECH_FRAMES


def _format_stance_slice(sl: StanceHarnessSliceV1) -> list[str]:
    lines = [
        f"Task mode: {sl.task_mode}",
        f"Conversation frame: {sl.conversation_frame}",
        f"Answer strategy: {sl.answer_strategy}",
    ]
    if sl.interaction_regime:
        lines.append(f"Interaction regime: {sl.interaction_regime}")
    if sl.response_priorities:
        lines.append(f"Response priorities: {', '.join(sl.response_priorities)}")
    if sl.response_hazards:
        lines.append(f"Response hazards: {', '.join(sl.response_hazards)}")
    return lines


def _format_answer_contract(contract: AnswerContract) -> list[str]:
    lines = [
        f"Request kind: {contract.request_kind}",
        f"Preferred render: {contract.preferred_render_style}",
    ]
    if contract.requires_repo_grounding:
        lines.append("Requires repo grounding: yes")
    if contract.requires_runtime_grounding:
        lines.append("Requires runtime grounding: yes")
    if contract.asks_for_steps:
        lines.append("Asks for steps: yes")
    return lines


def harness_motor_instruction(
    *,
    thought: ThoughtEventV1,
    answer_contract: AnswerContract | None,
) -> str:
    if _needs_repo_operator_brief(thought, answer_contract):
        return (
            "Use repo tools to gather evidence, then answer with concrete paths, symbols, "
            "and actionable steps. Preserve code snippets the user needs."
        )
    if _needs_runtime_operator_brief(thought, answer_contract):
        return (
            "Inspect runtime state with tools, then answer with exact services, commands, "
            "and the next triage action."
        )
    sl = thought.stance_harness_slice
    if sl.task_mode in _TECH_TASK_MODES:
        return "Answer technically and concretely. Use tools when facts matter."
    return "Respond to the user."


def compile_harness_prefix(
    thought: ThoughtEventV1,
    *,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str = "",
    answer_contract: AnswerContract | None = None,
) -> str:
    """Deterministic fcc system prefix from stance thought + repair overlay + contract."""
    parts: list[str] = []

    if _needs_repo_operator_brief(thought, answer_contract):
        parts.append(HARNESS_REPO_OPERATOR_BRIEF.strip())
    elif _needs_runtime_operator_brief(thought, answer_contract):
        parts.append(HARNESS_RUNTIME_OPERATOR_BRIEF.strip())

    parts.extend(
        [
            f"Imperative: {thought.imperative}",
            f"Tone: {thought.tone}",
        ]
    )
    parts.extend(_format_stance_slice(thought.stance_harness_slice))

    if answer_contract is not None:
        parts.append("Answer contract:")
        parts.extend(f"  - {line}" for line in _format_answer_contract(answer_contract))

    if thought.strain_refs:
        parts.append(f"Strain refs: {', '.join(thought.strain_refs)}")

    if user_message.strip():
        parts.append(f"User message: {user_message.strip()}")

    if repair_overlay.mode != "default":
        parts.append(f"Repair mode: {repair_overlay.mode}")

    if repair_overlay.prefix_overlay:
        parts.append(repair_overlay.prefix_overlay)

    if repair_overlay.rule_lines:
        parts.append("Rules: " + "; ".join(repair_overlay.rule_lines))

    return "\n".join(parts)
