from __future__ import annotations

import os

from orion.fcc.github_repo_context import append_github_mcp_harness_brief
from orion.harness.operator_brief import (
    HARNESS_MOTOR_MAX_READ_LINES,
    HARNESS_UNIFIED_OPERATOR_BRIEF,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import GroundingCapsuleV1, StanceHarnessSliceV1, ThoughtEventV1


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


def _format_grounding_self_block(capsule: GroundingCapsuleV1) -> list[str]:
    """Compact motor self block: identity + relationship + continuity/memory only.

    Response policy is intentionally excluded here — it is reserved for the voice
    finalize pass to respect the motor single-context-window budget.
    """
    lines: list[str] = ["WHO YOU ARE"]
    lines.extend(f"- {item}" for item in capsule.identity_summary)
    if capsule.relationship_summary:
        lines.append("RELATIONSHIP")
        lines.extend(f"- {item}" for item in capsule.relationship_summary)
    digest = (capsule.memory_digest or capsule.continuity_digest or "").strip()
    if digest:
        lines.append("DURABLE MEMORY / CONTINUITY")
        lines.append(digest)
    return lines


def harness_motor_instruction(
    *,
    thought: ThoughtEventV1,
    answer_contract: AnswerContract | None,
) -> str:
    _ = thought
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    return (
        "Execute your imperative. Use tools when the turn requires verified facts "
        "from the repo or runtime. "
        f"Do not Read whole files over {HARNESS_MOTOR_MAX_READ_LINES} lines — "
        "use rg/Grep or Read offset/limit."
    )


def compile_harness_prefix(
    thought: ThoughtEventV1,
    *,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str = "",
    answer_contract: AnswerContract | None = None,
    workspace: str | None = None,
) -> str:
    """Deterministic fcc system prefix from stance thought + repair overlay."""
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    parts: list[str] = [HARNESS_UNIFIED_OPERATOR_BRIEF.strip()]

    if thought.grounding_capsule is not None and thought.grounding_capsule.identity_summary:
        parts.extend(_format_grounding_self_block(thought.grounding_capsule))

    parts.extend(
        [
            f"Imperative: {thought.imperative}",
            f"Tone: {thought.tone}",
        ]
    )
    parts.extend(_format_stance_slice(thought.stance_harness_slice))

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

    append_github_mcp_harness_brief(
        parts,
        workspace=workspace or os.environ.get("HARNESS_FCC_WORKSPACE"),
    )

    return "\n".join(parts)
