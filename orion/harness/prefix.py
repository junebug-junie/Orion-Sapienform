from __future__ import annotations

import os

from orion.fcc.github_repo_context import append_github_mcp_harness_brief
from orion.fcc.self_index_brief import append_self_index_harness_brief
from orion.harness.operator_brief import (
    HARNESS_UNIFIED_OPERATOR_BRIEF,
    harness_motor_instruction as _stance_motor_instruction,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import (
    AutonomySliceV1,
    GroundingCapsuleV1,
    StanceHarnessSliceV1,
    ThoughtEventV1,
)


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


def _format_autonomy_slice(sl: AutonomySliceV1) -> list[str]:
    """Compact self-signal block for the harness system prefix. Only emits
    lines for fields that are actually present -- never fabricates content."""
    lines: list[str] = ["SELF SIGNAL (autonomy)"]
    if sl.dominant_drive:
        lines.append(f"Dominant drive: {sl.dominant_drive}")
    if sl.active_tensions:
        lines.append(f"Active tensions: {', '.join(sl.active_tensions)}")
    if sl.pressure_trend:
        lines.append(f"Pressure trend: {sl.pressure_trend}")
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
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    return _stance_motor_instruction(thought=thought)


def compile_harness_prefix(
    thought: ThoughtEventV1,
    *,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str = "",
    answer_contract: AnswerContract | None = None,
    workspace: str | None = None,
) -> str:
    """Orion capability: motor-context assembly for the unified turn.

    Deterministically materializes the stance-conditioned context of the FCC
    motor prompt: the unified operator brief, grounding self block, Thought
    imperative and stance slice, autonomy slice, repair overlay, user message,
    and enabled MCP tool briefs. The full `claude -p` prompt is this prefix
    plus the harness_motor_instruction that build_harness_prompt (runner.py)
    appends on user-message turns — check both when chasing unexpected motor
    context.

    Runtime evidence: the compiled prompt is what run_fcc_turn spawns with.
    Start here when the motor acted without stance or grounding context it
    should have had, or with context it should not have had.
    """
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
    if thought.autonomy_slice is not None:
        parts.extend(_format_autonomy_slice(thought.autonomy_slice))

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
    append_self_index_harness_brief(parts)

    return "\n".join(parts)
