from __future__ import annotations

import os
from typing import Any

from orion.fcc.github_repo_context import append_github_mcp_harness_brief
from orion.harness.operator_brief import (
    HARNESS_UNIFIED_OPERATOR_BRIEF,
    harness_motor_instruction as _stance_motor_instruction,
)
from orion.schemas.attention_frame import AttentionFrameV1
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


def _format_attention_frame(frame: AttentionFrameV1 | dict[str, Any] | None) -> list[str]:
    """Advisory curiosity/attention policy block for harness/FCC system prompts.

    Mirrors the Lane-A `chat_general.j2` instructions (advisory only, ask only when
    `selected_action.action_type == "ask"`, obey suppressions). Deterministic policy
    artifact — never fabricates curiosity, never overrides the actual task.

    Degrades to `[]` on `None`, empty, or malformed input. Never raises: this is a
    best-effort prompt seasoning, not a required field.
    """
    if not frame:
        return []
    try:
        data = frame.model_dump(mode="json") if isinstance(frame, AttentionFrameV1) else dict(frame)
    except Exception:
        return []

    selected_action = data.get("selected_action")
    selected_action = selected_action if isinstance(selected_action, dict) else {}
    action_type = selected_action.get("action_type")

    suppressions_raw = data.get("suppressions") or []
    suppressions = [
        str(s.get("reason") or "").strip()
        for s in suppressions_raw
        if isinstance(s, dict) and s.get("reason")
    ]

    open_loops_raw = data.get("open_loops") or []
    open_loop_descriptions = [
        str(loop.get("description") or "").strip()
        for loop in open_loops_raw
        if isinstance(loop, dict) and loop.get("description")
    ]

    if not action_type and not suppressions and not open_loop_descriptions:
        return []

    lines = ["ATTENTION FRAME (advisory — inspectable curiosity/attention policy)"]
    if open_loop_descriptions:
        lines.append(f"Open loops: {', '.join(open_loop_descriptions[:5])}")
    if action_type:
        lines.append(f"Selected action: {action_type}")
        question_text = selected_action.get("question_text")
        if action_type == "ask" and question_text:
            lines.append(f"Candidate question: {question_text}")
    if suppressions:
        lines.append(f"Suppressions (obey): {', '.join(suppressions)}")
    lines.append(
        "Advisory only: do not invent curiosity from scratch. Ask only when the selected "
        "action is 'ask' (at most one situated question); otherwise proceed on the actual "
        "task without adding a question. Obey suppressions."
    )
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
    attention_frame: AttentionFrameV1 | dict[str, Any] | None = None,
) -> str:
    """Deterministic fcc system prefix from stance thought + repair overlay.

    `attention_frame` is optional and defaults to `None` (no-op): `ThoughtEventV1`
    does not carry attention-frame data today (see `orion/schemas/thought.py`), so
    this is not read off `thought`. A future caller wires it in explicitly once the
    curiosity/attention frame is threaded through `HarnessRunRequestV1` or an
    equivalent request-time input; until then this parameter is dead but harmless.
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
    parts.extend(_format_attention_frame(attention_frame))

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
