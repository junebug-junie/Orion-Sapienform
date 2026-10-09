from __future__ import annotations

import os

from orion.fcc.github_repo_context import append_github_mcp_harness_brief
from orion.fcc.self_index_brief import append_self_index_harness_brief
from orion.gpu_pool.placement import ServingPlacement
from orion.harness.operator_brief import (
    HARNESS_UNIFIED_OPERATOR_BRIEF,
    harness_motor_instruction as _stance_motor_instruction,
)
from orion.harness.situation_brief import append_situation_block_harness_brief
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1
from orion.schemas.reading import ReadingToolBindingV1
from orion.schemas.thought import (
    AutonomySliceV1,
    GroundingCapsuleV1,
    StanceHarnessSliceV1,
    ThoughtEventV1,
)
from orion.introspect.binding import introspect_binding_for_turn
from orion.introspect.brief import append_introspect_harness_brief
from orion.world_pulse_read.tools import append_reading_mcp_harness_brief

HARNESS_TASK_HEADER = (
    "TASK THIS TURN (respond to this message, read with the recent conversation above):"
)
HARNESS_STANCE_GUIDANCE_HEADER = (
    "STANCE GUIDANCE (how to approach the task above, not a replacement for it; "
    "anything here the task did not ask for is optional: do it only if it is cheap "
    "and directly serves the answer; a follow-up question the guidance asks you to "
    "pose is not extra work, so ask it after answering):"
)
HARNESS_TURN_RULES_HEADER = (
    "TURN RULES AND TOOLS (binding for the task above; each tool note says when it applies):"
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
    if sl.recent_actions:
        lines.append(f"Recent actions: {'; '.join(sl.recent_actions)}")
    return lines


_PROVENANCE_LABELS: dict[str, str] = {
    "live_runtime_projection": "live now",
    "derived_summary": "derived from context",
    "memory_recall": "retrieved memory",
    "static_identity_config": "static identity/config",
    "user_input": "user input",
}


def _format_context_provenance_block(context_provenance: dict[str, str]) -> list[str]:
    """One line per source kind present this turn, so the motor (which has its
    own tool access, e.g. MCP file reads) can tell live substrate signal apart
    from retrieved/static/tool-fetched content instead of guessing. See
    project_orion_substrate_bridge_confabulation for the incident this closes:
    a GitHub file fetch narrated as live computation "this turn"."""
    if not context_provenance:
        return []
    by_kind: dict[str, list[str]] = {}
    for key, kind in context_provenance.items():
        by_kind.setdefault(kind, []).append(key)
    lines = ["CONTEXT PROVENANCE (only 'live now' items are happening this turn)"]
    for kind in sorted(by_kind):
        label = _PROVENANCE_LABELS.get(kind, kind)
        lines.append(f"- {label}: {', '.join(sorted(by_kind[kind]))}")
    return lines


def _format_grounding_self_block(capsule: GroundingCapsuleV1) -> list[str]:
    """Compact motor self block: identity + relationship + continuity/memory + policy."""
    lines: list[str] = ["WHO YOU ARE"]
    lines.extend(f"- {item}" for item in capsule.identity_summary)
    if capsule.relationship_summary:
        lines.append("RELATIONSHIP")
        lines.extend(f"- {item}" for item in capsule.relationship_summary)
    digest = (capsule.memory_digest or capsule.continuity_digest or "").strip()
    if digest:
        lines.append("DURABLE MEMORY / CONTINUITY")
        lines.append(digest)
    if capsule.response_policy_summary:
        lines.append("RESPONSE POLICY")
        lines.extend(f"- {item}" for item in capsule.response_policy_summary)
    lines.extend(_format_context_provenance_block(capsule.context_provenance))
    return lines


_RECENT_TURN_ROLE_LABELS: dict[str, str] = {
    "user": "User",
    "assistant": "Orion (you, prior turn)",
    "system": "System",
}


def _format_recent_turns(recent_turns: list[TurnWindowMessageV1]) -> list[str]:
    """Bounded recent-history block for the harness prompt.

    Named as a distinct, explicitly-labeled section (not folded into 'DURABLE
    MEMORY / CONTINUITY', which is a periodic long-term digest, not verbatim
    recent turns) so the motor has a concrete, checkable reason not to answer
    each turn in isolation or default to an opening greeting past turn 1. See
    HarnessRunRequestV1.recent_turns's docstring for the gap this closes.
    Returns [] (no section at all) when there is no history -- an empty list
    means a genuinely fresh session, not a fabricated absence.
    """
    if not recent_turns:
        return []
    lines = [
        f"RECENT CONVERSATION (this session so far, {len(recent_turns)} prior "
        "message(s) -- you are mid-conversation, not opening a new one)"
    ]
    for msg in recent_turns:
        label = _RECENT_TURN_ROLE_LABELS.get(msg.role, msg.role)
        lines.append(f"- {label}: {msg.content}")
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
    prior_tool_fetch_names: list[str] | None = None,
    serving_placement: ServingPlacement | None = None,
    recent_turns: list[TurnWindowMessageV1] | None = None,
    situation_prompt_fragment: str | None = None,
    reading_binding: ReadingToolBindingV1 | None = None,
    reading_only: bool = False,
) -> str:
    """Orion capability: motor-context assembly for the unified turn.

    Deterministically materializes the stance-conditioned context of the FCC
    motor prompt, in render order: the unified operator brief, grounding self
    block, backend self-context, situation context, prior tool-fetch line,
    recent-turn history, the task header and user message, the stance guidance
    header with Thought imperative, stance slice, autonomy slice and strain
    refs, then (under the turn-rules header on user-message turns) the repair
    overlay, enabled MCP tool briefs (including orion-introspect),
    and (when a situation fragment was rendered) the canonical Situation-block explainer
    (orion/harness/situation_brief.py). The full `claude -p` prompt is this
    prefix plus the harness_motor_instruction that build_harness_prompt
    (runner.py) appends on user-message turns — check both when chasing
    unexpected motor context.

    Runtime evidence: the compiled prompt is what run_fcc_turn spawns with.
    Start here when the motor acted without stance or grounding context it
    should have had, or with context it should not have had.
    """
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    parts: list[str] = [HARNESS_UNIFIED_OPERATOR_BRIEF.strip()]

    if thought.grounding_capsule is not None and thought.grounding_capsule.identity_summary:
        parts.extend(_format_grounding_self_block(thought.grounding_capsule))

    serving_line = serving_placement.self_line() if serving_placement is not None else None
    if serving_line:
        # Answers "which real backend am I running on right now". Resolved by
        # the caller BEFORE this function runs (runner.py: the turn's GPU pool
        # lease role -> the pool's discovered profile for that role, else the
        # route's default model) so this stays a pure formatter. Never the
        # route's default stated as fact: under the GPU pool a call
        # can be served by another role (agent -> agent-gpu2 or chat), and the
        # old "currently serving this turn" line was then false about Orion
        # itself (docs/superpowers/specs/2026-09-24-gpu-pool-design.md,
        # "Transport-metric and reader impacts" item 5). Omitted entirely when
        # nothing true is known, rather than shown as a placeholder.
        parts.append(serving_line)

    if situation_prompt_fragment:
        # Resolved by orion-hub BEFORE this function runs (turn_orchestrator.py::
        # execute_unified_turn calls orion.situational.context.build_situation_for_ctx),
        # same treatment as serving_placement above -- this stays a pure
        # formatter, no network/DB calls of its own. Omitted entirely when falsy
        # (situation context disabled or failed to build) rather than shown as a
        # placeholder, so a turn with no situation data renders byte-identical to
        # before this parameter existed.
        parts.append(situation_prompt_fragment)

    stance_lines: list[str] = [
        f"Imperative: {thought.imperative}",
        f"Tone: {thought.tone}",
    ]
    stance_lines.extend(_format_stance_slice(thought.stance_harness_slice))
    if thought.autonomy_slice is not None:
        stance_lines.extend(_format_autonomy_slice(thought.autonomy_slice))
    if thought.strain_refs:
        stance_lines.append(f"Strain refs: {', '.join(thought.strain_refs)}")

    if prior_tool_fetch_names:
        # Cross-turn continuity within this same session (see
        # orion/harness/last_tool_fetch_cache.py): what you fetched via a
        # tool last turn, not what's computing live right now -- named
        # explicitly so it isn't confused with the latter (see
        # project_orion_substrate_bridge_confabulation for why that
        # distinction matters).
        parts.append(
            "Last turn you fetched content via tool: " + ", ".join(prior_tool_fetch_names)
        )

    parts.extend(_format_recent_turns(recent_turns or []))

    if user_message.strip():
        parts.append(HARNESS_TASK_HEADER)
        parts.append(f"User message: {user_message.strip()}")
        parts.append(HARNESS_STANCE_GUIDANCE_HEADER)
    parts.extend(stance_lines)

    trailing: list[str] = []
    if repair_overlay.mode != "default":
        trailing.append(f"Repair mode: {repair_overlay.mode}")

    if repair_overlay.prefix_overlay:
        trailing.append(repair_overlay.prefix_overlay)

    if repair_overlay.rule_lines:
        trailing.append("Rules: " + "; ".join(repair_overlay.rule_lines))

    append_github_mcp_harness_brief(
        trailing,
        workspace=workspace or os.environ.get("HARNESS_FCC_WORKSPACE"),
    )
    append_self_index_harness_brief(trailing)
    append_reading_mcp_harness_brief(
        trailing, reading_binding=reading_binding, reading_only=reading_only
    )
    append_introspect_harness_brief(
        trailing, binding=introspect_binding_for_turn(reading_binding, reading_only=reading_only)
    )
    append_situation_block_harness_brief(
        trailing,
        situation_prompt_fragment=situation_prompt_fragment,
    )

    if trailing and user_message.strip():
        parts.append(HARNESS_TURN_RULES_HEADER)
    parts.extend(trailing)

    return "\n".join(parts)
