from __future__ import annotations

from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import ThoughtEventV1


def compile_harness_prefix(
    thought: ThoughtEventV1,
    *,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str = "",
) -> str:
    """Deterministic fcc system prefix from stance thought + repair overlay.

    Embeds imperative, tone, strain_refs, and repair overlay lines. Never calls
    compile_speech_contract — unified harness motor uses felt content directly.
    """
    parts = [
        f"Imperative: {thought.imperative}",
        f"Tone: {thought.tone}",
    ]

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
