from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models import AgentConfig, PhiSnapshot, SelfField


@dataclass
class PromptContext:
    """
    Shared context for building prompts for any agent.
    Keeps council logic DRY, even across universes.
    """
    prompt: str
    history: Optional[List[Dict[str, Any]]] = None
    phi: Optional[PhiSnapshot] = None
    self_field: Optional[SelfField] = None
    persona_state: Optional[Dict[str, Any]] = None


class PromptFactory:
    """
    Responsible for turning (AgentConfig + PromptContext) into LLM message lists.
    Council + LLM client never hand-build messages directly.
    """

    @staticmethod
    def _phi_block(
        phi: Optional[PhiSnapshot],
        self_field: Optional[SelfField],
    ) -> str:
        """
        Lightweight φ-aware note appended to the system prompt.

        Mirrors Brain's pattern:
        - φ is internal guidance only.
        - numeric details should NOT be spoken unless explicitly asked.
        """
        if phi is None and self_field is None:
            return ""

        lines: List[str] = []
        lines.append(
            "You have access to an internal self-field φ summarizing Orion's current stance "
            "for THIS deliberation. Use this only to shape tone, pacing, and strategy. "
            "Do NOT explicitly mention φ, valence, energy, coherence, novelty, or numeric values "
            "unless Juniper explicitly asks."
        )

        if phi is not None:
            lines.append(
                f"(INTERNAL φ snapshot: valence≈{phi.valence:.3f}, "
                f"energy≈{phi.energy:.3f}, coherence≈{phi.coherence:.3f}, "
                f"novelty≈{phi.novelty:.3f})"
            )

        if self_field is not None:
            sf_items = []
            if self_field.calm is not None:
                sf_items.append(f"calm≈{self_field.calm:.3f}")
            if self_field.stress_load is not None:
                sf_items.append(f"stress_load≈{self_field.stress_load:.3f}")
            if self_field.uncertainty is not None:
                sf_items.append(f"uncertainty≈{self_field.uncertainty:.3f}")
            if self_field.focus is not None:
                sf_items.append(f"focus≈{self_field.focus:.3f}")
            if self_field.attunement_to_juniper is not None:
                sf_items.append(
                    f"attunement_to_juniper≈{self_field.attunement_to_juniper:.3f}"
                )
            if self_field.curiosity is not None:
                sf_items.append(f"curiosity≈{self_field.curiosity:.3f}")

            if sf_items:
                lines.append("(INTERNAL SelfField: " + ", ".join(sf_items) + ")")

        return "\n".join(lines)

    @classmethod
    def build_messages(cls, agent: AgentConfig, ctx: PromptContext) -> List[Dict[str, Any]]:
        """
        Single entrypoint: build a messages[] list for an LLM from agent + context.
        """
        msgs: List[Dict[str, Any]] = []

        persona_header = (
            f"You are {agent.name}, one of several internal voices of Orion Sapienform.\n"
            f"Your role:\n{agent.role_description}\n\n"
            "Speak candidly, but stay grounded and precise. "
            "Address Juniper's question directly from your perspective, "
            "and avoid generic 'as an AI model' disclaimers."
        )

        extra_blocks: List[str] = []

        # Optional φ block
        if agent.use_phi:
            phi_text = cls._phi_block(ctx.phi, ctx.self_field)
            if phi_text:
                extra_blocks.append(phi_text)

        # Optional persona-specific state
        if ctx.persona_state:
            extra_blocks.append(
                "You also have access to your own persistent persona state for this universe. "
                "Treat it as context, not a script:\n"
                f"{ctx.persona_state}"
            )

        system_content = persona_header
        if extra_blocks:
            system_content = system_content + "\n\n" + "\n\n".join(extra_blocks)

        msgs.append({"role": "system", "content": system_content})

        # Optional prior history
        if ctx.history:
            for m in ctx.history:
                role = m.get("role", "user")
                content = m.get("content")
                if not content:
                    continue
                msgs.append({"role": role, "content": content})

        # Current user prompt
        msgs.append({"role": "user", "content": ctx.prompt})
        return msgs
