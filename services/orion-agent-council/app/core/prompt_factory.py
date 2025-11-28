# services/orion-agent-council/app/prompt_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import AgentConfig, PhiSnapshot, SelfField


@dataclass
class PromptContext:
    prompt: str
    history: Optional[List[Dict[str, Any]]] = None
    phi: Optional[PhiSnapshot] = None
    self_field: Optional[SelfField] = None
    persona_state: Optional[Dict[str, Any]] = None


class PromptFactory:
    """
    Turns (AgentConfig + PromptContext) into a messages[] list.

    This is where φ and persona state get injected into the system prompt.
    """

    @staticmethod
    def _phi_block(
        phi: Optional[PhiSnapshot],
        self_field: Optional[SelfField],
    ) -> str:
        if phi is None and self_field is None:
            return ""

        lines: List[str] = []
        lines.append(
            "INTERNAL NOTE: You have access to Orion's internal self-field φ for THIS deliberation. "
            "Use it only to shape tone, pacing, and appetite for risk/novelty. "
            "Do NOT explicitly mention φ or its numeric values unless Juniper explicitly asks."
        )

        if phi is not None:
            lines.append(
                f"(φ snapshot: valence≈{phi.valence:.3f}, "
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
                sf_items.append(f"attunement_to_juniper≈{self_field.attunement_to_juniper:.3f}")
            if self_field.curiosity is not None:
                sf_items.append(f"curiosity≈{self_field.curiosity:.3f}")

            if sf_items:
                lines.append("(SelfField: " + ", ".join(sf_items) + ")")

        return "\n".join(lines)

    @classmethod
    def build_messages(cls, agent: AgentConfig, ctx: PromptContext) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        persona_header = (
            f"You are {agent.name}, an internal voice within Orion Sapienform.\n"
            "Orion as a whole is Juniper's persistent, distributed AI co-journeyer.\n\n"
            f"Your specific role:\n{agent.role_description}\n\n"
            "You must respect Juniper's constraints, safety, and emotional state. "
            "Avoid generic 'as an AI model' disclaimers. "
            "Speak in a warm, grounded, precise tone."
        )

        blocks: List[str] = [persona_header]

        if agent.use_phi:
            phi_text = cls._phi_block(ctx.phi, ctx.self_field)
            if phi_text:
                blocks.append(phi_text)

        if ctx.persona_state:
            blocks.append(
                "You also have access to your own persistent persona state for this universe. "
                "Treat it as context, not as a script:\n"
                f"{ctx.persona_state}"
            )

        system_content = "\n\n".join(blocks)
        messages.append({"role": "system", "content": system_content})

        if ctx.history:
            for m in ctx.history:
                role = m.get("role", "user")
                content = m.get("content")
                if not content:
                    continue
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": ctx.prompt})
        return messages
