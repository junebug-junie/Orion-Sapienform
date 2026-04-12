"""Answer/runtime grounding strings split from delivery-oriented helpers."""

from __future__ import annotations

from typing import Any, Iterable

# Generic Orion runtime description — no Discord-specific guidance here (see DISCORD_ADAPTER_BLOCK).
ORION_ARCHITECTURE_GROUNDING = (
    "Ground answers in Orion's actual runtime. "
    "The live path is Hub/Client -> Cortex-Orch -> Cortex-Exec -> PlannerReact/AgentChain -> "
    "LLM Gateway and supporting services over the bus. "
    "Keep references anchored to current repo components such as Hub, Orch, Exec, PlannerReact, AgentChain, "
    "LLM Gateway, recall, and bus channels when relevant."
)

DISCORD_ADAPTER_BLOCK = (
    "When the user explicitly asks about Discord, describe a Discord bot or bridge as an adapter into the "
    "Orion flow above (session intake, routing, tool execution, reply path), not as a replacement web stack."
)

GENERIC_DRIFT_WARNING = (
    "Do not silently substitute a random stack. "
    "Do not rewrite an Orion architecture request into generic Flask, Ubuntu, Gunicorn, or Nginx deployment "
    "unless the user explicitly asks for that stack."
)

_DISCORD_TERMS = ("discord", "bot", "guild", "server", "channel", "gateway intent", "slash command")


def delivery_grounding_mode(*, user_text: str, output_mode: str | None) -> str:
    text = (user_text or "").strip().lower()
    if output_mode in {"implementation_guide", "tutorial", "code_delivery", "comparative_analysis", "decision_support"}:
        if "orion" in text:
            return "orion_repo_architecture"
        if any(term in text for term in _DISCORD_TERMS):
            return "integration_delivery"
    return "default_delivery"


def _compose_grounding_context(*, mode: str, user_text: str) -> str:
    text_l = (user_text or "").strip().lower()
    if mode == "orion_repo_architecture":
        base = ORION_ARCHITECTURE_GROUNDING
        if any(term in text_l for term in _DISCORD_TERMS):
            return f"{base}\n\n{DISCORD_ADAPTER_BLOCK}"
        return base
    if mode == "integration_delivery":
        return (
            "Describe the requested integration as an adapter around the existing Orion runtime flow. "
            "Keep the answer implementation-oriented and avoid replacing the runtime with an unrelated standalone app."
        )
    return ""


def build_answer_grounding_context(*, user_text: str, output_mode: str | None) -> dict[str, Any]:
    """Structured grounding for answer/render verbs (repo + runtime; Discord only when user mentions it)."""
    mode = delivery_grounding_mode(user_text=user_text, output_mode=output_mode)
    context = _compose_grounding_context(mode=mode, user_text=user_text)
    return {
        "delivery_grounding_mode": mode,
        "grounding_context": context,
        "anti_generic_drift": GENERIC_DRIFT_WARNING,
    }


def extract_trace_preferred_output(trace_snapshot: Iterable[Any]) -> tuple[str, bool]:
    for raw_step in reversed(list(trace_snapshot or [])):
        step = raw_step.model_dump(mode="json") if hasattr(raw_step, "model_dump") else raw_step
        if not isinstance(step, dict):
            continue
        obs = step.get("observation")
        if isinstance(obs, dict):
            for key in ("llm_output", "text", "content"):
                value = obs.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:8000], True
        if isinstance(obs, str) and obs.strip():
            return obs.strip()[:8000], True
    return "", False
