from __future__ import annotations

from typing import Any, Dict, Mapping

_VALID = frozenset({"chat", "spark", "background", "agent"})
_BACKGROUND_VERBS = frozenset(
    {
        "dream_cycle",
        "dream_synthesis",
        "log_orion_metacognition",
    }
)


def _coerce_allow_chat_fallback(opts: Dict[str, Any], ctx: Mapping[str, Any]) -> bool | None:
    """Return True/False if caller set allow_chat_fallback on options or ctx; else None (use lane default)."""
    if "allow_chat_fallback" in opts:
        v = opts.get("allow_chat_fallback")
    elif "allow_chat_fallback" in ctx:
        v = ctx.get("allow_chat_fallback")
    else:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off", ""}:
        return False
    return None


def resolve_llm_lane_for_step(*, step: Any, ctx: Mapping[str, Any], settings: Any) -> Dict[str, Any]:
    """
    Decide LLM lane metadata for gateway Phase 3 routing (orthogonal to route keys like quick/chat).
    """
    verb = str(getattr(step, "verb_name", None) or ctx.get("verb") or "").strip()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    explicit = str(opts.get("llm_lane") or ctx.get("llm_lane") or "").strip().lower()
    exec_from_ctx = str(opts.get("execution_lane") or ctx.get("execution_lane") or "").strip().lower()
    exec_from_settings = str(getattr(settings, "exec_lane", None) or "legacy").strip().lower() or "legacy"
    execution_lane = exec_from_ctx or exec_from_settings

    if explicit in _VALID:
        llm_lane = explicit
    elif verb == "introspect_spark" or execution_lane == "spark":
        llm_lane = "spark"
    elif execution_lane == "background" or verb in _BACKGROUND_VERBS:
        llm_lane = "background"
    elif execution_lane == "agent":
        llm_lane = "agent"
    else:
        llm_lane = "chat"

    acf = _coerce_allow_chat_fallback(opts, ctx)

    if llm_lane == "chat":
        return {
            "execution_lane": execution_lane,
            "llm_lane": "chat",
            "priority": "high",
            "allow_chat_fallback": True if acf is None else acf,
            "allow_degrade": False,
            "route_reason": "verb_chat",
        }
    if llm_lane == "spark":
        return {
            "execution_lane": execution_lane,
            "llm_lane": "spark",
            "priority": "low",
            "allow_chat_fallback": False if acf is None else acf,
            "allow_degrade": True,
            "route_reason": "verb_spark",
        }
    if llm_lane == "background":
        return {
            "execution_lane": execution_lane,
            "llm_lane": "background",
            "priority": "low",
            "allow_chat_fallback": False if acf is None else acf,
            "allow_degrade": True,
            "route_reason": "verb_background",
        }
    return {
        "execution_lane": execution_lane,
        "llm_lane": "agent",
        "priority": "normal",
        "allow_chat_fallback": False if acf is None else acf,
        "allow_degrade": True,
        "route_reason": "verb_agent",
    }
