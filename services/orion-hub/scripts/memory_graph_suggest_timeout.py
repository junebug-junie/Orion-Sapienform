"""Timeout budget helpers for memory_graph_suggest (verb yaml + Hub settings)."""

from __future__ import annotations

from typing import Any, Tuple

MEMORY_GRAPH_SUGGEST_VERB = "memory_graph_suggest"
DEFAULT_VERB_TIMEOUT_MS = 180_000


def memory_graph_verb_timeout_ms(settings: Any | None = None) -> int:
    """Verb timeout from settings override or cognition YAML (default 180s)."""
    if settings is not None:
        raw = getattr(settings, "MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS", None)
        if raw is not None:
            try:
                ms = int(raw)
                if ms > 0:
                    return ms
            except (TypeError, ValueError):
                pass
    try:
        from orion.cognition.plan_loader import build_plan_for_verb

        plan = build_plan_for_verb(MEMORY_GRAPH_SUGGEST_VERB)
        ms = int(getattr(plan, "timeout_ms", DEFAULT_VERB_TIMEOUT_MS) or DEFAULT_VERB_TIMEOUT_MS)
        return ms if ms > 0 else DEFAULT_VERB_TIMEOUT_MS
    except Exception:
        return DEFAULT_VERB_TIMEOUT_MS


def resolve_memory_graph_suggest_timeouts(settings: Any) -> Tuple[float, float, float, int]:
    """
    Return (verb_timeout_sec, quick_timeout_sec, brain_timeout_sec, verb_timeout_ms).

    Per-route caps default from verb timeout_ms (orion/cognition/verbs/memory_graph_suggest.yaml):
    - Quick primary: ~40% of verb budget (floor 45s)
    - Brain escalation: full verb budget
  """
    verb_ms = memory_graph_verb_timeout_ms(settings)
    verb_sec = float(verb_ms) / 1000.0

    quick_raw = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 0) or 0)
    brain_raw = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 0) or 0)

    quick = quick_raw if quick_raw > 0 else max(45.0, round(verb_sec * 0.4, 1))
    brain = brain_raw if brain_raw > 0 else verb_sec

    quick = min(quick, verb_sec)
    brain = min(brain, verb_sec)
    return verb_sec, quick, brain, verb_ms


def cortex_rpc_timeout_sec(hub_attempt_timeout_sec: float, settings: Any) -> float:
    """Bus RPC wait must exceed hub asyncio.wait_for for the same attempt."""
    outer = float(getattr(settings, "TIMEOUT_SEC", 400) or 400)
    return max(outer, float(hub_attempt_timeout_sec) + 15.0)
