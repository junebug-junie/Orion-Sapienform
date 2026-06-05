"""Timeout budget helpers for memory_graph_suggest (verb yaml + Hub settings)."""

from __future__ import annotations

from typing import Any, Tuple

MEMORY_GRAPH_SUGGEST_VERB = "memory_graph_suggest"
DEFAULT_VERB_TIMEOUT_MS = 180_000
DEFAULT_CLIENT_FETCH_BUFFER_SEC = 25.0


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


def resolve_memory_graph_suggest_timeouts(
    settings: Any,
    *,
    escalation_enabled: bool = True,
) -> Tuple[float, float, float, int]:
    """
    Return (verb_timeout_sec, quick_timeout_sec, brain_timeout_sec, verb_timeout_ms).

    When escalation is enabled, Quick + Brain share one verb budget (default 180s total),
    so a browser fetch limit of 180s is not exceeded by sequential attempts.

    Per-route overrides (MEMORY_GRAPH_SUGGEST_*_TIMEOUT_SEC) are capped to the verb budget.
    """
    verb_ms = memory_graph_verb_timeout_ms(settings)
    verb_sec = float(verb_ms) / 1000.0

    quick_raw = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 0) or 0)
    brain_raw = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 0) or 0)

    if escalation_enabled:
        if quick_raw > 0:
            quick = min(quick_raw, verb_sec)
        else:
            quick = max(45.0, round(verb_sec * 0.35, 1))
        if brain_raw > 0:
            brain = min(brain_raw, max(45.0, verb_sec - quick))
        else:
            brain = max(45.0, round(verb_sec - quick, 1))
        total = quick + brain
        if total > verb_sec:
            scale = verb_sec / total
            quick = round(quick * scale, 1)
            brain = round(verb_sec - quick, 1)
    else:
        if quick_raw > 0:
            quick = min(quick_raw, verb_sec)
        else:
            quick = max(45.0, round(verb_sec * 0.4, 1))
        if brain_raw > 0:
            brain = min(brain_raw, verb_sec)
        else:
            brain = verb_sec
        quick = min(quick, verb_sec)
        brain = min(brain, verb_sec)

    return verb_sec, quick, brain, verb_ms


def memory_graph_suggest_server_budget_sec(
    settings: Any,
    *,
    escalation_enabled: bool = True,
) -> float:
    """Wall-clock budget for all suggest attempts in one Hub request."""
    _, quick, brain, _ = resolve_memory_graph_suggest_timeouts(
        settings, escalation_enabled=escalation_enabled
    )
    if escalation_enabled:
        return float(quick) + float(brain)
    primary = str(getattr(settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", "quick") or "quick").strip().lower()
    return float(brain if primary == "brain" else quick)


def hub_client_fetch_timeout_ms(
    settings: Any,
    *,
    escalation_enabled: bool = True,
) -> int:
    """
    Browser / Hub fetch AbortController budget (must exceed server_budget + overhead).
    """
    override = int(getattr(settings, "MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_TIMEOUT_MS", 0) or 0)
    if override > 0:
        return override
    server_sec = memory_graph_suggest_server_budget_sec(
        settings, escalation_enabled=escalation_enabled
    )
    buffer = float(
        getattr(settings, "MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_BUFFER_SEC", 0)
        or DEFAULT_CLIENT_FETCH_BUFFER_SEC
    )
    return int((server_sec + buffer) * 1000)


def cortex_rpc_timeout_sec(hub_attempt_timeout_sec: float, settings: Any) -> float:
    """Bus RPC wait must exceed hub asyncio.wait_for for the same attempt."""
    outer = float(getattr(settings, "TIMEOUT_SEC", 400) or 400)
    return max(outer, float(hub_attempt_timeout_sec) + 15.0)
