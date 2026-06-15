"""Agent lane configuration health for context-exec ↔ Hub."""

from __future__ import annotations

from typing import Any

from .settings import ContextExecSettings

_FAKE_ENGINES = frozenset({"fake", "mock", "smoke", "stub"})

AGENT_LANE_FAKE_WARNING = (
    "WARNING: CONTEXT_EXEC_RLM_ENGINE=fake while HUB_AGENT_CONTEXT_EXEC_ENABLED=true. "
    "Agent mode will not perform real investigations."
)


def agent_lane_health_block(cfg: ContextExecSettings | None = None) -> dict[str, Any]:
    if cfg is None:
        from .settings import settings as live

        cfg = live
    engine = (cfg.rlm_engine or "fake").strip().lower()
    degraded = engine in _FAKE_ENGINES
    return {
        "rlm_engine": engine,
        "agent_lane_degraded": degraded,
        "agent_lane_warning": AGENT_LANE_FAKE_WARNING if degraded else None,
    }


def log_agent_lane_startup_warning(cfg: ContextExecSettings | None = None) -> None:
    import logging

    block = agent_lane_health_block(cfg)
    warning = block.get("agent_lane_warning")
    if warning:
        logging.getLogger("orion-context-exec.agent_lane").warning(warning)
