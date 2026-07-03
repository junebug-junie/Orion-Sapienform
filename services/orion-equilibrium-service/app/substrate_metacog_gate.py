from __future__ import annotations

import logging
from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.substrate.felt_state_reader import hydrate_felt_state_ctx
from orion.substrate.metacog_trigger_signals import compute_substrate_eventfulness

logger = logging.getLogger("orion.equilibrium.substrate_metacog_gate")


def build_substrate_metacog_trigger(
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    dense_threshold: float,
    pulse_threshold: float,
) -> MetacogTriggerV1 | None:
    """Read substrate projections and return a dense/pulse trigger, or None."""
    ctx: dict[str, Any] = {}
    hydrate_felt_state_ctx(ctx)
    if not ctx.get("self_state") and not ctx.get("execution_trajectory_projection"):
        return None

    ev = compute_substrate_eventfulness(
        self_state=ctx.get("self_state"),
        execution_trajectory=ctx.get("execution_trajectory_projection"),
        dense_threshold=dense_threshold,
        pulse_threshold=pulse_threshold,
    )
    if ev.trigger_kind is None:
        return None

    reason = f"substrate_eventfulness:{ev.score:.2f}"
    if ev.reasons:
        reason = f"{reason}:{','.join(ev.reasons[:4])}"

    return MetacogTriggerV1(
        trigger_kind=ev.trigger_kind,
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        upstream={
            "substrate_score": ev.score,
            "reasons": list(ev.reasons),
            "self_state_id": (ctx.get("self_state") or {}).get("self_state_id")
            if isinstance(ctx.get("self_state"), dict)
            else None,
        },
    )
