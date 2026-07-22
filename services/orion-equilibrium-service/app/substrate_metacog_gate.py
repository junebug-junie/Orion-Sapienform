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
    if not ctx.get("execution_trajectory_projection"):
        return None

    ev = compute_substrate_eventfulness(
        execution_trajectory=ctx.get("execution_trajectory_projection"),
        dense_threshold=dense_threshold,
        pulse_threshold=pulse_threshold,
    )
    if ev.trigger_kind is None:
        return None

    reason = f"substrate_eventfulness:{ev.score:.2f}"
    if ev.reasons:
        reason = f"{reason}:{','.join(ev.reasons[:4])}"

    # 2026-07-22 (SelfStateV1 burn): dropped the upstream.self_state_id lookup.
    # hydrate_felt_state_ctx() no longer populates ctx["self_state"] (its
    # producer, orion-self-state-runtime, is deleted), so this always
    # resolved to None -- a dead field rather than real provenance.
    return MetacogTriggerV1(
        trigger_kind=ev.trigger_kind,
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        upstream={
            "substrate_score": ev.score,
            "reasons": list(ev.reasons),
        },
    )
