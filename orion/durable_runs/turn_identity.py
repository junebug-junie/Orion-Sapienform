"""The correlation id a durable run's turn carries on every GPU pool lease it takes.

One rule, two readers: orion-durable-runs stamps it on the turn (services/orion-durable-runs/app/graph.py
``turn_correlation_id``), and the act ledger re-derives it to charge a reading act for the leases its
turns took (orion/grammar/act_cost_ids.py). Kept here so the two can never drift.
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


def durable_turn_correlation_id(run_id: str, correlation_id: str, lease_id: str, generation: int | str,
                                fence: int = 0) -> str:
    """The turn's correlation id under one pool hold (``lease_id``, ``generation``).

    ``fence`` > 0 only for a turn replayed after a durable-runs restart fenced the old one."""
    identity = f"orion:durable:turn:{run_id}:{correlation_id}:{lease_id}:{generation}"
    if fence:
        identity += f":fence:{int(fence)}"
    return str(uuid5(NAMESPACE_URL, identity))


__all__ = ["durable_turn_correlation_id"]
