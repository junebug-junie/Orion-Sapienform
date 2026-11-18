"""node:substrate.bus_synaptic and node:substrate.route were retired 2026-11
(PE suite rebuild step 5). Their history rows linger for the table's 7-day
retention; the outcome rollup must not treat them as body signals."""

from __future__ import annotations

from app.self_signal_outcome_rollup import body_signal_ids


def test_retired_nodes_are_not_body_signals() -> None:
    ids = body_signal_ids(
        [
            "node:substrate.bus_synaptic",
            "node:substrate.route",
            "node:substrate.execution",
            "node:substrate.biometrics",
        ]
    )
    assert "node:substrate.bus_synaptic" not in ids
    assert "node:substrate.route" not in ids
    assert {"node:substrate.execution", "node:substrate.biometrics"} <= set(ids)
