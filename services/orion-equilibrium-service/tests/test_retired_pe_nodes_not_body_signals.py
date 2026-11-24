"""node:substrate.bus_synaptic and node:substrate.route were retired 2026-11
(PE suite rebuild step 5), node:substrate.biometrics 2026-11-18 (step 6, #3575) and
node:substrate.execution 2026-11-24 (step 6, #3576), the last of them. Their history rows linger for the table's 7-day
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
    assert "node:substrate.biometrics" not in ids   # (biometrics retired 2026-11-18, #3575)
    assert "node:substrate.execution" not in ids    # (execution retired 2026-11-24, #3576)
    assert list(ids) == []  # none of these is a body signal any more
