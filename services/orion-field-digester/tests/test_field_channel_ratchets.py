"""Regression tests for the ratchet-channel bugs found via the 69h/122k-row
live corpus scan (2026-07-16): expected_offline_suppression, bus_health, and
delivery_confidence were structurally unable to move back down once
perturbed to 1.0 (mode="add", the Perturbation dataclass default, plus
absence from decay.py's NODE_DECAY_CHANNELS -- nothing ever multiplied them
back down and add-mode can't subtract). See
app/ingest/state_deltas.py and app/digestion/suppression.py for the fix
mechanism and reasoning.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.digestion.perturbation import apply_perturbations
from app.digestion.suppression import apply_suppression
from app.ingest.state_deltas import Perturbation, delta_to_perturbations
from app.tensor.update_rules import run_digestion_tick

from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _state(node_vectors: dict[str, dict[str, float]] | None = None) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_ratchet_test",
        node_vectors=node_vectors or {},
        edges=[],
    )


def _node_biometrics_delta(
    *,
    node_id: str = "circe",
    expected_online: bool | None,
    operation: str = "update",
) -> StateDeltaV1:
    after: dict = {"node_id": node_id, "expected_online": expected_online}
    return StateDeltaV1(
        delta_id="delta_biometrics_1",
        target_projection="node_biometrics",
        target_kind="node_biometrics",
        target_id=f"node:{node_id}",
        operation=operation,  # type: ignore[arg-type]
        after=after,
        caused_by_event_ids=["gev_bio_1"],
        reducer_id="biometrics_node_reducer",
    )


def _transport_bus_delta(*, node_id: str = "athena", hints: dict) -> StateDeltaV1:
    after = {"node_id": node_id, "pressure_hints": hints}
    return StateDeltaV1(
        delta_id="delta_transport_1",
        target_projection="transport_bus",
        target_kind="transport_bus",
        target_id=f"bus:{node_id}",
        operation="update",
        after=after,
        caused_by_event_ids=["gev_transport_1"],
        reducer_id="transport_bus_reducer",
    )


# ---------------------------------------------------------------------------
# expected_offline_suppression clear path (item 1)
# ---------------------------------------------------------------------------


def test_expected_online_false_still_sets_suppression_to_one() -> None:
    """Pre-existing behavior, unchanged: the original `is False` branch still sets the flag."""
    delta = _node_biometrics_delta(expected_online=False)
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p for p in perturbations}
    assert "expected_offline_suppression" in channels
    assert channels["expected_offline_suppression"].intensity == 1.0


def test_expected_online_true_clears_suppression_via_replace() -> None:
    """New mirror branch: expected_online=True must emit a mode='replace' 0.0 perturbation,
    not silently do nothing (which is exactly the original bug)."""
    delta = _node_biometrics_delta(expected_online=True)
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p for p in perturbations}
    assert "expected_offline_suppression" in channels, (
        f"expected a clear perturbation for expected_online=True, got {list(channels)}"
    )
    clear = channels["expected_offline_suppression"]
    assert clear.intensity == 0.0
    assert clear.mode == "replace"


def test_expected_online_none_emits_no_suppression_perturbation() -> None:
    """A delta that doesn't carry the expected_online field at all should not spuriously
    set or clear the channel."""
    delta = _node_biometrics_delta(expected_online=None)
    perturbations = delta_to_perturbations(delta)
    channels = [p.channel for p in perturbations]
    assert "expected_offline_suppression" not in channels


def test_suppression_latch_then_clear_end_to_end() -> None:
    """The regression this bug produced: perturb to 1.0 via the suppress path (mode="add",
    the ratchet), then apply the expected_online=True clearing signal. Before the fix, the
    channel had no writer that could ever move it below 1.0 -- mode="add" can only climb.
    After the fix, mode="replace" fully clears it rather than adding 0.0 on top of 1.0
    (which would have been a no-op with the old ceiling logic anyway, but proves the new
    path uses replace, not add-mode carryover)."""
    state = _state()

    # Latch via the existing "suppress" path (active_node_pressure operation="suppress").
    suppress_perturbations = [
        Perturbation(
            node_id="node:circe",
            channel="expected_offline_suppression",
            intensity=1.0,
            label="delta_suppress_1",
        )
    ]
    apply_perturbations(state, suppress_perturbations)
    assert state.node_vectors["node:circe"]["expected_offline_suppression"] == 1.0

    # A second additive perturbation on top -- this is exactly the ratchet: with
    # mode="add" there is no way back down, only further clamped-at-1.0 accumulation.
    apply_perturbations(state, suppress_perturbations)
    assert state.node_vectors["node:circe"]["expected_offline_suppression"] == 1.0

    # Now the reverse signal fires (expected_online flips back to True upstream).
    clear_delta = _node_biometrics_delta(node_id="circe", expected_online=True)
    clear_perturbations = delta_to_perturbations(clear_delta)
    apply_perturbations(state, clear_perturbations)

    assert state.node_vectors["node:circe"]["expected_offline_suppression"] == 0.0, (
        "expected_offline_suppression must be clearable back to 0.0, not stuck via "
        "add-mode carryover"
    )


# ---------------------------------------------------------------------------
# Collateral damage: apply_suppression's floor/zero effect on availability/staleness
# ---------------------------------------------------------------------------


def test_suppression_floors_availability_and_zeroes_staleness_while_latched() -> None:
    """Sanity check on the existing, legitimate apply_suppression behavior: while
    expected_offline_suppression is latched at 1.0, availability is floored and
    staleness is zeroed. This is supposed to happen -- the bug was that it could
    never stop happening."""
    state = _state(
        node_vectors={
            "node:circe": {
                "expected_offline_suppression": 1.0,
                "availability": 0.1,
                "staleness": 0.9,
            }
        }
    )
    apply_suppression(state)
    assert state.node_vectors["node:circe"]["availability"] == 0.85
    assert state.node_vectors["node:circe"]["staleness"] == 0.0


def test_clearing_suppression_unblocks_staleness_and_availability() -> None:
    """The collateral-damage regression: once expected_offline_suppression is cleared
    to 0.0, a subsequent staleness perturbation must no longer be zeroed out by
    apply_suppression, and availability must no longer be floor-clamped to 0.85."""
    state = _state(
        node_vectors={
            "node:circe": {
                "expected_offline_suppression": 1.0,
                # Below the 0.85 floor apply_suppression would otherwise impose --
                # proves the value is left alone, not re-floored, once cleared.
                "availability": 0.3,
                "staleness": 0.0,
            }
        }
    )

    # Node comes back: clear the suppression flag.
    clear_delta = _node_biometrics_delta(node_id="circe", expected_online=True)
    clear_perturbations = delta_to_perturbations(clear_delta)
    apply_perturbations(state, clear_perturbations)
    assert state.node_vectors["node:circe"]["expected_offline_suppression"] == 0.0

    # A genuinely stale reading arrives for the node afterward.
    stale_delta = StateDeltaV1(
        delta_id="delta_stale_1",
        target_projection="node_biometrics",
        target_kind="node_biometrics",
        target_id="node:circe",
        operation="update",
        after={"node_id": "circe", "availability_status": "stale"},
        caused_by_event_ids=["gev_stale_1"],
        reducer_id="biometrics_node_reducer",
    )
    stale_perturbations = delta_to_perturbations(stale_delta)
    apply_perturbations(state, stale_perturbations)
    # Real computed staleness before suppression runs.
    assert state.node_vectors["node:circe"]["staleness"] == 0.5

    apply_suppression(state)

    assert state.node_vectors["node:circe"]["staleness"] == 0.5, (
        "staleness should no longer be zeroed out once expected_offline_suppression "
        "is cleared"
    )
    assert state.node_vectors["node:circe"]["availability"] == 0.3, (
        "availability should no longer be floor-clamped to 0.85 once "
        "expected_offline_suppression is cleared"
    )


def test_full_tick_no_longer_permanently_floors_once_cleared() -> None:
    """End-to-end via run_digestion_tick: with expected_offline_suppression cleared,
    a tick that perturbs availability downward must actually reflect that drop
    instead of being floored back up to 0.85 by apply_suppression."""
    state = _state(
        node_vectors={
            "node:circe": {
                "expected_offline_suppression": 0.0,
                "availability": 0.85,
            }
        }
    )
    perturbations = [
        Perturbation(node_id="node:circe", channel="availability", intensity=0.2, label="p1"),
    ]
    run_digestion_tick(state, perturbations=perturbations, decay_rate=0.92, diffusion_rate=0.0)
    assert state.node_vectors["node:circe"]["availability"] == 0.2


# ---------------------------------------------------------------------------
# bus_health / delivery_confidence mode="replace" fix (item 2)
# ---------------------------------------------------------------------------


def test_bus_health_delta_uses_replace_mode() -> None:
    delta = _transport_bus_delta(hints={"bus_health": 1.0})
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p for p in perturbations}
    assert channels["bus_health"].mode == "replace"


def test_delivery_confidence_delta_uses_replace_mode() -> None:
    delta = _transport_bus_delta(hints={"delivery_confidence": 1.0})
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p for p in perturbations}
    assert channels["delivery_confidence"].mode == "replace"


def test_bus_health_drop_reflected_not_ceiling_clamped() -> None:
    """The regression: perturb up to 1.0, then perturb down to 0.3. Under the old
    mode="add" default this would stay clamped at 1.0 forever (add-mode ceiling).
    With mode="replace" the drop must be reflected exactly."""
    state = _state()

    up = delta_to_perturbations(_transport_bus_delta(hints={"bus_health": 1.0}))
    apply_perturbations(state, up)
    assert state.node_vectors["node:athena"]["bus_health"] == 1.0

    down = delta_to_perturbations(_transport_bus_delta(hints={"bus_health": 0.3}))
    apply_perturbations(state, down)
    assert state.node_vectors["node:athena"]["bus_health"] == 0.3


def test_delivery_confidence_drop_reflected_not_ceiling_clamped() -> None:
    state = _state()

    up = delta_to_perturbations(_transport_bus_delta(hints={"delivery_confidence": 1.0}))
    apply_perturbations(state, up)
    assert state.node_vectors["node:athena"]["delivery_confidence"] == 1.0

    down = delta_to_perturbations(_transport_bus_delta(hints={"delivery_confidence": 0.3}))
    apply_perturbations(state, down)
    assert state.node_vectors["node:athena"]["delivery_confidence"] == 0.3


def test_other_transport_bus_channels_still_default_add_mode() -> None:
    """Scope check: only bus_health/delivery_confidence changed mode. The other
    transport_bus channels (out of scope for this fix -- reliability_pressure,
    transport_pressure, etc.) must remain untouched (default mode="add")."""
    delta = _transport_bus_delta(
        hints={
            "transport_pressure": 0.4,
            "catalog_drift_pressure": 0.4,
            "observer_failure_pressure": 0.4,
            "reliability_pressure": 0.4,
            "contract_pressure": 0.4,
        }
    )
    perturbations = delta_to_perturbations(delta)
    for p in perturbations:
        assert p.mode == "add", f"{p.channel} unexpectedly changed mode to {p.mode}"
