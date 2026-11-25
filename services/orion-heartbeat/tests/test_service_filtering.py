from __future__ import annotations

import asyncio

import pytest

from app.service import HeartbeatService


# A role each routed service really emits, with an input rule that absorbs it.
_DEFAULT_ROLE = {
    "orion-hub": "repair_pressure_reading",  # measured
    "orion-biometrics": "gpu_pressure_signal",  # measured
    "orion-cortex-exec": "exec_step_started",  # presence
    "orion-cortex-orch": "route_arbitration_decided",  # presence
    "orion-bus": "rpc_transport_timeout",  # presence
    "orion-harness-governor": "harness_fcc_step",  # presence
}


def _atom_event(
    *,
    event_kind: str = "atom_emitted",
    source_service: str = "orion-hub",
    atom_type: str = "observation",
    semantic_role: str | None = None,
    confidence: float | None = 0.8,
    salience: float | None = 0.5,
    uncertainty: float | None = 0.2,
    include_atom: bool = True,
) -> dict:
    payload: dict = {
        "event_kind": event_kind,
        "provenance": {"source_service": source_service},
    }
    if include_atom:
        payload["atom"] = {
            "atom_type": atom_type,
            "semantic_role": semantic_role if semantic_role is not None else _DEFAULT_ROLE.get(source_service, "x"),
            "confidence": confidence,
            "salience": salience,
            "uncertainty": uncertainty,
        }
    return payload


@pytest.mark.asyncio
async def test_allowlisted_organ_gets_absorbed() -> None:
    svc = HeartbeatService()
    before = svc.ensemble.tick_count()

    await svc._handle_grammar_message(_atom_event(source_service="orion-biometrics"))
    await svc._drain_absorb_queue()  # no background worker running in this test

    assert svc.ensemble.tick_count() == before + 1
    assert svc.events_absorbed == 1
    assert svc.events_skipped_organ == 0


@pytest.mark.asyncio
async def test_non_allowlisted_organ_is_skipped_not_raised() -> None:
    svc = HeartbeatService()
    before = svc.ensemble.tick_count()

    await svc._handle_grammar_message(_atom_event(source_service="orion-vision-retina"))
    await svc._drain_absorb_queue()

    assert svc.ensemble.tick_count() == before
    assert svc.events_absorbed == 0
    assert svc.events_skipped_organ == 1


@pytest.mark.asyncio
async def test_non_atom_emitted_event_kind_is_ignored() -> None:
    svc = HeartbeatService()
    before = svc.ensemble.tick_count()

    await svc._handle_grammar_message(_atom_event(event_kind="trace_started"))
    await svc._drain_absorb_queue()

    assert svc.ensemble.tick_count() == before
    assert svc.events_absorbed == 0
    assert svc.events_skipped_organ == 0
    assert svc.events_seen == 1


@pytest.mark.asyncio
async def test_atom_emitted_event_missing_atom_is_skipped() -> None:
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(include_atom=False))

    assert svc.events_skipped_no_atom == 1
    assert svc.events_absorbed == 0


@pytest.mark.asyncio
async def test_events_seen_counts_everything_absorbed_or_not() -> None:
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(source_service="orion-hub"))
    await svc._handle_grammar_message(_atom_event(source_service="orion-vision-retina"))
    await svc._handle_grammar_message(_atom_event(event_kind="edge_emitted"))
    await svc._drain_absorb_queue()

    assert svc.events_seen == 3
    assert svc.events_absorbed == 1


@pytest.mark.asyncio
async def test_stats_reports_allowlisted_organs_and_substrate_health() -> None:
    svc = HeartbeatService()
    stats = await svc.stats()

    assert stats["allowlisted_organs"] == sorted(
        [
            "orion-hub",
            "orion-biometrics",
            "orion-cortex-exec",
            "orion-cortex-orch",
            "orion-bus",
            "orion-harness-governor",
        ]
    )
    assert stats["max_bond"] >= 1
    assert stats["norm"] == pytest.approx(1.0, abs=1e-6)
    assert stats["n_trajectories"] == svc.ensemble.config.n_trajectories
    assert len(stats["seeds"]) == svc.ensemble.config.n_trajectories


def test_latest_h1_dict_is_none_before_first_computation() -> None:
    svc = HeartbeatService()
    assert svc.latest_h1_dict() is None


@pytest.mark.asyncio
async def test_absorb_records_organ_fire_window_for_h1() -> None:
    from app.substrate.reconstruction import compute_h1_ensemble

    svc = HeartbeatService()
    await svc._handle_grammar_message(_atom_event(source_service="orion-cortex-exec"))
    await svc._handle_grammar_message(_atom_event(source_service="orion-cortex-exec"))
    await svc._handle_grammar_message(_atom_event(source_service="orion-bus"))
    await svc._drain_absorb_queue()

    counts = svc.organ_fires.counts()
    assert counts["orion-cortex-exec"] == 2
    assert counts["orion-bus"] == 1
    assert counts["orion-hub"] == 0

    result = compute_h1_ensemble(svc.ensemble, fire_counts=counts)
    assert "orion-hub" in result.dark_seats
    assert "orion-cortex-exec" not in result.dark_seats


@pytest.mark.asyncio
async def test_unknown_atom_type_is_skipped_through_full_handler_path() -> None:
    # Review gap: events_skipped_atom_type was only tested at the routing
    # module level in isolation (test_routing.py), never through
    # _handle_atom_payload/_handle_grammar_message the way
    # events_skipped_organ already was above.
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(atom_type="not_a_real_atom_type"))
    await svc._drain_absorb_queue()

    assert svc.ensemble.tick_count() == 0
    assert svc.events_absorbed == 0
    assert svc.events_skipped_atom_type == 1


@pytest.mark.asyncio
async def test_malformed_numeric_fields_are_skipped_not_raised() -> None:
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(salience="not-a-number"))
    await svc._drain_absorb_queue()

    assert svc.events_absorbed == 0
    assert svc.events_skipped_malformed == 1


@pytest.mark.asyncio
async def test_clockwork_input_is_skipped_and_counted_by_reason() -> None:
    svc = HeartbeatService()
    await svc._handle_grammar_message(
        _atom_event(source_service="orion-bus", atom_type="signal", semantic_role="bus_observer_tick_started")
    )
    await svc._handle_grammar_message(
        _atom_event(source_service="orion-biometrics", atom_type="signal", semantic_role="memory_pressure_signal")
    )
    await svc._drain_absorb_queue()

    assert svc.events_absorbed == 0
    assert svc.events_skipped_input == 2
    assert sum(svc.events_skipped_input_by_reason.values()) == 2
    assert any(k.startswith("clockwork") for k in svc.events_skipped_input_by_reason)
    assert any(k.startswith("structural floor") for k in svc.events_skipped_input_by_reason)


@pytest.mark.asyncio
async def test_cortex_orch_fires_the_execution_seat() -> None:
    svc = HeartbeatService()
    await svc._handle_grammar_message(
        _atom_event(source_service="orion-cortex-orch", atom_type="reasoning_step")
    )
    await svc._drain_absorb_queue()
    assert svc.events_absorbed == 1
    assert svc.organ_fires.counts()["orion-cortex-exec"] == 1
    assert svc.inputs_absorbed_by_kind == {"presence": 1}


@pytest.mark.asyncio
async def test_unlisted_role_on_a_routed_service_is_counted() -> None:
    svc = HeartbeatService()
    await svc._handle_grammar_message(
        _atom_event(source_service="orion-cortex-exec", atom_type="signal", semantic_role="brand_new_role")
    )
    await svc._drain_absorb_queue()
    assert svc.events_absorbed == 1  # cortex-exec default is presence
    assert svc.inputs_unlisted_role == {"orion-cortex-exec:brand_new_role": 1}


@pytest.mark.asyncio
async def test_absorb_queue_full_drops_and_counts_instead_of_blocking() -> None:
    # HEARTBEAT_ABSORB_QUEUE_MAXSIZE overflow -- previously untested. A full
    # queue must degrade to a dropped-and-counted event (with a log line),
    # never raise out of _handle_grammar_message/_handle_atom_payload.
    svc = HeartbeatService()
    svc._absorb_queue = asyncio.Queue(maxsize=1)  # tiny queue, easy to fill

    await svc._handle_grammar_message(_atom_event(source_service="orion-hub"))
    assert svc.events_queued == 1
    assert svc.events_dropped_queue_full == 0

    await svc._handle_grammar_message(_atom_event(source_service="orion-biometrics"))

    assert svc.events_queued == 1  # second atom did NOT get queued
    assert svc.events_dropped_queue_full == 1
    assert svc._absorb_queue.qsize() == 1
