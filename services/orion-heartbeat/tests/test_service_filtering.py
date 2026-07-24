from __future__ import annotations

import pytest

from app.service import HeartbeatService


def _atom_event(
    *,
    event_kind: str = "atom_emitted",
    source_service: str = "orion-hub",
    atom_type: str = "observation",
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
            "confidence": confidence,
            "salience": salience,
            "uncertainty": uncertainty,
        }
    return payload


@pytest.mark.asyncio
async def test_allowlisted_organ_gets_absorbed() -> None:
    svc = HeartbeatService()
    before = svc.substrate.tick_count

    await svc._handle_grammar_message(_atom_event(source_service="orion-biometrics"))

    assert svc.substrate.tick_count == before + 1
    assert svc.events_absorbed == 1
    assert svc.events_skipped_organ == 0


@pytest.mark.asyncio
async def test_non_allowlisted_organ_is_skipped_not_raised() -> None:
    svc = HeartbeatService()
    before = svc.substrate.tick_count

    await svc._handle_grammar_message(_atom_event(source_service="orion-vision-retina"))

    assert svc.substrate.tick_count == before
    assert svc.events_absorbed == 0
    assert svc.events_skipped_organ == 1


@pytest.mark.asyncio
async def test_non_atom_emitted_event_kind_is_ignored() -> None:
    svc = HeartbeatService()
    before = svc.substrate.tick_count

    await svc._handle_grammar_message(_atom_event(event_kind="trace_started"))

    assert svc.substrate.tick_count == before
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

    assert svc.events_seen == 3
    assert svc.events_absorbed == 1


@pytest.mark.asyncio
async def test_stats_reports_allowlisted_organs_and_substrate_health() -> None:
    svc = HeartbeatService()
    stats = svc.stats()

    assert stats["allowlisted_organs"] == sorted(
        ["orion-hub", "orion-biometrics", "orion-cortex-exec", "orion-bus", "orion-cortex-orch"]
    )
    assert stats["max_bond"] >= 1
    assert stats["norm"] == pytest.approx(1.0, abs=1e-6)


def test_latest_h1_dict_is_none_before_first_computation() -> None:
    svc = HeartbeatService()
    assert svc.latest_h1_dict() is None


@pytest.mark.asyncio
async def test_unknown_atom_type_is_skipped_through_full_handler_path() -> None:
    # Review gap: events_skipped_atom_type was only tested at the routing
    # module level in isolation (test_routing.py), never through
    # _handle_atom_payload/_handle_grammar_message the way
    # events_skipped_organ already was above.
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(atom_type="not_a_real_atom_type"))

    assert svc.substrate.tick_count == 0
    assert svc.events_absorbed == 0
    assert svc.events_skipped_atom_type == 1


@pytest.mark.asyncio
async def test_malformed_numeric_fields_are_skipped_not_raised() -> None:
    svc = HeartbeatService()

    await svc._handle_grammar_message(_atom_event(confidence="not-a-number"))

    assert svc.events_absorbed == 0
    assert svc.events_skipped_malformed == 1
