from __future__ import annotations

import math

import pytest

from app.substrate.routing import (
    ABS_Z_CALM_FLOOR,
    ATOM_TYPE_OPERATOR_KIND,
    BOUNDARY_BULK_CUT,
    BOUNDARY_SITES,
    BULK_SITES,
    N_SITES,
    ORGAN_SITE_MAP,
    PRESENCE_LEVEL,
    ROLE_RULES,
    SEAT_ORDER,
    SERVICE_DEFAULT_RULES,
    SITE_ORGAN_MAP,
    UNROUTED_PRODUCERS,
    SkippedInputError,
    UnroutableAtomTypeError,
    UnroutableOrganError,
    input_rule,
    route_atom,
)


def _atom(role: str, atom_type: str = "signal", **fields) -> dict:
    return {"atom_type": atom_type, "semantic_role": role, **fields}


def test_seats_are_the_five_boundary_sites_in_order() -> None:
    assert SEAT_ORDER == (
        "orion-hub",
        "orion-biometrics",
        "orion-cortex-exec",
        "orion-bus",
        "orion-harness-governor",
    )
    assert SITE_ORGAN_MAP == dict(enumerate(SEAT_ORDER))
    assert tuple(sorted(SITE_ORGAN_MAP)) == BOUNDARY_SITES


def test_cortex_orch_folds_into_the_execution_seat() -> None:
    # #3838: ~70 atoms/day, every field stamped -> merged, not its own seat.
    assert ORGAN_SITE_MAP["orion-cortex-orch"] == ORGAN_SITE_MAP["orion-cortex-exec"] == 2


def test_every_routed_service_has_a_default_rule_and_lands_on_a_boundary_seat() -> None:
    assert set(SERVICE_DEFAULT_RULES) == set(ORGAN_SITE_MAP)
    assert set(ORGAN_SITE_MAP.values()) == set(BOUNDARY_SITES)
    for service, _role in ROLE_RULES:
        assert service in ORGAN_SITE_MAP


def test_unrouted_producers_are_disjoint_from_the_seats_and_carry_a_reason() -> None:
    assert not set(UNROUTED_PRODUCERS) & set(ORGAN_SITE_MAP)
    assert all(reason.strip() for reason in UNROUTED_PRODUCERS.values())


def test_every_catalogued_producer_is_routed_or_named_as_unrouted() -> None:
    from app.substrate.routing import SELF_SOURCE_SERVICE, catalog_grammar_producers

    producers = catalog_grammar_producers()
    assert producers, "channels.yaml could not be read"
    missing = producers - set(ORGAN_SITE_MAP) - set(UNROUTED_PRODUCERS) - {SELF_SOURCE_SERVICE}
    assert not missing, f"new grammar producer(s) need a seat or an UNROUTED_PRODUCERS reason: {missing}"


def test_lattice_geometry_unchanged() -> None:
    assert N_SITES == 10
    assert set(BOUNDARY_SITES) | set(BULK_SITES) == set(range(N_SITES))
    assert set(BOUNDARY_SITES).isdisjoint(BULK_SITES)
    assert BOUNDARY_BULK_CUT == max(BOUNDARY_SITES) + 1 == min(BULK_SITES)


def test_every_grammar_atom_type_has_an_operator_kind() -> None:
    from orion.schemas.grammar import AtomType

    assert set(ATOM_TYPE_OPERATOR_KIND.keys()) == set(AtomType.__args__)


# --- input rules ----------------------------------------------------------


def test_presence_atom_ignores_stamped_fields() -> None:
    # cortex-exec stamps confidence/salience/uncertainty per atom type; two
    # different stamps must gate identically.
    a = route_atom(
        source_service="orion-cortex-exec",
        atom=_atom("exec_step_started", "action_candidate", confidence=1.0, salience=0.85, uncertainty=0.0),
    )
    b = route_atom(
        source_service="orion-cortex-exec",
        atom=_atom("exec_step_started", "action_candidate", confidence=0.2, salience=0.1, uncertainty=0.9),
    )
    assert a.input_kind == b.input_kind == "presence"
    assert a.level == b.level == PRESENCE_LEVEL
    assert a.site_index == 2


def test_harness_governor_routes_to_seat_four_as_presence() -> None:
    a = route_atom(source_service="orion-harness-governor", atom=_atom("harness_fcc_step", "reasoning_step"))
    assert a.site_index == 4
    assert a.input_kind == "presence"


def test_measured_biometrics_gauge_passes_salience_through() -> None:
    a = route_atom(source_service="orion-biometrics", atom=_atom("gpu_pressure_signal", salience=0.37, confidence=0.9))
    assert a.input_kind == "measured"
    assert a.level == pytest.approx(0.37)
    assert a.site_index == 1


def test_measured_gauge_at_rest_reads_zero() -> None:
    a = route_atom(source_service="orion-biometrics", atom=_atom("thermal_pressure_signal", salience=0.0))
    assert a.level == 0.0


def test_stability_is_inverted_so_calm_reads_zero() -> None:
    calm = route_atom(source_service="orion-biometrics", atom=_atom("stability_signal", salience=1.0))
    volatile = route_atom(source_service="orion-biometrics", atom=_atom("stability_signal", salience=0.2))
    assert calm.level == 0.0
    assert volatile.level == pytest.approx(0.8)


@pytest.mark.parametrize(
    "role",
    [
        "telemetry_sample",  # default: clockwork
        "node_context",
        "memory_pressure_signal",  # structural floor
        "disk_capacity_pressure_signal",
        "cabinet_uv_activity_signal",
        "body_state",
    ],
)
def test_biometrics_skips_clockwork_and_floored_gauges(role: str) -> None:
    with pytest.raises(SkippedInputError):
        route_atom(source_service="orion-biometrics", atom=_atom(role, salience=0.5))


@pytest.mark.parametrize(
    "role", ["bus_observer_tick_started", "bus_observer_tick_completed", "bus_health_observed", "bus_census_computed"]
)
def test_bus_clockwork_is_skipped(role: str) -> None:
    with pytest.raises(SkippedInputError):
        route_atom(source_service="orion-bus", atom=_atom(role, confidence=1.0, salience=0.5, uncertainty=0.08))


def test_bus_activity_level_subtracts_the_calm_floor() -> None:
    def level(abs_z: float) -> float:
        return route_atom(
            source_service="orion-bus",
            atom=_atom("bus_activity_zscore_computed", confidence=0.9, salience=0.6, uncertainty=abs_z / 3.0),
        ).level

    assert level(0.0) == 0.0
    assert level(ABS_Z_CALM_FLOOR) == pytest.approx(0.0, abs=1e-9)
    assert level(0.5) == 0.0  # below the calm floor reads rest, not negative
    assert level(3.0) == pytest.approx(1.0)
    assert 0.0 < level(2.0) < 1.0


def test_bus_activity_level_calm_stream_mostly_reads_rest() -> None:
    # Theory anchor: for a calm z ~ N(0,1), P(|z| < sqrt(2/pi)) ~= 0.575.
    import random

    rng = random.Random(7)
    levels = []
    for _ in range(4000):
        z = rng.gauss(0.0, 1.0)
        levels.append(
            route_atom(
                source_service="orion-bus",
                atom=_atom("bus_activity_zscore_computed", confidence=0.9, uncertainty=min(1.0, abs(z) / 3.0)),
            ).level
        )
    rest_share = sum(1 for v in levels if v == 0.0) / len(levels)
    assert 0.52 < rest_share < 0.63
    assert math.fsum(levels) / len(levels) < 0.12


def test_bus_activity_without_baseline_is_skipped() -> None:
    # confidence 0.5 = tracker has no baseline yet; uncertainty is a stamped 0.75.
    with pytest.raises(SkippedInputError):
        route_atom(
            source_service="orion-bus",
            atom=_atom("bus_activity_zscore_computed", confidence=0.5, uncertainty=0.75),
        )


def test_rpc_timeout_is_presence() -> None:
    a = route_atom(source_service="orion-bus", atom=_atom("rpc_transport_timeout", "uncertainty_marker"))
    assert a.input_kind == "presence"
    assert a.site_index == 3


def test_hub_session_context_is_a_skipped_duplicate_and_utterance_is_presence() -> None:
    with pytest.raises(SkippedInputError):
        route_atom(source_service="orion-hub", atom=_atom("session_context", "entity"))
    u = route_atom(source_service="orion-hub", atom=_atom("user_utterance", "observation", salience=1.0))
    assert u.input_kind == "presence"
    r = route_atom(source_service="orion-hub", atom=_atom("repair_pressure_reading", "observation", salience=0.13))
    assert r.input_kind == "measured" and r.level == pytest.approx(0.13)


def test_measured_role_missing_its_field_is_skipped_not_defaulted() -> None:
    with pytest.raises(SkippedInputError):
        route_atom(source_service="orion-biometrics", atom=_atom("gpu_pressure_signal"))


def test_measured_role_with_junk_field_raises_value_error() -> None:
    with pytest.raises(ValueError):
        route_atom(source_service="orion-biometrics", atom=_atom("gpu_pressure_signal", salience="hot"))


def test_unlisted_role_falls_back_to_service_default() -> None:
    rule, listed = input_rule("orion-cortex-exec", "brand_new_role")
    assert not listed and rule.kind == "presence"
    rule, listed = input_rule("orion-biometrics", "brand_new_gauge")
    assert not listed and rule.kind == "skip"
    rule, listed = input_rule("orion-bus", "bus_activity_zscore_computed")
    assert listed and rule.kind == "measured"


def test_route_atom_rejects_unlisted_organ() -> None:
    with pytest.raises(UnroutableOrganError):
        route_atom(source_service="orion-sql-writer", atom=_atom("storage_write_window_observed", "observation"))


def test_route_atom_rejects_unknown_atom_type() -> None:
    with pytest.raises(UnroutableAtomTypeError):
        route_atom(source_service="orion-hub", atom=_atom("user_utterance", "not_a_real_atom_type"))
