"""Room sensors are not Orion's machine body (Juniper, 2026-11-12).

athena's ``cabinet_*`` pressure hints (cabinet sensor board + ambient microphone) read the
room around the machines. Live 2026-11-12 they were 8 of the 26 keys averaged into
``node:substrate.biometrics``. These tests pin that the two generic readers of node hints
-- the biometrics surprise number and the pressure organ -- no longer see them, while the
projection still carries them for the named consumers (field digester, spike consumer).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.prediction_error_definitions import prediction_error_definition_version
from orion.substrate.biometrics_loop.grammar_extract import _CABINET_ROLE_TO_HINT
from orion.substrate.biometrics_loop.pressure_organ import (
    build_absence_trigger_event,
    invoke_biometrics_pressure,
)
from orion.substrate.biometrics_loop.room_hints import (
    ROOM_HINT_KEYS,
    is_room_hint,
    machine_body_hints,
)
from orion.substrate.cabinet_ambient_spike_consumer import HINT_KEY as SPIKE_HINT_KEY
from orion.substrate.prediction_error import _THRESHOLD, biometrics_prediction_error

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 12, 3, 0, 0, tzinfo=timezone.utc)

# athena's real hints, substrate_node_biometrics_projection 2026-11-12 03:30Z.
ATHENA_BODY = {
    "gpu": 0.0,
    "strain": 0.1857,
    "stability": 0.9799,
    "fan_pressure": 0.59,
    "disk_pressure": 0.0314,
    "power_pressure": 0.3139,
    "memory_pressure": 0.1886,
    "thermal_pressure": 0.6,
    "disk_capacity_pressure": 0.8131,
}
ATHENA_ROOM = {
    "cabinet_em_activity": 0.2896,
    "cabinet_uv_activity": 0.0,
    "cabinet_climate_activity": 0.1563,
    "cabinet_sensor_staleness": 0.0,
    "cabinet_proximity_activity": 0.2980,
    "cabinet_vibration_activity": 0.0433,
    "cabinet_ambient_audio_activity": 0.2719,
    "cabinet_ambient_audio_staleness": 0.0,
}


def _proj(nodes: dict[str, dict]) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=T0,
        nodes={
            n: NodeBiometricsStateV1(node_id=n, last_seen_at=T0, pressure_hints=dict(h))
            for n, h in nodes.items()
        },
    )


def test_every_cabinet_hint_the_extractor_writes_is_a_room_key() -> None:
    assert set(_CABINET_ROLE_TO_HINT.values()) == set(ROOM_HINT_KEYS)
    assert SPIKE_HINT_KEY in ROOM_HINT_KEYS
    assert set(ATHENA_ROOM) <= ROOM_HINT_KEYS
    assert not any(is_room_hint(k) for k in ATHENA_BODY)


def test_every_cabinet_pressure_orion_biometrics_emits_is_a_room_key() -> None:
    """The producer's list lives in another service; read it without importing it."""
    src = (REPO / "services/orion-biometrics/app/grammar_emit.py").read_text()
    tree = ast.parse(src)
    specs = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_CABINET_PRESSURE_SPECS"
    )
    pressure_keys = {elt.elts[0].value for elt in specs.elts}
    roles = {elt.elts[1].value for elt in specs.elts}
    assert pressure_keys and pressure_keys <= ROOM_HINT_KEYS
    assert roles <= set(_CABINET_ROLE_TO_HINT)


def test_a_new_cabinet_key_is_still_kept_out() -> None:
    assert is_room_hint("cabinet_co2_activity")
    assert machine_body_hints({"cabinet_co2_activity": 0.9, "gpu": 0.1}) == {"gpu": 0.1}


def test_room_only_change_reads_calm() -> None:
    """Someone walks past the cabinet and talks: the machines did not change."""
    prev = _proj({"athena": {**ATHENA_BODY, **ATHENA_ROOM}})
    loud = {**ATHENA_ROOM, "cabinet_proximity_activity": 0.95,
            "cabinet_ambient_audio_activity": 0.9, "cabinet_vibration_activity": 0.7}
    curr = _proj({"athena": {**ATHENA_BODY, **loud}})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_body_change_is_not_diluted_by_room_keys() -> None:
    prev = _proj({"athena": {**ATHENA_BODY, **ATHENA_ROOM}, "circe": dict(ATHENA_BODY)})
    curr_body = {**ATHENA_BODY, "gpu": 0.18}
    curr = _proj({"athena": {**curr_body, **ATHENA_ROOM}, "circe": dict(ATHENA_BODY)})
    # 18 machine keys (9 + 9), one moved by 0.18. Before v3 the 8 room keys made it 26.
    assert biometrics_prediction_error(prev, curr) == pytest.approx((0.18 / 18) / _THRESHOLD)


def test_room_hints_stay_on_the_projection() -> None:
    """Retired from the body reading, not deleted: field digester and Hub read them."""
    curr = _proj({"athena": {**ATHENA_BODY, **ATHENA_ROOM}})
    assert set(ATHENA_ROOM) <= set(curr.nodes["athena"].pressure_hints)


def test_definition_version_is_three() -> None:
    assert prediction_error_definition_version("node_biometrics") == 3


@pytest.fixture
def catalog(tmp_path: Path) -> NodeCatalog:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "nodes:\n"
        "  roomhost:\n"
        "    role: orchestration\n"
        "    expected_online: true\n"
    )
    return NodeCatalog.load(path)


def _organ_roles(catalog: NodeCatalog, hints: dict, active: list[str] | None = None) -> set[str]:
    node_bio = NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=T0,
        nodes={
            "roomhost": NodeBiometricsStateV1(
                node_id="roomhost",
                last_seen_at=T0 - timedelta(seconds=5),
                pressure_hints=hints,
            )
        },
    )
    emission = invoke_biometrics_pressure(
        trigger_event=build_absence_trigger_event("roomhost", now=T0),
        node_bio=node_bio,
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="active_node_pressure_projection",
            generated_at=T0,
            nodes={
                "roomhost": ActiveNodePressureStateV1(
                    node_id="roomhost",
                    availability_status="online",
                    active_pressures=list(active),
                    last_updated_at=T0 - timedelta(minutes=5),
                )
            }
            if active
            else {},
        ),
        catalog=catalog,
        now=T0,
    )
    return {ev.atom.semantic_role for ev in emission.candidate_events if ev.atom}


def test_a_loud_room_alone_is_not_node_pressure(catalog: NodeCatalog) -> None:
    calm_body = {"strain": 0.1, "gpu": 0.0, "memory_pressure": 0.2}
    assert _organ_roles(catalog, calm_body) == set()
    loud_room = {"cabinet_proximity_activity": 0.97, "cabinet_ambient_audio_activity": 0.9}
    assert _organ_roles(catalog, {**calm_body, **loud_room}) == set()


def test_a_strained_body_is_still_node_pressure(catalog: NodeCatalog) -> None:
    roles = _organ_roles(catalog, {"strain": 0.8, **ATHENA_ROOM})
    assert "node_pressure_detected" in roles


def test_room_held_strain_decays_after_the_change(catalog: NodeCatalog) -> None:
    """Deploy transition: athena holds strain that only a loud room kept alive. With a
    calm body the organ now lets it decay instead of reinforcing it."""
    hints = {"strain": 0.1, "cabinet_proximity_activity": 0.97}
    roles = _organ_roles(catalog, hints, active=["strain"])
    assert "node_pressure_decayed" in roles
    assert "node_pressure_reinforced" not in roles
