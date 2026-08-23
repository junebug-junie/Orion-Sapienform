from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
# Many services expose a top-level `app` package; clear any previously imported
# namespace so biometrics' app wins for this module.
for _name in list(sys.modules):
    if _name == "app" or _name.startswith("app."):
        del sys.modules[_name]
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from app.grammar_emit import build_biometrics_node_grammar_events  # noqa: E402
from app.node_catalog import NodeCatalog  # noqa: E402
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1  # noqa: E402
from orion.schemas.telemetry.biometrics import (  # noqa: E402
    BiometricsInductionMetricV1,
    BiometricsInductionV1,
    BiometricsSampleV1,
    BiometricsSummaryV1,
)
from orion.substrate.biometrics_loop.grammar_extract import (  # noqa: E402
    extract_node_state_from_events,
)

CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)

CABINET_PRESSURE_KEYS = (
    "cabinet_climate_activity",
    "cabinet_particulate_activity",
    "cabinet_em_activity",
    "cabinet_uv_activity",
    "cabinet_vibration_activity",
    "cabinet_proximity_activity",
)

CABINET_SIGNAL_ROLES = tuple(f"{key}_signal" for key in CABINET_PRESSURE_KEYS)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _fixtures(
    node: str,
    *,
    pressures: dict[str, float] | None = None,
    sensors: dict | None = None,
    strain: float = 0.42,
):
    sample = BiometricsSampleV1(
        timestamp=FIXED_TS,
        node=node,
        cpu={"util": 0.1},
        sensors=sensors,
    )
    summary = BiometricsSummaryV1(
        timestamp=FIXED_TS,
        node=node,
        pressures=pressures or {},
        composites={"strain": strain},
        telemetry_error_rate=0.0,
    )
    induction = BiometricsInductionV1(
        timestamp=FIXED_TS,
        node=node,
        metrics={
            "cpu": BiometricsInductionMetricV1(
                level=0.5, trend=0.5, volatility=0.1, spike_rate=0.0
            )
        },
    )
    return sample, summary, induction


def _emit(catalog: NodeCatalog, **fixture_kwargs):
    sample, summary, induction = _fixtures("athena", **fixture_kwargs)
    profile = catalog.resolve("athena")
    return build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )


def _atoms_by_role(events) -> dict[str, GrammarAtomV1]:
    return {e.atom.semantic_role: e.atom for e in events if e.atom is not None}


def _provenance() -> GrammarProvenanceV1:
    return GrammarProvenanceV1(
        source_service="orion-biometrics",
        source_component="biometrics_grammar_emit",
    )


def _atom_event(*, trace_id: str, role: str, salience: float) -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{role}",
        trace_id=trace_id,
        atom_type="signal",
        semantic_role=role,
        layer="biometrics",
        summary="test",
        salience=salience,
    )
    return GrammarEventV1(
        event_id=f"gev_{role}",
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=atom,
        provenance=_provenance(),
    )


def test_host_grammar_unchanged_when_cabinet_pressures_absent(
    catalog: NodeCatalog,
) -> None:
    events = _emit(catalog, pressures={"mem": 0.61, "thermal": 0.33})
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert "memory_pressure_signal" in roles
    assert "thermal_pressure_signal" in roles
    assert not any(role.startswith("cabinet_") for role in roles)


def test_emits_cabinet_activity_atoms_only_when_pressure_key_present(
    catalog: NodeCatalog,
) -> None:
    pressures = {
        "cabinet_climate_activity": 0.42,
        "cabinet_em_activity": 0.17,
    }
    atoms = _atoms_by_role(_emit(catalog, pressures=pressures))
    assert atoms["cabinet_climate_activity_signal"].salience == pytest.approx(0.42)
    assert atoms["cabinet_em_activity_signal"].salience == pytest.approx(0.17)
    assert "cabinet_particulate_activity_signal" not in atoms
    assert "cabinet_uv_activity_signal" not in atoms


def test_emits_all_cabinet_activity_atoms_when_all_keys_present(
    catalog: NodeCatalog,
) -> None:
    pressures = {key: 0.1 * (idx + 1) for idx, key in enumerate(CABINET_PRESSURE_KEYS)}
    atoms = _atoms_by_role(_emit(catalog, pressures=pressures))
    for role in CABINET_SIGNAL_ROLES:
        key = role.removesuffix("_signal")
        assert atoms[role].salience == pytest.approx(pressures[key])


def test_cabinet_staleness_atom_emitted_only_when_sensors_present(
    catalog: NodeCatalog,
) -> None:
    fresh_atoms = _atoms_by_role(
        _emit(catalog, sensors={"frame": {}, "received_at": "2026-08-23T12:00:00Z", "stale": False})
    )
    assert fresh_atoms["cabinet_sensor_staleness_signal"].salience == 0.0

    stale_atoms = _atoms_by_role(
        _emit(catalog, sensors={"frame": {}, "received_at": "2026-08-23T12:00:00Z", "stale": True})
    )
    assert stale_atoms["cabinet_sensor_staleness_signal"].salience == 1.0

    no_sensor_atoms = _atoms_by_role(_emit(catalog))
    assert "cabinet_sensor_staleness_signal" not in no_sensor_atoms


def test_cabinet_signals_have_edges_from_telemetry_sample_and_to_capability_surface(
    catalog: NodeCatalog,
) -> None:
    pressures = {"cabinet_climate_activity": 0.5}
    events = _emit(
        catalog,
        pressures=pressures,
        sensors={"frame": {}, "received_at": "2026-08-23T12:00:00Z", "stale": False},
    )
    atoms = _atoms_by_role(events)
    edges = [
        (e.edge.from_atom_id, e.edge.to_atom_id, e.edge.relation_type)
        for e in events
        if e.edge is not None
    ]
    sample_id = atoms["telemetry_sample"].atom_id
    cap_id = atoms["capability_surface"].atom_id
    climate_id = atoms["cabinet_climate_activity_signal"].atom_id
    staleness_id = atoms["cabinet_sensor_staleness_signal"].atom_id

    assert (sample_id, climate_id, "derived_from") in edges
    assert (climate_id, cap_id, "influenced") in edges
    assert (sample_id, staleness_id, "derived_from") in edges
    assert (staleness_id, cap_id, "influenced") in edges


def test_extract_maps_cabinet_signal_roles_to_exact_pressure_hints(
    catalog: NodeCatalog,
) -> None:
    trace_id = "biometrics.node:athena:2026-08-23T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            role="cabinet_climate_activity_signal",
            salience=0.42,
        ),
        _atom_event(
            trace_id=trace_id,
            role="cabinet_particulate_activity_signal",
            salience=0.31,
        ),
        _atom_event(
            trace_id=trace_id,
            role="cabinet_sensor_staleness_signal",
            salience=1.0,
        ),
    ]
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=FIXED_TS)
    assert state.pressure_hints["cabinet_climate_activity"] == pytest.approx(0.42)
    assert state.pressure_hints["cabinet_particulate_activity"] == pytest.approx(0.31)
    assert state.pressure_hints["cabinet_sensor_staleness"] == pytest.approx(1.0)


def test_extract_ignores_cabinet_signals_with_no_salience(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:athena:2026-08-23T12:00:00Z"
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:cabinet_uv_activity_signal",
        trace_id=trace_id,
        atom_type="signal",
        semantic_role="cabinet_uv_activity_signal",
        layer="biometrics",
        summary="test",
        salience=None,
    )
    events = [
        GrammarEventV1(
            event_id="gev_uv",
            event_kind="atom_emitted",
            trace_id=trace_id,
            emitted_at=FIXED_TS,
            observed_at=FIXED_TS,
            atom=atom,
            provenance=_provenance(),
        )
    ]
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=FIXED_TS)
    assert "cabinet_uv_activity" not in state.pressure_hints
