"""Grammar coverage for Athena cabinet ambient audio levels."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for _name in list(sys.modules):
    if _name == "app" or _name.startswith("app."):
        del sys.modules[_name]
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from app.grammar_emit import build_biometrics_node_grammar_events  # noqa: E402
from app.node_catalog import NodeCatalog  # noqa: E402
from orion.schemas.telemetry.biometrics import (  # noqa: E402
    BiometricsInductionV1,
    BiometricsSampleV1,
    BiometricsSummaryV1,
)
from orion.substrate.biometrics_loop.grammar_extract import (  # noqa: E402
    extract_node_state_from_events,
)


FIXED_TS = datetime(2026, 8, 25, 5, 0, 0, tzinfo=timezone.utc)
CATALOG = NodeCatalog.load(REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml")


def _atoms(*, pressure: float | None = None, ambient_audio: dict | None = None):
    pressures = {}
    if pressure is not None:
        pressures["cabinet_ambient_audio_activity"] = pressure
    events = build_biometrics_node_grammar_events(
        sample=BiometricsSampleV1(
            timestamp=FIXED_TS,
            node="athena",
            cpu={"util": 0.1},
            ambient_audio=ambient_audio,
        ),
        summary=BiometricsSummaryV1(
            timestamp=FIXED_TS,
            node="athena",
            pressures=pressures,
            composites={"strain": 0.1},
        ),
        induction=BiometricsInductionV1(timestamp=FIXED_TS, node="athena"),
        node_profile=CATALOG.resolve("athena"),
        source_channel="orion:biometrics:induction",
    )
    return {
        event.atom.semantic_role: event.atom
        for event in events
        if event.atom is not None
    }, events


def test_emits_ambient_audio_activity_only_when_pressure_present() -> None:
    atoms, _ = _atoms(pressure=0.63)
    assert atoms["cabinet_ambient_audio_activity_signal"].salience == pytest.approx(0.63)

    absent_atoms, _ = _atoms()
    assert "cabinet_ambient_audio_activity_signal" not in absent_atoms


@pytest.mark.parametrize(("stale", "salience"), [(False, 0.0), (True, 1.0)])
def test_emits_ambient_audio_staleness_when_sample_field_present(
    stale: bool,
    salience: float,
) -> None:
    atoms, _ = _atoms(
        ambient_audio={
            "rms": 412.3,
            "peak": 1820,
            "received_at": "2026-08-25T05:00:00.000Z",
            "stale": stale,
        }
    )
    assert atoms["cabinet_ambient_audio_staleness_signal"].salience == salience

    absent_atoms, _ = _atoms()
    assert "cabinet_ambient_audio_staleness_signal" not in absent_atoms


def test_ambient_audio_atoms_connect_sample_and_capability() -> None:
    atoms, events = _atoms(
        pressure=0.63,
        ambient_audio={
            "rms": 412.3,
            "peak": 1820,
            "received_at": "2026-08-25T05:00:00.000Z",
            "stale": False,
        },
    )
    edges = {
        (event.edge.from_atom_id, event.edge.to_atom_id, event.edge.relation_type)
        for event in events
        if event.edge is not None
    }
    sample_id = atoms["telemetry_sample"].atom_id
    capability_id = atoms["capability_surface"].atom_id
    for role in (
        "cabinet_ambient_audio_activity_signal",
        "cabinet_ambient_audio_staleness_signal",
    ):
        atom_id = atoms[role].atom_id
        assert (sample_id, atom_id, "derived_from") in edges
        assert (atom_id, capability_id, "influenced") in edges


def test_ambient_audio_atoms_reach_substrate_pressure_hints() -> None:
    _, events = _atoms(
        pressure=0.63,
        ambient_audio={
            "rms": 412.3,
            "peak": 1820,
            "received_at": "2026-08-25T05:00:00.000Z",
            "stale": True,
        },
    )

    state = extract_node_state_from_events(
        events,
        CATALOG,
        stale_after_sec=180,
        now=FIXED_TS,
    )

    assert state.pressure_hints["cabinet_ambient_audio_activity"] == pytest.approx(0.63)
    assert state.pressure_hints["cabinet_ambient_audio_staleness"] == pytest.approx(1.0)
