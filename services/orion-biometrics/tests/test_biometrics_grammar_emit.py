from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest

from app.grammar_emit import build_biometrics_node_grammar_events
from app.node_catalog import NodeCatalog
from orion.schemas.grammar import AtomType, RelationType
from orion.schemas.telemetry.biometrics import (
    BiometricsInductionMetricV1,
    BiometricsInductionV1,
    BiometricsSampleV1,
    BiometricsSummaryV1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 20, 6, 18, 624380, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _fixtures(node: str, *, strain: float = 0.42, pressures: dict | None = None):
    sample = BiometricsSampleV1(timestamp=FIXED_TS, node=node, cpu={"util": 0.1})
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


def test_builds_node_scoped_trace_for_atlas(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
        code_version="0.1.0",
    )
    assert events
    assert all(e.trace_id.startswith("biometrics.node:atlas:") for e in events)
    atoms = [e.atom for e in events if e.atom]
    node_context = next(a for a in atoms if a.semantic_role == "node_context")
    assert node_context.text_value == "atlas"
    assert "capability" in node_context.dimensions


def test_uses_allowed_atom_types_only(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    allowed = set(get_args(AtomType))
    for event in events:
        if event.atom:
            assert event.atom.atom_type in allowed


def test_uses_allowed_relation_types_only(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    allowed = set(get_args(RelationType))
    for event in events:
        if event.edge:
            assert event.edge.relation_type in allowed


def test_athena_capability_surface_mentions_graphdb_not_heavy_llm(
    catalog: NodeCatalog,
) -> None:
    sample, summary, induction = _fixtures("athena")
    profile = catalog.resolve("athena")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    cap = next(
        e.atom
        for e in events
        if e.atom and e.atom.semantic_role == "capability_surface"
    )
    assert "graphdb" in cap.summary
    assert "local_llm_heavy" not in cap.summary


def test_trace_has_start_atoms_edges_end(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("prometheous")
    profile = catalog.resolve("prometheous")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    kinds = [e.event_kind for e in events]
    assert kinds[0] == "trace_started"
    assert "atom_emitted" in kinds
    assert "edge_emitted" in kinds
    assert kinds[-1] == "trace_ended"
    assert events[0].trace_id.startswith("biometrics.node:prometheus:")


def test_memory_thermal_disk_pressure_signals_carry_individual_values(
    catalog: NodeCatalog,
) -> None:
    sample, summary, induction = _fixtures(
        "atlas", pressures={"mem": 0.61, "thermal": 0.33, "disk": 0.12, "cpu": 0.9}
    )
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    atoms_by_role = {
        e.atom.semantic_role: e.atom for e in events if e.atom is not None
    }
    assert atoms_by_role["memory_pressure_signal"].salience == pytest.approx(0.61)
    assert atoms_by_role["thermal_pressure_signal"].salience == pytest.approx(0.33)
    assert atoms_by_role["disk_pressure_signal"].salience == pytest.approx(0.12)
    # strain/gpu remain the composite/capability-derived values, unaffected.
    assert atoms_by_role["body_state"].salience == pytest.approx(summary.composites["strain"])


def test_memory_thermal_disk_pressure_signals_default_to_zero_when_absent(
    catalog: NodeCatalog,
) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    atoms_by_role = {
        e.atom.semantic_role: e.atom for e in events if e.atom is not None
    }
    assert atoms_by_role["memory_pressure_signal"].salience == 0.0
    assert atoms_by_role["thermal_pressure_signal"].salience == 0.0
    assert atoms_by_role["disk_pressure_signal"].salience == 0.0


def test_circe_node_availability_reflects_expected_offline(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("circe")
    profile = catalog.resolve("circe")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    avail = next(
        e.atom
        for e in events
        if e.atom and e.atom.semantic_role == "node_availability"
    )
    assert "expected offline" in avail.summary.lower()
