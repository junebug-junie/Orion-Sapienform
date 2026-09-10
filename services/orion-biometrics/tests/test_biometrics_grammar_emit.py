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


def _fixtures(
    node: str,
    *,
    strain: float = 0.42,
    pressures: dict | None = None,
    stability: float | None = None,
):
    sample = BiometricsSampleV1(timestamp=FIXED_TS, node=node, cpu={"util": 0.1})
    composites = {"strain": strain}
    if stability is not None:
        composites["stability"] = stability
    summary = BiometricsSummaryV1(
        timestamp=FIXED_TS,
        node=node,
        pressures=pressures or {},
        composites=composites,
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


def test_stability_signal_carries_composite_value(catalog: NodeCatalog) -> None:
    # stability lives in `composites`, not `pressures`, like strain/body_state
    # above -- NOT in the memory/thermal/disk trio's `pressures` dict.
    sample, summary, induction = _fixtures("atlas", stability=0.91)
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
    assert atoms_by_role["stability_signal"].salience == pytest.approx(0.91)


def test_stability_signal_uncertainty_scales_with_low_not_high_salience(
    catalog: NodeCatalog,
) -> None:
    # Regression for a real review finding: _apply_biometrics_atom_uncertainty's
    # `sal * 0.5` scaling (used for every *_pressure_signal/*_activity_signal
    # atom) would be backwards for stability_signal, since stability's alarming
    # extreme is LOW salience (volatile), not high (calm) -- the opposite of
    # every other physical_substrate signal. A low-stability reading must get
    # >= uncertainty than a high-stability reading, not less.
    profile = catalog.resolve("atlas")

    sample_volatile, summary_volatile, induction_volatile = _fixtures(
        "atlas", stability=0.05
    )
    events_volatile = build_biometrics_node_grammar_events(
        sample=sample_volatile,
        summary=summary_volatile,
        induction=induction_volatile,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    volatile_atom = next(
        e.atom for e in events_volatile if e.atom and e.atom.semantic_role == "stability_signal"
    )

    sample_calm, summary_calm, induction_calm = _fixtures("atlas", stability=0.95)
    events_calm = build_biometrics_node_grammar_events(
        sample=sample_calm,
        summary=summary_calm,
        induction=induction_calm,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    calm_atom = next(
        e.atom for e in events_calm if e.atom and e.atom.semantic_role == "stability_signal"
    )

    assert volatile_atom.uncertainty > calm_atom.uncertainty
    # base = max(telemetry_error_rate=0.0, induction "cpu" volatility=0.1) = 0.1
    # (both fixtures use _fixtures()'s fixed induction metrics, see above).
    assert volatile_atom.uncertainty == pytest.approx(0.475)  # max(0.1, (1-0.05)*0.5)
    assert calm_atom.uncertainty == pytest.approx(0.1)  # max(0.1, (1-0.95)*0.5) -- base wins


def test_stability_signal_defaults_to_half_when_absent(catalog: NodeCatalog) -> None:
    # Matches _stability_from_induction()'s own no-data fallback (0.5, a
    # neutral prior) -- not 0.0 like the *_pressure trio, since 0.0 would
    # falsely claim "maximally volatile" rather than "unknown."
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
    assert atoms_by_role["stability_signal"].salience == 0.5


def test_gpu_pressure_signal_carries_real_gpu_util_not_hardcoded_capability_salience(
    catalog: NodeCatalog,
) -> None:
    # Regression for the bug where node:atlas's gpu_pressure field channel read a
    # flat 0.8 for ~42,400 consecutive real ticks: `capability_surface`'s salience
    # is an unconditional hardcoded 0.8 (see grammar_emit.py's capability_surface
    # atom), not a telemetry sample. The dedicated gpu_pressure_signal atom must
    # carry the real computed `gpu_util` value from
    # orion/telemetry/biometrics_pipeline.py's `pressures` dict instead.
    sample, summary, induction = _fixtures("atlas", pressures={"gpu_util": 0.27})
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
    assert atoms_by_role["gpu_pressure_signal"].salience == pytest.approx(0.27)
    # capability_surface stays fixed at its own hardcoded value -- unaffected.
    assert atoms_by_role["capability_surface"].salience == pytest.approx(0.8)


def test_gpu_pressure_signal_defaults_to_zero_when_absent(
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
    assert atoms_by_role["gpu_pressure_signal"].salience == 0.0


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


def test_all_atoms_emit_non_null_uncertainty(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas", strain=0.42)
    summary.telemetry_error_rate = 0.11
    induction.metrics["cpu"] = BiometricsInductionMetricV1(volatility=0.33)
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    atoms = [e.atom for e in events if e.atom is not None]
    assert atoms
    for atom in atoms:
        assert atom.uncertainty is not None
    body = next(a for a in atoms if a.semantic_role == "body_state")
    assert body.uncertainty >= 0.11
