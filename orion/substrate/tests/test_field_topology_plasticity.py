from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.causal_geometry import CausalGeometryDivergenceEntryV1, CausalGeometrySnapshotV1
from orion.schemas.field_state import FieldEdgeV1
from orion.substrate.mutation_contracts import CONTRACTS
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.field_topology_plasticity import (
    MIN_MEANINGFUL_DELTA,
    MUTATION_CLASS,
    edge_ref_for,
    propose_field_topology_patches,
)


def _snapshot(divergence: list[CausalGeometryDivergenceEntryV1]) -> CausalGeometrySnapshotV1:
    now = datetime.now(timezone.utc)
    return CausalGeometrySnapshotV1(
        snapshot_id="snap-1",
        generated_at=now,
        window_start=now,
        window_end=now,
        divergence=divergence,
    )


def _cap_cap_edge(source: str, target: str) -> FieldEdgeV1:
    return FieldEdgeV1(source_id=source, target_id=target, edge_type="capability_capability", weight=0.4)


def _node_cap_edge(source: str, target: str) -> FieldEdgeV1:
    return FieldEdgeV1(source_id=source, target_id=target, edge_type="node_capability", weight=0.4)


def test_proposal_generated_for_diverging_cap_cap_edge_with_clamped_delta() -> None:
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:reasoning",
            target_id="cap:memory",
            observed_strength=0.55,
            designed_weight=0.40,
            delta=0.15,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    field_edges = [_cap_cap_edge("cap:reasoning", "cap:memory")]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.mutation_class == MUTATION_CLASS
    assert proposal.target_surface == "field_topology"
    assert proposal.lane == "operational"
    bounds = CONTRACTS[MUTATION_CLASS].bounds["edge_weight_delta"]
    expected_delta = max(bounds[0], min(bounds[1], 0.55 - 0.40))
    assert proposal.patch.target_ref == edge_ref_for("cap:reasoning", "cap:memory")
    assert proposal.patch.patch["edge_weight_delta"] == expected_delta
    assert proposal.patch.rollback_payload == {"edge_weight_delta": 0.0}
    assert proposal.patch.mutation_class == MUTATION_CLASS
    assert proposal.evidence_refs
    assert proposal.source_signal_ids


def test_below_threshold_divergence_produces_no_proposal() -> None:
    tiny_delta = MIN_MEANINGFUL_DELTA / 2
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:a",
            target_id="cap:b",
            observed_strength=0.40 + tiny_delta,
            designed_weight=0.40,
            delta=tiny_delta,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    field_edges = [_cap_cap_edge("cap:a", "cap:b")]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert proposals == []


def test_suffixed_target_id_from_real_divergence_builder_still_joins_to_cap_cap_edge() -> None:
    """Regression: causal_geometry_report.build_divergence() always appends
    '#<capability_channel>' to target_id for "both"/"designed_only" entries (to
    disambiguate multiple capability channels aliasing to the same physical YAML
    edge). The cap->cap join must strip that suffix before matching against
    FieldEdgeV1.target_id, which never carries it -- otherwise every real snapshot
    produces zero candidates regardless of how much real divergence exists."""
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:transport",
            target_id="cap:orchestration#reasoning_pressure",
            observed_strength=0.9,
            designed_weight=0.4,
            delta=0.5,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    field_edges = [_cap_cap_edge("cap:transport", "cap:orchestration")]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert len(proposals) == 1
    assert proposals[0].patch.target_ref == edge_ref_for("cap:transport", "cap:orchestration")


def test_non_cap_cap_edge_is_excluded_even_with_large_divergence() -> None:
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="node:atlas",
            target_id="cap:memory",
            observed_strength=0.9,
            designed_weight=0.1,
            delta=0.8,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    # This pair exists in field_edges but as a node_capability edge, not capability_capability.
    field_edges = [_node_cap_edge("node:atlas", "cap:memory")]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert proposals == []


def test_wildly_divergent_edge_still_clamps_to_contract_bounds() -> None:
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:x",
            target_id="cap:y",
            observed_strength=1.0,
            designed_weight=0.0,
            delta=1.0,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    field_edges = [_cap_cap_edge("cap:x", "cap:y")]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert len(proposals) == 1
    lo, hi = CONTRACTS[MUTATION_CLASS].bounds["edge_weight_delta"]
    delta = proposals[0].patch.patch["edge_weight_delta"]
    assert lo <= delta <= hi
    assert delta == hi  # raw delta of 1.0 clamps to the upper bound


def test_non_both_status_entries_are_skipped() -> None:
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:a",
            target_id="cap:b",
            observed_strength=0.9,
            designed_weight=None,
            delta=None,
            status="observed_only",
        ),
        CausalGeometryDivergenceEntryV1(
            source_id="cap:c",
            target_id="cap:d",
            observed_strength=None,
            designed_weight=0.2,
            delta=None,
            status="designed_only",
        ),
        CausalGeometryDivergenceEntryV1(
            source_id="cap:e",
            target_id="cap:f",
            observed_strength=None,
            designed_weight=None,
            delta=None,
            status="insufficient_data",
        ),
    ]
    snapshot = _snapshot(divergence)
    field_edges = [
        _cap_cap_edge("cap:a", "cap:b"),
        _cap_cap_edge("cap:c", "cap:d"),
        _cap_cap_edge("cap:e", "cap:f"),
    ]

    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)

    assert proposals == []


def test_run_trial_end_to_end_is_inconclusive_without_registered_replay_corpus() -> None:
    divergence = [
        CausalGeometryDivergenceEntryV1(
            source_id="cap:reasoning",
            target_id="cap:memory",
            observed_strength=0.55,
            designed_weight=0.40,
            delta=0.15,
            status="both",
        )
    ]
    snapshot = _snapshot(divergence)
    field_edges = [_cap_cap_edge("cap:reasoning", "cap:memory")]
    proposals = propose_field_topology_patches(snapshot, field_edges=field_edges)
    assert len(proposals) == 1
    proposal = proposals[0]

    runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(corpus_by_class={}, baseline_metric_ref_by_class={}),
    )
    trial = runner.run_trial(proposal=proposal, measured_metrics={"divergence_delta": 0.05})

    assert trial.status == "inconclusive"
    assert "missing_replay_corpus_or_baseline_metrics" in trial.notes


def test_new_modules_never_import_patch_applier() -> None:
    # Guardrail per the hard constraint: field_topology_weight_patch is HITL-adoption
    # only in this rung. Parse each new module's AST for an actual import statement
    # referencing mutation_apply / PatchApplier -- a future accidental auto-apply
    # wiring should fail this test immediately. (Prose mentions of "PatchApplier" in
    # docstrings, e.g. explaining *why* it's not used, are fine and expected; only a
    # real import is disallowed, so this checks imports specifically rather than
    # doing a substring scan of the whole file.)
    import orion.substrate.field_topology_plasticity as plasticity_module
    import orion.substrate.field_topology_learned_store as store_module

    for module in (plasticity_module, store_module):
        assert not _imports_patch_applier(Path(module.__file__))


def _imports_patch_applier(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("mutation_apply" in alias.name for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if "mutation_apply" in module_name:
                return True
            if any(alias.name == "PatchApplier" for alias in node.names):
                return True
    return False
