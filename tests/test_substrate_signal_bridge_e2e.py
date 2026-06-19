"""End-to-end: bridge → MoleculeJsonlStore → harness → daily rollup."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.experiment.daily_rollup import compute_daily_rollup
from orion.substrate.experiment.harness import SubstrateExperimentHarness
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.signal_bridge import signal_to_molecule


def _signal(*, signal_id: str, signal_kind: str, dimensions: dict[str, float]) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id=signal_id,
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions,
        causal_parents=[],
        source_event_id=f"corr-{signal_id}",
        observed_at=now,
        emitted_at=now,
        summary=f"{signal_kind} {signal_id}",
        notes=[],
    )


def test_bridged_signals_land_in_daily_rollup_under_cortex_exec(tmp_path):
    store = MoleculeJsonlStore(tmp_path / "molecules.jsonl")
    harness = SubstrateExperimentHarness()

    raw_signals = [
        _signal(signal_id="s-ok", signal_kind="cognition_run",
                dimensions={"success": 1.0, "step_count": 0.10, "latency_level": 0.20}),
        _signal(signal_id="s-fail", signal_kind="cognition_run",
                dimensions={"success": 0.0, "step_count": 0.40, "latency_level": 0.80}),
        _signal(signal_id="s-step-err", signal_kind="cognition_step",
                dimensions={"success": 0.0, "latency_level": 0.30, "error_present": 1.0, "service_count": 0.40}),
    ]

    for raw in raw_signals:
        molecule = signal_to_molecule(raw)
        store.add(molecule)
        harness.record_emit(molecule, organ=raw.organ_id)

    today = datetime.now(timezone.utc).date()
    metrics = compute_daily_rollup(day=today, harness=harness, store=store)

    assert metrics.organ_coverage.by_organ.get("cortex_exec") == 3

    contradiction_stat = next(
        g for g in metrics.gradient_distribution if g.key == "contradiction"
    )
    assert contradiction_stat.max == 1.0
    assert contradiction_stat.mean > 0.0

    salience_stat = next(
        g for g in metrics.gradient_distribution if g.key == "salience"
    )
    assert salience_stat.max == pytest.approx(0.80)

    assert len(metrics.contradiction_clusters) >= 1
    cluster = metrics.contradiction_clusters[0]
    assert "signal" in cluster.shared_atoms
    assert cluster.contradiction_sum >= 1.0
