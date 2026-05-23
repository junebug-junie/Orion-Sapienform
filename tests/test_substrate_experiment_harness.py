"""End-to-end smoke for the experiment harness, daily rollup, and weekly report."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from orion.autonomy import substrate_emit as autonomy_emit
from orion.mind import substrate_emit as mind_emit
from orion.substrate.experiment import (
    SubstrateExperimentHarness,
    compute_daily_rollup,
    generate_week_report,
    write_daily_rollup,
)
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.operators import (
    amplify_contradiction,
    decay_molecule,
    find_resonant_molecules,
    reinforce_molecule,
)


def _seed_one_day(
    *,
    day: date,
    store: MoleculeJsonlStore,
    harness: SubstrateExperimentHarness,
) -> None:
    when = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc)
    obs = mind_emit.emit_observation(surface_text="hi")
    obs.created_at = when
    obs.last_touched_at = when
    pressure = autonomy_emit.emit_pressure(label="goal:learn", magnitude=0.5)
    pressure.created_at = when
    pressure.last_touched_at = when

    store.add(obs)
    store.add(pressure)
    harness.record_emit(obs, organ="mind", when=when)
    harness.record_emit(pressure, organ="autonomy", when=when)

    # Reinforce + contradiction so the harness sees gradient events.
    delta_reinforce = reinforce_molecule(obs)
    harness.record_gradient_delta(delta_reinforce, when=when)
    delta_contradict = amplify_contradiction(obs)
    harness.record_gradient_delta(delta_contradict, when=when)
    delta_decay = decay_molecule(pressure)
    harness.record_decay(delta_decay)

    # Cross-organ traversal: autonomy looks up molecules with salience pressure.
    hits = find_resonant_molecules(
        store.all(), gradients=["salience"], threshold=0.1
    )
    harness.record_traversal(
        query_gradients=["salience"],
        threshold=0.1,
        results=hits,
        requesting_organ="autonomy",
        when=when,
    )


def test_full_harness_to_report(tmp_path):
    store = MoleculeJsonlStore(tmp_path / "molecules.jsonl")
    harness = SubstrateExperimentHarness()

    start = date(2026, 5, 17)
    end = date(2026, 5, 23)

    runs_dir = tmp_path / "runs"
    for offset in range(7):
        day = start + timedelta(days=offset)
        _seed_one_day(day=day, store=store, harness=harness)
        metrics = compute_daily_rollup(day=day, harness=harness, store=store)
        write_daily_rollup(metrics, runs_dir=runs_dir)
        # Two organs always present on every day.
        assert metrics.organ_coverage.total() >= 2
        assert metrics.molecule_count >= 2

    report_path = tmp_path / "weekly.md"
    body = generate_week_report(
        start_date=start,
        end_date=end,
        runs_dir=runs_dir,
        out_path=report_path,
    )
    assert "Substrate Experiment Report" in body
    assert "Daily snapshot" in body
    assert report_path.exists()
    assert "salience" not in body.split("End-of-window gradient distribution")[0]


def test_empty_report_when_no_rollups(tmp_path):
    body = generate_week_report(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        runs_dir=tmp_path / "missing",
    )
    assert "No daily rollups" in body


def test_orphan_rate_drops_with_traversal(tmp_path):
    """A molecule never touched should be counted as orphan; touched ones not."""

    store = MoleculeJsonlStore(tmp_path / "molecules.jsonl")
    harness = SubstrateExperimentHarness()

    day = date(2026, 5, 23)
    when = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc)

    touched = mind_emit.emit_observation(surface_text="touched")
    orphan = mind_emit.emit_observation(surface_text="orphan")
    for molecule in (touched, orphan):
        molecule.created_at = when
        molecule.last_touched_at = when
        store.add(molecule)
        harness.record_emit(molecule, organ="mind", when=when)

    harness.record_gradient_delta(reinforce_molecule(touched), when=when)

    metrics = compute_daily_rollup(day=day, harness=harness, store=store)
    assert metrics.molecule_count == 2
    assert 0.0 < metrics.orphan_molecule_rate <= 0.5 + 1e-9
