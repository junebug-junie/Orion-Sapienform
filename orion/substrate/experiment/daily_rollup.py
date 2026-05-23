"""Per-day rollup computation + JSON persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

from orion.schema_kernel import DEFAULT_GRADIENT_KEYS
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.molecules import SubstrateMoleculeV1

from .harness import SubstrateExperimentHarness, _DayBucket
from .metrics import (
    ContradictionCluster,
    DailyMetricsV1,
    GradientStats,
    OrganCoverage,
)


def _gradient_stats(molecules: list[SubstrateMoleculeV1]) -> list[GradientStats]:
    out: list[GradientStats] = []
    for key in DEFAULT_GRADIENT_KEYS:
        values = [m.gradient(key) for m in molecules]
        if not values:
            out.append(GradientStats(key=key))
            continue
        out.append(
            GradientStats(
                key=key,
                min=min(values),
                mean=mean(values),
                max=max(values),
            )
        )
    return out


def _contradiction_clusters(
    molecules: list[SubstrateMoleculeV1],
) -> list[ContradictionCluster]:
    """Cluster molecules with contradiction>0 by their atom signature."""

    grouped: dict[tuple[str, ...], list[SubstrateMoleculeV1]] = defaultdict(list)
    for molecule in molecules:
        if molecule.gradient("contradiction") <= 0.0:
            continue
        signature = tuple(sorted(molecule.atoms.values()))
        grouped[signature].append(molecule)
    clusters: list[ContradictionCluster] = []
    for signature, members in grouped.items():
        if not members:
            continue
        clusters.append(
            ContradictionCluster(
                shared_atoms=list(signature),
                molecule_ids=[m.molecule_id for m in members],
                contradiction_sum=sum(m.gradient("contradiction") for m in members),
            )
        )
    clusters.sort(key=lambda c: c.contradiction_sum, reverse=True)
    return clusters


def _health_score(
    *,
    coherence_mean: float,
    contradiction_mean: float,
    cross_organ_rate: float,
    reinforcement_rate: float,
    orphan_rate: float,
) -> float:
    score = (
        coherence_mean
        + cross_organ_rate
        + reinforcement_rate
        - contradiction_mean
        - orphan_rate
    )
    # Soft clamp into [-1, 1] purely for readability.
    return max(-1.0, min(1.0, score))


def compute_daily_rollup(
    *,
    day: date,
    harness: SubstrateExperimentHarness,
    store: MoleculeJsonlStore,
) -> DailyMetricsV1:
    """Compute a DailyMetricsV1 for ``day`` using harness events and store state."""

    bucket: _DayBucket | None = harness.bucket_for(day)
    metrics = DailyMetricsV1.empty(day)

    day_molecules = [
        molecule
        for molecule in store.all()
        if molecule.created_at.date() == day
    ]
    metrics.molecule_count = len(day_molecules)

    coverage = OrganCoverage()
    for molecule in day_molecules:
        organ = molecule.provenance.get("organ", "unknown")
        coverage.by_organ[organ] = coverage.by_organ.get(organ, 0) + 1
    metrics.organ_coverage = coverage

    metrics.gradient_distribution = _gradient_stats(day_molecules)
    metrics.contradiction_clusters = _contradiction_clusters(day_molecules)

    reinforcement_count = 0
    decay_count = 0
    if bucket is not None:
        metrics.resonance_hits = sum(len(t.hit_ids) for t in bucket.traversals)
        metrics.cross_organ_reuse = sum(
            len(ids) for ids in bucket.references_by_organ.values()
        )
        for gradient_record in bucket.gradient_changes:
            cause = gradient_record.cause
            if cause in {"reinforce", "stabilize", "contradiction"}:
                reinforcement_count += 1
            elif cause == "decay":
                decay_count += 1
        touched_ids = bucket.referenced_ids
    else:
        touched_ids = set()

    metrics.reinforcement_count = reinforcement_count
    metrics.decay_count = decay_count

    if metrics.molecule_count:
        orphans = sum(
            1 for molecule in day_molecules if molecule.molecule_id not in touched_ids
        )
        metrics.orphan_molecule_rate = orphans / metrics.molecule_count
    else:
        metrics.orphan_molecule_rate = 0.0

    coherence_stat = next(
        (g for g in metrics.gradient_distribution if g.key == "coherence"),
        None,
    )
    contradiction_stat = next(
        (g for g in metrics.gradient_distribution if g.key == "contradiction"),
        None,
    )
    coherence_mean = coherence_stat.mean if coherence_stat else 0.0
    contradiction_mean = contradiction_stat.mean if contradiction_stat else 0.0
    cross_organ_rate = (
        metrics.cross_organ_reuse / metrics.molecule_count
        if metrics.molecule_count
        else 0.0
    )
    reinforcement_rate = (
        metrics.reinforcement_count / metrics.molecule_count
        if metrics.molecule_count
        else 0.0
    )
    metrics.substrate_health_score = _health_score(
        coherence_mean=coherence_mean,
        contradiction_mean=contradiction_mean,
        cross_organ_rate=cross_organ_rate,
        reinforcement_rate=reinforcement_rate,
        orphan_rate=metrics.orphan_molecule_rate,
    )
    return metrics


def write_daily_rollup(
    metrics: DailyMetricsV1,
    *,
    runs_dir: str | Path,
) -> Path:
    """Write a daily rollup to ``runs_dir/YYYY-MM-DD.json`` and return the path."""

    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    target = runs_path / f"{metrics.day.isoformat()}.json"
    payload = metrics.model_dump(mode="json")
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target
