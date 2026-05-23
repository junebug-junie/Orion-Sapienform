"""Metric models for the 7-day substrate harness.

These are plain Pydantic models so daily rollups can be written/loaded as JSON
without ceremony.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from orion.schema_kernel import DEFAULT_GRADIENT_KEYS


class GradientStats(BaseModel):
    key: str
    min: float = 0.0
    mean: float = 0.0
    max: float = 0.0


class OrganCoverage(BaseModel):
    by_organ: dict[str, int] = Field(default_factory=dict)

    def total(self) -> int:
        return sum(self.by_organ.values())


class ContradictionCluster(BaseModel):
    shared_atoms: list[str]
    molecule_ids: list[str]
    contradiction_sum: float


class DailyMetricsV1(BaseModel):
    """One day's rollup of substrate activity."""

    day: date

    molecule_count: int = 0
    organ_coverage: OrganCoverage = Field(default_factory=OrganCoverage)
    gradient_distribution: list[GradientStats] = Field(default_factory=list)

    resonance_hits: int = 0
    cross_organ_reuse: int = 0
    reinforcement_count: int = 0
    decay_count: int = 0

    contradiction_clusters: list[ContradictionCluster] = Field(default_factory=list)

    orphan_molecule_rate: float = 0.0
    substrate_health_score: float = 0.0

    @staticmethod
    def empty(day: date) -> "DailyMetricsV1":
        return DailyMetricsV1(
            day=day,
            gradient_distribution=[GradientStats(key=key) for key in DEFAULT_GRADIENT_KEYS],
        )
