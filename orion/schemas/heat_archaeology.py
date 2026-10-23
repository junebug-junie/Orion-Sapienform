"""Typed result of heat archaeology: where a node's watts went, since when, and on what evidence.

Produced by ``orion.physiology.heat_archaeology.analyze`` (pure) and returned to Orion by the
orion-introspect ``power_attribution`` tool (answered by orion-hub). Every number here is a
correlation over hourly averages, not a measurement of cause: ``verified`` stays False until an
intervention (stop the suspect, watch the watts) confirms it, and nothing in this module can
set it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# What the fit treats a GPU-pool holder as. A durable-run hold is a reservation: its compute
# shows up as the child leases it starts (cortex-exec, harness), so fitting both would count
# the same watts twice. Measured 2026-10-23 on 694 h: holds carried ~0 marginal W.
FeatureKind = Literal["lease", "regime", "intercept"]


class EvidenceRefV1(BaseModel):
    """One claim's backing: which query, which window, which values it rests on."""

    model_config = ConfigDict(extra="forbid")
    claim: str
    query: Literal["energy_hourly", "biometrics_hourly", "pool_busy_hours"]
    values: dict[str, float | int | str | None] = Field(default_factory=dict)


class PowerStepV1(BaseModel):
    """A sustained change in hourly wall watts, found by binary segmentation."""

    model_config = ConfigDict(extra="forbid")
    at: datetime
    before_w: float
    after_w: float
    delta_w: float
    before_hours: int
    after_hours: int
    # Separation of the two segment means in pooled standard errors. Hours are autocorrelated,
    # so this overstates certainty; it is a ranking/threshold number, not a p-value.
    t_stat: float
    # Hourly-average composition change across the step (None = no biometrics reading).
    gpu_delta_w: float | None = None
    cpu_delta_w: float | None = None
    rest_delta_w: float | None = None
    load_15m_before: float | None = None
    load_15m_after: float | None = None


class FitTermV1(BaseModel):
    """One column of the fit. A lease column counts lease-hours, not card-hours: a 2-card lease
    counts the same as a 1-card one, so watts_per_unit mixes card counts within a group. ``se``
    is corrected for lag-1 autocorrelation of residuals and column (Bartlett AR(1) factor)."""

    model_config = ConfigDict(extra="forbid")
    name: str
    kind: FeatureKind
    watts_per_unit: float
    se: float | None = None
    # Busy-hours per hour (lease), fraction of hours on (regime), 1 (intercept).
    mean_level: float


class WorkloadFitV1(BaseModel):
    """Least squares of one hourly series on per-workload busy-hours, regimes and an intercept."""

    model_config = ConfigDict(extra="forbid")
    target: Literal["wall_w", "gpu_w"]
    n_hours: int
    r2: float
    residual_sd_w: float
    terms: list[FitTermV1]
    dropped: list[str] = Field(default_factory=list)


class StepAttributionV1(BaseModel):
    """How the fit splits one change's delta across the workloads that changed with it.

    ``anchor`` says what the change is: a step in wall watts, or a CPU-load shift (a crisp
    clock even when hourly wall watts wander too much for a wall step to clear its bar).
    ``delta_w`` is measured on the fit's own hours (those with a CPU-load reading), so it can
    differ slightly from the wall step's own delta when biometrics have gaps.
    """

    model_config = ConfigDict(extra="forbid")
    step_at: datetime
    anchor: Literal["wall_step", "load_shift"] = "wall_step"
    before_w: float | None = None
    after_w: float | None = None
    delta_w: float
    contributions_w: dict[str, float]
    unexplained_w: float


class RegimeShiftV1(BaseModel):
    """A step in CPU load or CPU watts. GPU leases cannot explain these: look at host processes
    and container restart counts."""

    model_config = ConfigDict(extra="forbid")
    series: Literal["load_15m", "cpu_w"]
    at: datetime
    before: float
    after: float
    delta: float
    # The neighbouring change points (or window edges): the span the before/after means cover.
    segment_start: datetime | None = None
    segment_end: datetime | None = None


class NodeHeatArchaeologyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str
    window_start: datetime
    window_end: datetime
    hours_with_wall: int
    mean_wall_w: float | None = None
    daily_wall_w: list[tuple[str, float]] = Field(default_factory=list)
    steps: list[PowerStepV1] = Field(default_factory=list)
    fit: WorkloadFitV1 | None = None
    # Same columns fitted to GPU watts (pool node only); quality fields only, terms=[].
    gpu_fit: WorkloadFitV1 | None = None
    # Why there is no fit, when there is none (no CPU-load readings, no workload columns).
    fit_note: str | None = None
    attributions: list[StepAttributionV1] = Field(default_factory=list)
    regime_shifts: list[RegimeShiftV1] = Field(default_factory=list)
    # Durable-run holds per workflow: mean busy-hours per hour over the window. Reported, never
    # fitted (see FeatureKind).
    reservations: dict[str, float] = Field(default_factory=dict)
    load_regime_threshold: float | None = None
    evidence: list[EvidenceRefV1] = Field(default_factory=list)
    summary: str
    caveats: list[str] = Field(default_factory=list)
    verified: Literal[False] = False
