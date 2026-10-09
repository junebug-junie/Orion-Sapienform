"""House electricity contracts: usage, accrual, run cost, bills, reconcile, stakes.

UNKNOWN IS NEVER ZERO. A run whose settlement saw nothing, or a billing cycle with
no metered usage yet, carries None plus a gap reason -- never 0.0. A zero here would
tell a spend gate that a GPU run was free.

TWO RUN COSTS, NEVER MERGED. `estimated_run_cost_usd` prices what Orion's own meter
saw at the tariff's marginal rate; it is the only number autonomy may read.
`house_share_cost_usd` apportions the whole-house meter and is context only -- it
mixes in every other load in the house.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENERGY_USAGE_KIND = "energy.usage.observed.v1"
ENERGY_ACCRUED_KIND = "energy.cost.accrued.v1"
ENERGY_RUN_COST_KIND = "energy.run_cost.estimated.v1"
ENERGY_BILL_ACTUAL_KIND = "energy.bill.actual.v1"
ENERGY_BILL_FORECAST_KIND = "energy.bill.forecast.v1"
ENERGY_RECONCILE_KIND = "energy.reconcile.v1"
ENERGY_STAKES_KIND = "energy.stakes.snapshot.v1"
ENERGY_IMPORTER_STATUS_KIND = "energy.importer.status.v1"

EnergySource = Literal["rockymountain_power", "file_drop"]
EnergyBasis = Literal["incremental_over_baseline", "gross"]
RunCostGap = Literal["settlement_not_measured", "no_cycle_usage"]
HouseShareGap = Literal["settlement_not_measured", "house_interval_missing"]
CostBasis = Literal["pre_tax"]
ReconcileKind = Literal["actual", "forecast"]
ReconcileMethod = Literal["metered_period", "linear_run_rate"]
ReconcileGap = Literal["no_usage", "usage_incomplete"]
UtilityBasis = Literal["pre_tax", "tax_unknown"]
ImporterState = Literal["healthy", "stale", "reauth_required", "degraded"]
ImporterSource = Literal["portal", "file_drop"]
StakesPressure = Literal["unknown", "normal", "near_forecast", "over_forecast"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class EnergyUsageIntervalV1(BaseModel):
    """One metered interval of whole-house delivered energy (Green Button ESPI)."""

    model_config = ConfigDict(extra="forbid")

    source: EnergySource
    usage_point_id: str = Field(min_length=1)
    interval_start: datetime
    interval_end: datetime
    energy_kwh: float = Field(ge=0.0, allow_inf_nan=False)
    quality: Optional[str] = None
    # When the utility data was fetched. Late AMI corrections re-deliver the same
    # interval; the newer retrieved_at wins.
    retrieved_at: datetime
    source_file: Optional[str] = None

    @field_validator("interval_start", "interval_end", "retrieved_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def _ordered(self) -> "EnergyUsageIntervalV1":
        if self.interval_end <= self.interval_start:
            raise ValueError("interval_end must be after interval_start")
        return self

    @property
    def interval_seconds(self) -> int:
        return int((self.interval_end - self.interval_start).total_seconds())


class EnergyCostAccruedV1(BaseModel):
    """A usage interval priced at its position in the billing cycle's usage blocks."""

    model_config = ConfigDict(extra="forbid")

    usage_point_id: str = Field(min_length=1)
    interval_start: datetime
    interval_end: datetime
    energy_kwh: float = Field(ge=0.0)
    interval_cost_usd: float = Field(ge=0.0)
    # Rate for the NEXT kWh after this interval -- what one more hour of load costs now.
    marginal_usd_per_kwh: float = Field(ge=0.0)
    cycle_start: date
    cycle_accumulated_kwh: float = Field(ge=0.0)
    cycle_energy_cost_usd: float = Field(ge=0.0)
    # Energy so far plus the full month's fixed charges. Pre-tax.
    cycle_to_date_total_usd: float = Field(ge=0.0)
    tariff_version: str = Field(min_length=1)
    cost_basis: CostBasis = "pre_tax"
    computed_at: datetime

    @field_validator("interval_start", "interval_end", "computed_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        return _utc(value)


class EnergyRunCostEstimatedV1(BaseModel):
    """Dollar cost of one settled power intent. See module docstring for the two costs."""

    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(min_length=1)
    workload_kind: str
    node: str
    gpu_index: Optional[int] = None
    window_start: datetime
    window_end: datetime
    settlement_outcome: str

    energy_kwh: Optional[float] = Field(default=None, ge=0.0)
    energy_basis: Optional[EnergyBasis] = None

    estimated_run_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    marginal_usd_per_kwh: Optional[float] = Field(default=None, ge=0.0)
    cycle_kwh_basis: Optional[float] = Field(default=None, ge=0.0)
    cycle_kwh_basis_as_of: Optional[datetime] = None
    run_cost_gap: Optional[RunCostGap] = None

    house_share_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    house_kwh_overlap: Optional[float] = Field(default=None, ge=0.0)
    house_share_gap: Optional[HouseShareGap] = None

    tariff_version: Optional[str] = None
    cost_basis: CostBasis = "pre_tax"
    computed_at: datetime

    @field_validator("window_start", "window_end", "cycle_kwh_basis_as_of", "computed_at")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _null_means_reason(self) -> "EnergyRunCostEstimatedV1":
        if (self.estimated_run_cost_usd is None) == (self.run_cost_gap is None):
            raise ValueError("exactly one of estimated_run_cost_usd / run_cost_gap must be set")
        if (self.house_share_cost_usd is None) == (self.house_share_gap is None):
            raise ValueError("exactly one of house_share_cost_usd / house_share_gap must be set")
        return self


class EnergyBillActualV1(BaseModel):
    """A closed Rocky Mountain Power billing period, as printed on the bill.

    Lines the bill does not show stay None -- a missing tax line is unknown, not $0.
    The period is [billing_period_start, billing_period_end) at local midnight.
    """

    model_config = ConfigDict(extra="forbid")

    source: EnergySource
    usage_point_id: Optional[str] = None
    billing_period_start: date
    billing_period_end: date
    kwh_billed: float = Field(ge=0.0, allow_inf_nan=False)
    energy_charge: Optional[float] = Field(default=None, allow_inf_nan=False)
    customer_charge: Optional[float] = Field(default=None, allow_inf_nan=False)
    adjustments: Optional[float] = Field(default=None, allow_inf_nan=False)
    fees: Optional[float] = Field(default=None, allow_inf_nan=False)
    taxes: Optional[float] = Field(default=None, allow_inf_nan=False)
    credits: Optional[float] = Field(default=None, allow_inf_nan=False)
    current_charges: float = Field(allow_inf_nan=False)
    amount_due: Optional[float] = Field(default=None, allow_inf_nan=False)
    due_date: Optional[date] = None
    statement_artifact_id: Optional[str] = None
    retrieved_at: datetime
    source_file: Optional[str] = None

    @field_validator("retrieved_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def _ordered(self) -> "EnergyBillActualV1":
        if self.billing_period_end <= self.billing_period_start:
            raise ValueError("billing_period_end must be after billing_period_start")
        return self


class EnergyBillForecastV1(BaseModel):
    """RMP's own in-cycle projection. A forecast with no projected number is rejected."""

    model_config = ConfigDict(extra="forbid")

    source: EnergySource
    usage_point_id: Optional[str] = None
    billing_period_start: date
    billing_period_end: Optional[date] = None
    as_of: datetime
    days_into_cycle: Optional[int] = Field(default=None, ge=0)
    projected_kwh: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    projected_total_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    retrieved_at: datetime
    source_file: Optional[str] = None

    @field_validator("as_of", "retrieved_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def _has_projection(self) -> "EnergyBillForecastV1":
        if self.projected_kwh is None and self.projected_total_usd is None:
            raise ValueError("forecast needs projected_kwh or projected_total_usd")
        if self.billing_period_end is not None and self.billing_period_end <= self.billing_period_start:
            raise ValueError("billing_period_end must be after billing_period_start")
        return self


class EnergyReconcileV1(BaseModel):
    """Orion's tariff estimate for a bill period vs what RMP billed or projects.

    Deltas are Orion minus utility. Orion is pre-tax; `utility_basis` says whether tax
    was removed from the utility number. A systematic miss is fixed by a tariff patch.

    `bucket_deltas` compare like with like against the printed bill: `energy_charge` is the
    block charges before riders; `energy_charge_plus_adjustments` adds every rider, including
    those on the customer charge; `customer_charge` is the prorated customer charge alone
    (all fixed charges for a tariff that does not mark one).
    """

    model_config = ConfigDict(extra="forbid")

    reconcile_kind: ReconcileKind
    usage_point_id: str = Field(min_length=1)
    billing_period_start: date
    billing_period_end: Optional[date] = None
    utility_as_of: datetime
    utility_kwh: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    utility_total_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    utility_basis: UtilityBasis
    orion_method: ReconcileMethod
    orion_covered_through: Optional[datetime] = None
    orion_kwh: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    orion_energy_usd: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    # May be negative: a per-bill credit (e.g. paperless -$0.50) can outweigh a very short
    # period's prorated fixed charges, as it can on the utility's own bill.
    orion_fixed_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    orion_total_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    reconcile_gap: Optional[ReconcileGap] = None
    delta_kwh: Optional[float] = Field(default=None, allow_inf_nan=False)
    delta_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    delta_pct: Optional[float] = Field(default=None, allow_inf_nan=False)
    bucket_deltas: dict[str, float] = Field(default_factory=dict)
    tariff_version: str = Field(min_length=1)
    cost_basis: CostBasis = "pre_tax"
    computed_at: datetime

    @field_validator("utility_as_of", "orion_covered_through", "computed_at")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        return None if value is None else _utc(value)

    @field_validator("bucket_deltas")
    @classmethod
    def _finite_bucket_deltas(cls, value: dict[str, float]) -> dict[str, float]:
        for key, amount in value.items():
            if not math.isfinite(amount):
                raise ValueError(f"bucket_deltas[{key!r}] must be finite")
        return value

    @model_validator(mode="after")
    def _null_means_reason(self) -> "EnergyReconcileV1":
        if (self.orion_total_usd is None) == (self.reconcile_gap is None):
            raise ValueError("exactly one of orion_total_usd / reconcile_gap must be set")
        if self.reconcile_gap is not None and (
            self.delta_kwh is not None
            or self.delta_usd is not None
            or self.delta_pct is not None
            or self.bucket_deltas
        ):
            raise ValueError("a reconcile with a gap cannot carry deltas")
        return self


class EnergyStakesSnapshotV1(BaseModel):
    """What the house bill looks like right now, for spend gates and the Hub.

    A projection of already-gated inputs (metered usage, tariff, RMP forecast, importer
    health) -- not a new signal. `pressure` is `unknown` whenever an input is missing or
    stale; a gate must never hold on unknown.
    """

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    usage_point_id: Optional[str] = None
    cycle_start: Optional[date] = None
    cycle_end: Optional[date] = None
    covered_through: Optional[datetime] = None
    cycle_accumulated_kwh: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    cycle_to_date_total_usd: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    marginal_usd_per_kwh: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    orion_projected_total_usd: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    forecast_total_usd: Optional[float] = Field(default=None, allow_inf_nan=False)
    forecast_as_of: Optional[datetime] = None
    projected_to_forecast_ratio: Optional[float] = Field(default=None, allow_inf_nan=False)
    importer_state: ImporterState
    pressure: StakesPressure
    pressure_reason: str = Field(min_length=1)
    tariff_version: Optional[str] = None

    @field_validator("as_of", "covered_through", "forecast_as_of")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _compared_means_ratio(self) -> "EnergyStakesSnapshotV1":
        if self.pressure != "unknown" and self.importer_state != "healthy":
            raise ValueError("a compared pressure requires a healthy importer")
        if self.pressure != "unknown" and self.projected_to_forecast_ratio is None:
            raise ValueError("a compared pressure needs projected_to_forecast_ratio")
        return self


class EnergyImporterStatusV1(BaseModel):
    """Is house usage arriving? Silence is `stale`, never a quiet day of $0."""

    model_config = ConfigDict(extra="forbid")

    state: ImporterState
    reason: str = Field(min_length=1)
    source: ImporterSource
    last_success_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    latest_interval_end: Optional[datetime] = None
    usage_lag_hours: Optional[float] = Field(default=None, ge=0.0)
    as_of: datetime

    @field_validator("last_success_at", "last_attempt_at", "latest_interval_end", "as_of")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        return None if value is None else _utc(value)
