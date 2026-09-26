"""House electricity contracts: metered usage, tariff accrual, and run cost.

UNKNOWN IS NEVER ZERO. A run whose settlement saw nothing, or a billing cycle with
no metered usage yet, carries None plus a gap reason -- never 0.0. A zero here would
tell a spend gate that a GPU run was free.

TWO RUN COSTS, NEVER MERGED. `estimated_run_cost_usd` prices what Orion's own meter
saw at the tariff's marginal rate; it is the only number autonomy may read.
`house_share_cost_usd` apportions the whole-house meter and is context only -- it
mixes in every other load in the house.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENERGY_USAGE_KIND = "energy.usage.observed.v1"
ENERGY_ACCRUED_KIND = "energy.cost.accrued.v1"
ENERGY_RUN_COST_KIND = "energy.run_cost.estimated.v1"

EnergySource = Literal["rockymountain_power", "file_drop"]
EnergyBasis = Literal["incremental_over_baseline", "gross"]
RunCostGap = Literal["settlement_not_measured", "no_cycle_usage"]
HouseShareGap = Literal["settlement_not_measured", "house_interval_missing"]
CostBasis = Literal["pre_tax"]


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
    energy_kwh: float = Field(ge=0.0)
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
