# Orion Energy Watcher — Plan 1 (cost primitive) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a dropped Rocky Mountain Power Green Button XML file into hourly house kWh, tariff-priced cycle cost, and a dollar estimate for every settled GPU power intent — all on the bus and in Postgres.

**Architecture:** Pure logic lives in a shared package `orion/energy/` (ESPI parser, versioned tariff, cycle ledger, run-cost join) so Plan 2 (portal scraper, bill reconcile, curiosity stakes) reuses it. A thin new service `services/orion-energy/` watches a drop directory, subscribes to `orion:power:intent:settled`, and publishes three new event kinds. `orion-sql-writer` persists them with idempotent upserts. No portal, no Hub strip, no attention consumer in this plan.

**Tech Stack:** Python 3.12, pydantic 2.9, pydantic-settings, stdlib `xml.etree.ElementTree` + `zoneinfo` (+ `tzdata` wheel in the slim image), PyYAML, redis bus via `orion.core.bus`, SQLAlchemy 2.0 + Postgres `ON CONFLICT` in sql-writer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md`

## Global Constraints

- Work only in worktree `/mnt/scripts/Orion-Sapienform-orion-energy-watcher`. Before the first code commit: `git branch -m docs/orion-energy-watcher feat/orion-energy-watcher`.
- Local test interpreter: `PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python` (system python3 has no pydantic). CI uses plain `python`.
- **Unknown is never zero.** Missing / unmeasured → `None` with an explicit gap reason. Never coerce to `0.0` (same rule as `PowerIntentV1`).
- **Two cost numbers, never merged.** `estimated_run_cost_usd` (Orion meters × tariff; the only number autonomy may ever read) and `house_share_cost_usd` (context only).
- **Marginal, not average.** Run cost is priced at the run's position in the billing cycle's usage blocks.
- **Incremental over baseline** when the settlement has a baseline; otherwise gross, and say which (`energy_basis`).
- Tariff numbers come only from the published RMP Utah Schedule 1 / Price Summary "In Effect as of August 10, 2026". Taxes are not modeled: `cost_basis: pre_tax`.
- Bus URL: `ORION_BUS_URL=redis://100.92.216.81:6379/0` (tailscale node IP, same as `orion-zwave`).
- Channels (exact): `orion:energy:usage:observed`, `orion:energy:cost:accrued`, `orion:energy:run_cost:estimated`.
- Kinds (exact): `energy.usage.observed.v1`, `energy.cost.accrued.v1`, `energy.run_cost.estimated.v1`.
- Never commit `.env`. After any `.env_example` change run `python scripts/sync_local_env_from_example.py` from the worktree root and report skipped keys.
- Compose host paths must be absolute (`scripts/check_compose_no_relative_mounts.py`).
- Docker via `scripts/safe_docker_build.sh <service> <args>` only.
- MFA / portal login / bill scraping / curiosity gating are **out of scope** (Plan 2).

## File Structure

| Path | Responsibility |
|---|---|
| `orion/schemas/energy.py` | The three event contracts + kind constants + gap/basis literals |
| `orion/schemas/registry.py` | Register the three schemas |
| `orion/bus/channels.yaml` | Catalog three channels; add `orion-energy` as consumer of `orion:power:intent:settled` |
| `config/metrics/metric_definitions.lock.json` | Re-locked by `check_definition_drift.py --update` |
| `tests/test_energy_bus_catalog.py` | Catalog ↔ registry alignment |
| `orion/energy/__init__.py` | Package marker |
| `orion/energy/espi.py` | Green Button ESPI XML → `EnergyUsageIntervalV1` list |
| `orion/energy/tariff.py` | Versioned block tariff: load, marginal rate, block-split energy cost |
| `config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml` | First `tariff_version` |
| `orion/energy/ledger.py` | In-memory usage ledger: late-data upsert, billing cycles, accrual, overlap lookup |
| `orion/energy/run_cost.py` | `PowerIntentSettledV1` + ledger → `EnergyRunCostEstimatedV1` |
| `orion/energy/tests/…` | Unit tests + ESPI fixture |
| `services/orion-sql-writer/app/models/energy.py` | Three SQL tables with natural-key unique constraints |
| `services/orion-sql-writer/app/energy_persist.py` | Newest-wins `ON CONFLICT` upserts |
| `services/orion-sql-writer/app/{models/__init__,worker,settings}.py`, `.env_example` | Route + subscribe wiring |
| `services/orion-sql-writer/tests/test_energy_sql_shape.py` | Columns, routing, subscription, upsert SQL |
| `services/orion-energy/` | New service: settings, inbox, pipeline, main, Dockerfile, compose, README, tests, evals |
| `.github/workflows/orion-energy-tests.yml` | CI for package + service |
| `.github/workflows/orion-sql-writer-tests.yml` | Add new sql-writer + catalog tests |

---

### Task 1: Energy contracts on the bus

**Files:**
- Create: `orion/schemas/energy.py`
- Modify: `orion/schemas/registry.py` (import near line 441; name map near line 1256; `SCHEMA_REGISTRY` near line 1507)
- Modify: `orion/bus/channels.yaml` (after the `orion:home:cooling:sample` entry ~line 1995; and the `orion:power:intent:settled` entry ~line 2753)
- Modify: `config/metrics/metric_definitions.lock.json` (regenerated, never hand-edited)
- Test: `tests/test_energy_bus_catalog.py`, `orion/energy/tests/test_energy_schemas.py`

**Interfaces:**
- Produces:
  - `EnergySource = Literal["rockymountain_power", "file_drop"]`
  - `EnergyBasis = Literal["incremental_over_baseline", "gross"]`
  - `RunCostGap = Literal["settlement_not_measured", "no_cycle_usage"]`
  - `HouseShareGap = Literal["settlement_not_measured", "house_interval_missing"]`
  - `ENERGY_USAGE_KIND`, `ENERGY_ACCRUED_KIND`, `ENERGY_RUN_COST_KIND` (str constants)
  - `EnergyUsageIntervalV1`, `EnergyCostAccruedV1`, `EnergyRunCostEstimatedV1` (pydantic models, fields below)

- [ ] **Step 1: Rename branch and write failing schema tests**

```bash
cd /mnt/scripts/Orion-Sapienform-orion-energy-watcher
git branch -m docs/orion-energy-watcher feat/orion-energy-watcher
mkdir -p orion/energy/tests/fixtures
```

Create `orion/energy/__init__.py`:

```python
"""House-level electricity: Green Button usage, tariff pricing, run-cost join."""
```

Create `orion/energy/tests/test_energy_schemas.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)

T0 = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)


def _usage(**kw) -> EnergyUsageIntervalV1:
    base = dict(
        source="file_drop",
        usage_point_id="UP123",
        interval_start=T0,
        interval_end=T1,
        energy_kwh=1.234,
        retrieved_at=T1,
    )
    base.update(kw)
    return EnergyUsageIntervalV1(**base)


def test_usage_interval_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        _usage(interval_end=T0)


def test_usage_interval_naive_datetimes_become_utc() -> None:
    iv = _usage(interval_start=datetime(2026, 9, 10, 18), interval_end=datetime(2026, 9, 10, 19))
    assert iv.interval_start.tzinfo is not None
    assert iv.interval_seconds == 3600


def test_usage_interval_rejects_negative_kwh() -> None:
    with pytest.raises(ValidationError):
        _usage(energy_kwh=-0.1)


def test_accrued_round_trips_json() -> None:
    acc = EnergyCostAccruedV1(
        usage_point_id="UP123",
        interval_start=T0,
        interval_end=T1,
        energy_kwh=1.0,
        interval_cost_usd=0.11,
        marginal_usd_per_kwh=0.11,
        cycle_start=date(2026, 9, 1),
        cycle_accumulated_kwh=101.0,
        cycle_energy_cost_usd=11.2,
        cycle_to_date_total_usd=23.36,
        tariff_version="rmp-ut-sch1-2026-08-10",
        computed_at=T1,
    )
    again = EnergyCostAccruedV1.model_validate(acc.model_dump(mode="json"))
    assert again == acc
    assert again.cost_basis == "pre_tax"


def _run(**kw) -> EnergyRunCostEstimatedV1:
    base = dict(
        intent_id="i-1",
        workload_kind="reverie.diffusion",
        node="circe",
        window_start=T0,
        window_end=T1,
        settlement_outcome="settled",
        computed_at=T1,
    )
    base.update(kw)
    return EnergyRunCostEstimatedV1(**base)


def test_run_cost_null_requires_gap_reason() -> None:
    with pytest.raises(ValidationError):
        _run(house_share_gap="house_interval_missing")  # run cost null, no run_cost_gap


def test_run_cost_value_forbids_gap_reason() -> None:
    with pytest.raises(ValidationError):
        _run(
            estimated_run_cost_usd=0.02,
            run_cost_gap="no_cycle_usage",
            house_share_gap="house_interval_missing",
        )


def test_run_cost_both_unknown_is_valid_with_reasons() -> None:
    est = _run(run_cost_gap="settlement_not_measured", house_share_gap="settlement_not_measured")
    assert est.estimated_run_cost_usd is None
    assert est.house_share_cost_usd is None
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_schemas.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.schemas.energy'`

- [ ] **Step 3: Write the contracts**

Create `orion/schemas/energy.py`:

```python
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
```

- [ ] **Step 4: Run schema tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_schemas.py -q`
Expected: 7 passed

- [ ] **Step 5: Write failing catalog test**

Create `tests/test_energy_bus_catalog.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("orion:energy:usage:observed", "EnergyUsageIntervalV1", "energy.usage.observed.v1", EnergyUsageIntervalV1),
    ("orion:energy:cost:accrued", "EnergyCostAccruedV1", "energy.cost.accrued.v1", EnergyCostAccruedV1),
    ("orion:energy:run_cost:estimated", "EnergyRunCostEstimatedV1", "energy.run_cost.estimated.v1", EnergyRunCostEstimatedV1),
]


def _channels() -> dict:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    return {c["name"]: c for c in raw["channels"]}


@pytest.mark.parametrize("channel,schema_id,kind,model", CASES)
def test_energy_channel_cataloged(channel, schema_id, kind, model) -> None:
    entry = _channels()[channel]
    assert entry["schema_id"] == schema_id
    assert entry["message_kind"] == kind
    assert "orion-energy" in entry["producer_services"]
    assert "orion-sql-writer" in entry["consumer_services"]


@pytest.mark.parametrize("channel,schema_id,kind,model", CASES)
def test_energy_schema_registered(channel, schema_id, kind, model) -> None:
    assert SCHEMA_REGISTRY[schema_id].kind == kind
    assert resolve(schema_id) is model


def test_energy_consumes_power_settlements() -> None:
    entry = _channels()["orion:power:intent:settled"]
    assert "orion-energy" in entry["consumer_services"]
```

Run: `PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py -q`
Expected: FAIL — `KeyError: 'orion:energy:usage:observed'`

- [ ] **Step 6: Register schemas and channels**

In `orion/schemas/registry.py`, next to `from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1` add:

```python
from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
```

Next to `"HomeCoolingSampleV1": HomeCoolingSampleV1,` (the plain name map ~line 1256) add:

```python
    "EnergyUsageIntervalV1": EnergyUsageIntervalV1,
    "EnergyCostAccruedV1": EnergyCostAccruedV1,
    "EnergyRunCostEstimatedV1": EnergyRunCostEstimatedV1,
```

Next to the `"HomeCoolingSampleV1": SchemaRegistration(...)` block in `SCHEMA_REGISTRY` add:

```python
    "EnergyUsageIntervalV1": SchemaRegistration(
        model=EnergyUsageIntervalV1,
        kind="energy.usage.observed.v1",
    ),
    "EnergyCostAccruedV1": SchemaRegistration(
        model=EnergyCostAccruedV1,
        kind="energy.cost.accrued.v1",
    ),
    "EnergyRunCostEstimatedV1": SchemaRegistration(
        model=EnergyRunCostEstimatedV1,
        kind="energy.run_cost.estimated.v1",
    ),
```

In `orion/bus/channels.yaml`, directly after the `orion:home:cooling:sample` entry, add:

```yaml
  # House-level electricity (docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md).
  # Whole-house Green Button intervals, priced by a versioned tariff, joined to settled
  # power intents. Unknown is never zero: run cost carries a gap reason instead.
  - name: "orion:energy:usage:observed"
    kind: "event"
    schema_id: "EnergyUsageIntervalV1"
    message_kind: "energy.usage.observed.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-26"
    description: "Whole-house metered kWh per interval from Rocky Mountain Power Green Button (file drop in Plan 1)."

  - name: "orion:energy:cost:accrued"
    kind: "event"
    schema_id: "EnergyCostAccruedV1"
    message_kind: "energy.cost.accrued.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-26"
    description: "Usage interval priced at its billing-cycle block position (pre-tax, versioned tariff)."

  - name: "orion:energy:run_cost:estimated"
    kind: "event"
    schema_id: "EnergyRunCostEstimatedV1"
    message_kind: "energy.run_cost.estimated.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-26"
    description: "Dollar cost of a settled power intent: meter-side estimate plus separately labeled house share."
```

In the `orion:power:intent:settled` entry change:

```yaml
    consumer_services: ["orion-sql-writer", "orion-energy"]
```

- [ ] **Step 7: Run catalog test and re-lock metric definitions**

```bash
PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py orion/energy/tests/test_energy_schemas.py -q
$PY scripts/check_definition_drift.py --update
$PY scripts/check_definition_drift.py --gate
$PY scripts/check_metric_lineage.py --gate
```

Expected: 7 + 7 passed; drift `--update` prints new `metric://bus_channel/orion-energy/...` entries; `--gate` PASS; lineage gate PASS. If lineage gate fails, read its message and add exactly what it names — do not suppress.

- [ ] **Step 8: Commit**

```bash
git add orion/schemas/energy.py orion/schemas/registry.py orion/bus/channels.yaml \
  config/metrics/metric_definitions.lock.json tests/test_energy_bus_catalog.py \
  orion/energy/__init__.py orion/energy/tests/test_energy_schemas.py
git diff --cached --check
git commit -m "feat(energy): contracts for house usage, accrual, and run cost"
```

---

### Task 2: Green Button ESPI parser

**Files:**
- Create: `orion/energy/espi.py`
- Create: `orion/energy/tests/fixtures/espi_two_flows.xml`
- Test: `orion/energy/tests/test_energy_espi.py`

**Interfaces:**
- Consumes: `EnergyUsageIntervalV1`, `EnergySource` (Task 1)
- Produces: `parse_espi(xml_bytes: bytes, *, retrieved_at: datetime, source: EnergySource, source_file: str | None = None) -> list[EnergyUsageIntervalV1]`; `class EspiError(ValueError)`

- [ ] **Step 1: Write the fixture**

Create `orion/energy/tests/fixtures/espi_two_flows.xml` (one forward-flow block with 3 hourly readings, one reverse-flow block that must be skipped):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:espi="http://naesb.org/espi">
  <id>urn:uuid:feed-1</id>
  <title>Green Button Usage Feed</title>
  <entry>
    <id>urn:uuid:up</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/RetailCustomer/9/UsagePoint/UP123"/>
    <content><espi:UsagePoint><espi:ServiceCategory><espi:kind>0</espi:kind></espi:ServiceCategory></espi:UsagePoint></content>
  </entry>
  <entry>
    <id>urn:uuid:mr1</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/RetailCustomer/9/UsagePoint/UP123/MeterReading/MR1"/>
    <link rel="related" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-FWD"/>
    <content><espi:MeterReading/></content>
  </entry>
  <entry>
    <id>urn:uuid:rtf</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-FWD"/>
    <content><espi:ReadingType>
      <espi:flowDirection>1</espi:flowDirection>
      <espi:intervalLength>3600</espi:intervalLength>
      <espi:powerOfTenMultiplier>0</espi:powerOfTenMultiplier>
      <espi:uom>72</espi:uom>
    </espi:ReadingType></content>
  </entry>
  <entry>
    <id>urn:uuid:ib1</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/RetailCustomer/9/UsagePoint/UP123/MeterReading/MR1/IntervalBlock/IB1"/>
    <content><espi:IntervalBlock>
      <espi:interval><espi:duration>10800</espi:duration><espi:start>1789063200</espi:start></espi:interval>
      <espi:IntervalReading><espi:timePeriod><espi:duration>3600</espi:duration><espi:start>1789063200</espi:start></espi:timePeriod><espi:value>1234</espi:value></espi:IntervalReading>
      <espi:IntervalReading><espi:timePeriod><espi:duration>3600</espi:duration><espi:start>1789066800</espi:start></espi:timePeriod><espi:value>500</espi:value></espi:IntervalReading>
      <espi:IntervalReading><espi:timePeriod><espi:duration>3600</espi:duration><espi:start>1789070400</espi:start></espi:timePeriod><espi:value>2000</espi:value></espi:IntervalReading>
    </espi:IntervalBlock></content>
  </entry>
  <entry>
    <id>urn:uuid:mr2</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/RetailCustomer/9/UsagePoint/UP123/MeterReading/MR2"/>
    <link rel="related" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-REV"/>
    <content><espi:MeterReading/></content>
  </entry>
  <entry>
    <id>urn:uuid:rtr</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-REV"/>
    <content><espi:ReadingType>
      <espi:flowDirection>19</espi:flowDirection>
      <espi:intervalLength>3600</espi:intervalLength>
      <espi:powerOfTenMultiplier>0</espi:powerOfTenMultiplier>
      <espi:uom>72</espi:uom>
    </espi:ReadingType></content>
  </entry>
  <entry>
    <id>urn:uuid:ib2</id>
    <link rel="self" href="https://csapps.example/espi/1_1/resource/RetailCustomer/9/UsagePoint/UP123/MeterReading/MR2/IntervalBlock/IB2"/>
    <content><espi:IntervalBlock>
      <espi:interval><espi:duration>3600</espi:duration><espi:start>1789063200</espi:start></espi:interval>
      <espi:IntervalReading><espi:timePeriod><espi:duration>3600</espi:duration><espi:start>1789063200</espi:start></espi:timePeriod><espi:value>999</espi:value></espi:IntervalReading>
    </espi:IntervalBlock></content>
  </entry>
</feed>
```

- [ ] **Step 2: Write failing tests**

Create `orion/energy/tests/test_energy_espi.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.energy.espi import EspiError, parse_espi

FIXTURE = Path(__file__).parent / "fixtures" / "espi_two_flows.xml"
RETRIEVED = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def test_parses_forward_flow_hourly_kwh() -> None:
    rows = parse_espi(FIXTURE.read_bytes(), retrieved_at=RETRIEVED, source="file_drop", source_file="x.xml")
    assert [r.energy_kwh for r in rows] == pytest.approx([1.234, 0.5, 2.0])
    assert rows[0].interval_start == datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    assert rows[0].interval_end == datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)
    assert {r.usage_point_id for r in rows} == {"UP123"}
    assert all(r.source == "file_drop" and r.retrieved_at == RETRIEVED for r in rows)
    assert rows[0].source_file == "x.xml"


def test_reverse_flow_block_is_skipped() -> None:
    rows = parse_espi(FIXTURE.read_bytes(), retrieved_at=RETRIEVED, source="file_drop")
    assert 0.999 not in [r.energy_kwh for r in rows]


def test_power_of_ten_multiplier_applied() -> None:
    # First occurrence is the forward-flow ReadingType (RT-FWD precedes RT-REV).
    xml = FIXTURE.read_bytes().replace(
        b"<espi:powerOfTenMultiplier>0<", b"<espi:powerOfTenMultiplier>-1<", 1
    )
    rows = parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
    assert rows[0].energy_kwh == pytest.approx(0.1234)


def test_unsupported_uom_raises() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:uom>72</espi:uom>", b"<espi:uom>38</espi:uom>")
    with pytest.raises(EspiError, match="uom"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_garbage_raises_espi_error() -> None:
    with pytest.raises(EspiError):
        parse_espi(b"<not-xml", retrieved_at=RETRIEVED, source="file_drop")


def test_feed_without_readings_raises() -> None:
    empty = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with pytest.raises(EspiError, match="no forward-flow"):
        parse_espi(empty, retrieved_at=RETRIEVED, source="file_drop")
```

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_espi.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.energy.espi'`

- [ ] **Step 3: Implement the parser**

Create `orion/energy/espi.py`:

```python
"""Green Button (NAESB ESPI) Atom feed -> whole-house usage intervals.

Linking follows the ESPI href convention: an IntervalBlock's self link is
``.../UsagePoint/<up>/MeterReading/<mr>/IntervalBlock/<ib>``; the MeterReading at
``.../UsagePoint/<up>/MeterReading/<mr>`` carries a ``related`` link to its
ReadingType, which holds unit, scale and flow direction. When a feed has exactly
one ReadingType and the link is absent, that ReadingType is used.

Only forward flow (delivered to the house, flowDirection 1) is kept. Reverse flow
(19, e.g. solar export) is a different quantity and is skipped, not netted.

Input is the operator's own utility export or Orion's own scraper output, parsed
with the stdlib parser (expat does not fetch external entities).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

from orion.schemas.energy import EnergySource, EnergyUsageIntervalV1

ATOM = "{http://www.w3.org/2005/Atom}"
ESPI = "{http://naesb.org/espi}"
UOM_WATT_HOURS = 72
FLOW_FORWARD = 1

_USAGE_POINT = re.compile(r"/UsagePoint/([^/]+)")
_METER_READING = re.compile(r"^(.*/UsagePoint/[^/]+/MeterReading/[^/]+)")


class EspiError(ValueError):
    """The file is not a usable ESPI usage feed."""


def _links(entry: ET.Element) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for link in entry.findall(f"{ATOM}link"):
        out.setdefault(link.get("rel", ""), []).append(link.get("href", ""))
    return out


def _int(el: Optional[ET.Element], default: Optional[int] = None) -> Optional[int]:
    if el is None or el.text is None or not el.text.strip():
        return default
    return int(el.text.strip())


def _reading_type(content: ET.Element) -> Optional[dict[str, int]]:
    rt = content.find(f"{ESPI}ReadingType")
    if rt is None:
        return None
    return {
        "uom": _int(rt.find(f"{ESPI}uom"), UOM_WATT_HOURS),
        "pow10": _int(rt.find(f"{ESPI}powerOfTenMultiplier"), 0),
        "flow": _int(rt.find(f"{ESPI}flowDirection"), FLOW_FORWARD),
    }


def parse_espi(
    xml_bytes: bytes,
    *,
    retrieved_at: datetime,
    source: EnergySource,
    source_file: Optional[str] = None,
) -> list[EnergyUsageIntervalV1]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise EspiError(f"not parseable XML: {exc}") from exc

    reading_types: dict[str, dict[str, int]] = {}
    meter_to_rt: dict[str, str] = {}
    blocks: list[tuple[str, ET.Element]] = []

    for entry in root.iter(f"{ATOM}entry"):
        content = entry.find(f"{ATOM}content")
        if content is None:
            continue
        links = _links(entry)
        self_href = (links.get("self") or [""])[0]
        rt = _reading_type(content)
        if rt is not None:
            reading_types[self_href] = rt
            continue
        if content.find(f"{ESPI}MeterReading") is not None:
            related = links.get("related") or []
            if related:
                meter_to_rt[self_href] = related[0]
            continue
        block = content.find(f"{ESPI}IntervalBlock")
        if block is not None:
            blocks.append((self_href, block))

    only_rt = next(iter(reading_types.values())) if len(reading_types) == 1 else None
    rows: list[EnergyUsageIntervalV1] = []
    for self_href, block in blocks:
        up_match = _USAGE_POINT.search(self_href)
        mr_match = _METER_READING.match(self_href)
        rt = None
        if mr_match is not None:
            rt = reading_types.get(meter_to_rt.get(mr_match.group(1), ""))
        rt = rt or only_rt
        if rt is None:
            raise EspiError(f"no ReadingType resolvable for block {self_href!r}")
        if rt["flow"] != FLOW_FORWARD:
            continue
        if rt["uom"] != UOM_WATT_HOURS:
            raise EspiError(f"unsupported uom {rt['uom']} (only 72 = Wh)")
        usage_point = up_match.group(1) if up_match else "unknown"
        scale = 10.0 ** rt["pow10"]
        for reading in block.findall(f"{ESPI}IntervalReading"):
            period = reading.find(f"{ESPI}timePeriod")
            start = _int(period.find(f"{ESPI}start")) if period is not None else None
            duration = _int(period.find(f"{ESPI}duration")) if period is not None else None
            value = _int(reading.find(f"{ESPI}value"))
            if start is None or not duration or value is None:
                continue
            begin = datetime.fromtimestamp(start, tz=timezone.utc)
            rows.append(
                EnergyUsageIntervalV1(
                    source=source,
                    usage_point_id=usage_point,
                    interval_start=begin,
                    interval_end=begin + timedelta(seconds=duration),
                    energy_kwh=value * scale / 1000.0,
                    retrieved_at=retrieved_at,
                    source_file=source_file,
                )
            )
    if not rows:
        raise EspiError("no forward-flow interval readings in feed")
    rows.sort(key=lambda r: (r.usage_point_id, r.interval_start))
    return rows
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_espi.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add orion/energy/espi.py orion/energy/tests/test_energy_espi.py orion/energy/tests/fixtures/espi_two_flows.xml
git commit -m "feat(energy): parse Green Button ESPI usage feeds"
```

---

### Task 3: Versioned tariff (Utah Schedule 1)

**Files:**
- Create: `orion/energy/tariff.py`
- Create: `config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml`
- Test: `orion/energy/tests/test_energy_tariff.py`

**Interfaces:**
- Produces:
  - `load_tariff(path: str | Path) -> Tariff`
  - `class TariffError(ValueError)`
  - `Tariff` (frozen dataclass): `version: str`, `cost_basis: str`, `energy_multiplier: float`, `fixed_monthly_usd: float`, `season_for(month: int) -> Season`, `energy_cost_usd(kwh: float, *, cycle_kwh_before: float, month: int) -> float`, `marginal_usd_per_kwh(*, cycle_kwh: float, month: int) -> float`
  - `Season(name: str, months: frozenset[int], blocks: tuple[Block, ...])`, `Block(up_to_kwh: float | None, usd_per_kwh: float)`

- [ ] **Step 1: Write the tariff YAML**

Create `config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml`:

```yaml
# Rocky Mountain Power, Utah Schedule 1 (Residential Service), standard non-TOU option.
# Every number below is copied from the published sources; nothing is estimated.
# Changing any number means a NEW file with a NEW tariff_version -- old accruals keep
# the version they were priced with.
tariff_version: rmp-ut-sch1-2026-08-10
utility: rocky_mountain_power
schedule: "Utah Schedule 1 - Residential Service (standard, non-TOU)"
effective_from: "2026-08-10"
sources:
  - "https://www.rockymountainpower.net/content/dam/pcorp/documents/en/rockymountainpower/rates-regulation/utah/rates/001_Residential_Service.pdf"
  - "https://www.rockymountainpower.net/content/dam/pcorp/documents/en/rockymountainpower/rates-regulation/utah/Utah_Price_Summary.pdf (In Effect as of August 10, 2026)"
cost_basis: pre_tax

seasons:
  summer:
    months: [6, 7, 8, 9]
    blocks:
      - {up_to_kwh: 400, cents_per_kwh: 9.8332}
      - {up_to_kwh: null, cents_per_kwh: 12.5263}
  winter:
    months: [10, 11, 12, 1, 2, 3, 4, 5]
    blocks:
      - {up_to_kwh: 400, cents_per_kwh: 8.7020}
      - {up_to_kwh: null, cents_per_kwh: 11.0852}

# Percent riders on the energy charge, from the Price Summary's Schedule 1 rows.
# Stage 1 multiplies the base energy charge. Stage 2 multiplies base + stage 1
# (Price Summary footnote 2: "Applies to Base Rate and Schedules 94 and 98").
energy_adjustments:
  on_base_pct:
    - {name: schedule_94_eba, pct: 7.63, column_verified: true}
    - {name: schedule_98_rec, pct: -0.53, column_verified: true}
  on_adjusted_pct:
    # Values are as printed on the Schedule 1 energy rows; which rider column each
    # belongs to is not legible in the extracted summary. The first real bill
    # reconciliation (Plan 2) confirms or corrects them.
    - {name: price_summary_rider_a, pct: 1.17, column_verified: false}
    - {name: price_summary_rider_b, pct: 3.84, column_verified: false}
    - {name: price_summary_rider_c, pct: 0.17, column_verified: false}

fixed_monthly:
  - {name: customer_charge_single_phase_single_family, usd: 12.00}
  - {name: schedule_91_lifeline_surcharge, usd: 0.16}

known_gaps:
  - "Sales and municipal taxes are not modeled; all costs are pre-tax."
  - "Percent riders on the customer charge are not modeled."
  - "Season is assigned by each interval's local calendar month, not RMP's billing month."
```

- [ ] **Step 2: Write failing tests**

Create `orion/energy/tests/test_energy_tariff.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.energy.tariff import TariffError, load_tariff

ROOT = Path(__file__).resolve().parents[3]
TARIFF = ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"
# (1 + 7.63% - 0.53%) * (1 + 1.17% + 3.84% + 0.17%), computed by hand.
M = 1.071 * 1.0518


def test_loads_published_numbers() -> None:
    t = load_tariff(TARIFF)
    assert t.version == "rmp-ut-sch1-2026-08-10"
    assert t.cost_basis == "pre_tax"
    assert t.energy_multiplier == pytest.approx(M)
    assert t.fixed_monthly_usd == pytest.approx(12.16)


def test_marginal_summer_first_block() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=0.0, month=7) == pytest.approx(0.098332 * M)


def test_marginal_at_block_boundary_is_second_block() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=400.0, month=9) == pytest.approx(0.125263 * M)


def test_marginal_winter() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=10.0, month=1) == pytest.approx(0.087020 * M)


def test_energy_cost_splits_across_block_boundary() -> None:
    t = load_tariff(TARIFF)
    cost = t.energy_cost_usd(10.0, cycle_kwh_before=395.0, month=7)
    assert cost == pytest.approx((5 * 0.098332 + 5 * 0.125263) * M)


def test_zero_kwh_costs_zero() -> None:
    t = load_tariff(TARIFF)
    assert t.energy_cost_usd(0.0, cycle_kwh_before=100.0, month=7) == 0.0


def test_rejects_month_coverage_gap(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["seasons"]["winter"]["months"] = [10, 11, 12, 1, 2, 3, 4]  # May missing
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="month"):
        load_tariff(bad)


def test_rejects_block_without_open_top(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["seasons"]["summer"]["blocks"][-1]["up_to_kwh"] = 1000
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="open"):
        load_tariff(bad)
```

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_tariff.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.energy.tariff'`

- [ ] **Step 3: Implement**

Create `orion/energy/tariff.py`:

```python
"""Deterministic block tariff: marginal $/kWh and block-split energy cost.

Blocks reset each billing cycle, so the price of the next kWh depends on how much
the house has already used this cycle. That position is the whole reason run cost
is priced here and not as kWh x an average rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


class TariffError(ValueError):
    """Tariff config is structurally invalid."""


@dataclass(frozen=True)
class Block:
    up_to_kwh: Optional[float]
    usd_per_kwh: float


@dataclass(frozen=True)
class Season:
    name: str
    months: frozenset[int]
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Tariff:
    version: str
    cost_basis: str
    seasons: tuple[Season, ...]
    energy_multiplier: float
    fixed_monthly_usd: float

    def season_for(self, month: int) -> Season:
        for season in self.seasons:
            if month in season.months:
                return season
        raise TariffError(f"no season covers month {month}")

    def energy_cost_usd(self, kwh: float, *, cycle_kwh_before: float, month: int) -> float:
        remaining = max(0.0, float(kwh))
        position = max(0.0, float(cycle_kwh_before))
        base = 0.0
        for block in self.season_for(month).blocks:
            if remaining <= 0.0:
                break
            if block.up_to_kwh is not None and position >= block.up_to_kwh:
                continue
            room = remaining if block.up_to_kwh is None else min(remaining, block.up_to_kwh - position)
            base += room * block.usd_per_kwh
            remaining -= room
            position += room
        return base * self.energy_multiplier

    def marginal_usd_per_kwh(self, *, cycle_kwh: float, month: int) -> float:
        position = max(0.0, float(cycle_kwh))
        for block in self.season_for(month).blocks:
            if block.up_to_kwh is None or position < block.up_to_kwh:
                return block.usd_per_kwh * self.energy_multiplier
        raise TariffError("tariff has no open top block")


def _season(name: str, raw: dict) -> Season:
    blocks: list[Block] = []
    for item in raw.get("blocks") or []:
        up_to = item.get("up_to_kwh")
        blocks.append(Block(None if up_to is None else float(up_to), float(item["cents_per_kwh"]) / 100.0))
    if not blocks or blocks[-1].up_to_kwh is not None:
        raise TariffError(f"season {name!r} must end with an open (up_to_kwh: null) block")
    bounds = [b.up_to_kwh for b in blocks[:-1]]
    if any(b is None for b in bounds) or bounds != sorted(bounds):
        raise TariffError(f"season {name!r} blocks must ascend with only the last open")
    return Season(name=name, months=frozenset(int(m) for m in raw.get("months") or []), blocks=tuple(blocks))


def load_tariff(path: str | Path) -> Tariff:
    raw = yaml.safe_load(Path(path).read_text())
    seasons = tuple(_season(name, body) for name, body in (raw.get("seasons") or {}).items())
    covered = sorted(m for s in seasons for m in s.months)
    if covered != list(range(1, 13)):
        raise TariffError(f"seasons must cover each month 1-12 exactly once, got {covered}")
    adj = raw.get("energy_adjustments") or {}
    stage1 = sum(float(a["pct"]) for a in adj.get("on_base_pct") or [])
    stage2 = sum(float(a["pct"]) for a in adj.get("on_adjusted_pct") or [])
    return Tariff(
        version=str(raw["tariff_version"]),
        cost_basis=str(raw.get("cost_basis", "pre_tax")),
        seasons=seasons,
        energy_multiplier=(1.0 + stage1 / 100.0) * (1.0 + stage2 / 100.0),
        fixed_monthly_usd=sum(float(f["usd"]) for f in raw.get("fixed_monthly") or []),
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_tariff.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add orion/energy/tariff.py config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml orion/energy/tests/test_energy_tariff.py
git commit -m "feat(energy): versioned Utah Schedule 1 block tariff"
```

---

### Task 4: Usage ledger — late data, billing cycles, accrual

**Files:**
- Create: `orion/energy/ledger.py`
- Test: `orion/energy/tests/test_energy_ledger.py`

**Interfaces:**
- Consumes: `Tariff` (Task 3), `EnergyUsageIntervalV1`, `EnergyCostAccruedV1` (Task 1)
- Produces: `UsageLedger(tariff: Tariff, *, tz: ZoneInfo, cycle_start_day: int)` with:
  - properties `tariff -> Tariff`, `tz -> ZoneInfo`
  - `upsert(interval: EnergyUsageIntervalV1) -> bool`
  - `usage_points() -> set[str]`
  - `cycle_bounds(ts: datetime) -> tuple[datetime, datetime]` (local-midnight aware datetimes, end exclusive)
  - `all_cycles() -> set[tuple[str, datetime]]`
  - `accrue_cycle(usage_point_id: str, cycle_start: datetime, *, computed_at: datetime) -> list[EnergyCostAccruedV1]`
  - `cycle_kwh_before(usage_point_id: str, ts: datetime) -> Optional[tuple[float, datetime]]`
  - `intervals_overlapping(usage_point_id: str, start: datetime, end: datetime) -> list[EnergyUsageIntervalV1]`
  - `interval_cost_usd(usage_point_id: str, interval_start: datetime) -> Optional[float]`

- [ ] **Step 1: Write failing tests**

Create `orion/energy/tests/test_energy_ledger.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import EnergyUsageIntervalV1

ROOT = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
DENVER = ZoneInfo("America/Denver")
M = 1.071 * 1.0518
B1S, B2S = 0.098332 * M, 0.125263 * M
NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _iv(start: datetime, kwh: float, *, retrieved: datetime = NOW, point: str = "UP123") -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop",
        usage_point_id=point,
        interval_start=start,
        interval_end=start + timedelta(hours=1),
        energy_kwh=kwh,
        retrieved_at=retrieved,
    )


def _ledger(day: int = 1) -> UsageLedger:
    return UsageLedger(TARIFF, tz=DENVER, cycle_start_day=day)


def test_newer_retrieval_replaces_older_is_ignored() -> None:
    led = _ledger()
    t = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    assert led.upsert(_iv(t, 1.0, retrieved=NOW)) is True
    assert led.upsert(_iv(t, 2.0, retrieved=NOW + timedelta(days=1))) is True
    assert led.upsert(_iv(t, 9.0, retrieved=NOW - timedelta(days=1))) is False
    assert led.intervals_overlapping("UP123", t, t + timedelta(hours=1))[0].energy_kwh == 2.0


def test_cycle_bounds_mid_month_start_day() -> None:
    led = _ledger(day=15)
    start, end = led.cycle_bounds(datetime(2026, 9, 10, 12, tzinfo=DENVER))
    assert (start, end) == (datetime(2026, 8, 15, tzinfo=DENVER), datetime(2026, 9, 15, tzinfo=DENVER))
    start, _ = led.cycle_bounds(datetime(2026, 9, 15, 0, 30, tzinfo=DENVER))
    assert start == datetime(2026, 9, 15, tzinfo=DENVER)


def test_cycle_start_day_clamps_to_short_month() -> None:
    led = _ledger(day=31)
    start, end = led.cycle_bounds(datetime(2026, 3, 5, tzinfo=DENVER))
    assert start == datetime(2026, 2, 28, tzinfo=DENVER)
    assert end == datetime(2026, 3, 31, tzinfo=DENVER)


def test_accrual_crosses_block_boundary_in_order() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    for i in range(3):
        led.upsert(_iv(base + timedelta(hours=i), 200.0))
    start, _ = led.cycle_bounds(base)
    rows = led.accrue_cycle("UP123", start, computed_at=NOW)
    assert [r.interval_cost_usd for r in rows] == pytest.approx([200 * B1S, 200 * B1S, 200 * B2S])
    assert rows[-1].cycle_accumulated_kwh == pytest.approx(600.0)
    assert rows[-1].cycle_to_date_total_usd == pytest.approx(400 * B1S + 200 * B2S + 12.16)
    assert rows[0].marginal_usd_per_kwh == pytest.approx(B1S)
    assert rows[1].marginal_usd_per_kwh == pytest.approx(B2S)  # 400 used: next kWh is block 2
    assert rows[0].cycle_start == date(2026, 7, 1)
    assert led.interval_cost_usd("UP123", base) == pytest.approx(200 * B1S)


def test_late_interval_reprices_later_ones() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    led.upsert(_iv(base + timedelta(hours=1), 200.0))
    start, _ = led.cycle_bounds(base)
    led.accrue_cycle("UP123", start, computed_at=NOW)
    led.upsert(_iv(base, 300.0))  # arrives late, earlier in the cycle
    rows = led.accrue_cycle("UP123", start, computed_at=NOW)
    assert rows[1].interval_cost_usd == pytest.approx(100 * B1S + 100 * B2S)


def test_cycle_kwh_before_counts_only_finished_intervals_in_cycle() -> None:
    led = _ledger()
    t = datetime(2026, 9, 10, 16, tzinfo=timezone.utc)
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(datetime(2026, 8, 31, 5, tzinfo=timezone.utc), 50.0))  # previous cycle (Denver Aug 30)
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(t - timedelta(hours=2), 100.0))
    led.upsert(_iv(t, 7.0))  # starts at t: not finished before t
    kwh, as_of = led.cycle_kwh_before("UP123", t)
    assert kwh == pytest.approx(100.0)
    assert as_of == t - timedelta(hours=1)


def test_all_cycles_and_usage_points() -> None:
    led = _ledger()
    led.upsert(_iv(datetime(2026, 7, 2, 18, tzinfo=timezone.utc), 1.0))
    led.upsert(_iv(datetime(2026, 8, 2, 18, tzinfo=timezone.utc), 1.0, point="UP9"))
    assert led.usage_points() == {"UP123", "UP9"}
    assert led.all_cycles() == {
        ("UP123", datetime(2026, 7, 1, tzinfo=DENVER)),
        ("UP9", datetime(2026, 8, 1, tzinfo=DENVER)),
    }
```

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_ledger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.energy.ledger'`

- [ ] **Step 2: Implement**

Create `orion/energy/ledger.py`:

```python
"""In-memory whole-house usage ledger.

Rebuilt from the processed drop directory at boot, so it needs no database of its
own. Accrual always recomputes a whole billing cycle in time order: a late interval
earlier in the cycle moves every later interval's block position.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from orion.energy.tariff import Tariff
from orion.schemas.energy import EnergyCostAccruedV1, EnergyUsageIntervalV1


class UsageLedger:
    def __init__(self, tariff: Tariff, *, tz: ZoneInfo, cycle_start_day: int) -> None:
        if not 1 <= int(cycle_start_day) <= 31:
            raise ValueError("cycle_start_day must be 1-31")
        self._tariff = tariff
        self._tz = tz
        self._cycle_start_day = int(cycle_start_day)
        self._intervals: dict[str, dict[datetime, EnergyUsageIntervalV1]] = {}
        self._accrued: dict[tuple[str, datetime], EnergyCostAccruedV1] = {}

    @property
    def tariff(self) -> Tariff:
        return self._tariff

    @property
    def tz(self) -> ZoneInfo:
        return self._tz

    def upsert(self, interval: EnergyUsageIntervalV1) -> bool:
        key = interval.interval_start.astimezone(timezone.utc)
        bucket = self._intervals.setdefault(interval.usage_point_id, {})
        existing = bucket.get(key)
        if existing is not None and existing.retrieved_at > interval.retrieved_at:
            return False
        bucket[key] = interval
        return True

    def usage_points(self) -> set[str]:
        return set(self._intervals)

    def _start_on(self, year: int, month: int) -> datetime:
        day = min(self._cycle_start_day, calendar.monthrange(year, month)[1])
        return datetime(year, month, day, tzinfo=self._tz)

    def cycle_bounds(self, ts: datetime) -> tuple[datetime, datetime]:
        local = ts.astimezone(self._tz)
        this = self._start_on(local.year, local.month)
        if local >= this:
            ny, nm = (local.year + 1, 1) if local.month == 12 else (local.year, local.month + 1)
            return this, self._start_on(ny, nm)
        py, pm = (local.year - 1, 12) if local.month == 1 else (local.year, local.month - 1)
        return self._start_on(py, pm), this

    def all_cycles(self) -> set[tuple[str, datetime]]:
        return {
            (point, self.cycle_bounds(start)[0])
            for point, bucket in self._intervals.items()
            for start in bucket
        }

    def _in_cycle(self, usage_point_id: str, start: datetime, end: datetime) -> list[EnergyUsageIntervalV1]:
        bucket = self._intervals.get(usage_point_id, {})
        return sorted(
            (iv for iv in bucket.values() if start <= iv.interval_start < end),
            key=lambda iv: iv.interval_start,
        )

    def accrue_cycle(
        self, usage_point_id: str, cycle_start: datetime, *, computed_at: datetime
    ) -> list[EnergyCostAccruedV1]:
        start, end = self.cycle_bounds(cycle_start)
        cycle_kwh = 0.0
        cycle_cost = 0.0
        out: list[EnergyCostAccruedV1] = []
        for iv in self._in_cycle(usage_point_id, start, end):
            month = iv.interval_start.astimezone(self._tz).month
            cost = self._tariff.energy_cost_usd(iv.energy_kwh, cycle_kwh_before=cycle_kwh, month=month)
            cycle_kwh += iv.energy_kwh
            cycle_cost += cost
            accrued = EnergyCostAccruedV1(
                usage_point_id=usage_point_id,
                interval_start=iv.interval_start,
                interval_end=iv.interval_end,
                energy_kwh=iv.energy_kwh,
                interval_cost_usd=cost,
                marginal_usd_per_kwh=self._tariff.marginal_usd_per_kwh(cycle_kwh=cycle_kwh, month=month),
                cycle_start=start.date(),
                cycle_accumulated_kwh=cycle_kwh,
                cycle_energy_cost_usd=cycle_cost,
                cycle_to_date_total_usd=cycle_cost + self._tariff.fixed_monthly_usd,
                tariff_version=self._tariff.version,
                cost_basis=self._tariff.cost_basis,
                computed_at=computed_at,
            )
            self._accrued[(usage_point_id, iv.interval_start.astimezone(timezone.utc))] = accrued
            out.append(accrued)
        return out

    def cycle_kwh_before(self, usage_point_id: str, ts: datetime) -> Optional[tuple[float, datetime]]:
        start, _ = self.cycle_bounds(ts)
        done = [iv for iv in self._in_cycle(usage_point_id, start, ts) if iv.interval_end <= ts]
        if not done:
            return None
        return sum(iv.energy_kwh for iv in done), max(iv.interval_end for iv in done)

    def intervals_overlapping(
        self, usage_point_id: str, start: datetime, end: datetime
    ) -> list[EnergyUsageIntervalV1]:
        bucket = self._intervals.get(usage_point_id, {})
        return sorted(
            (iv for iv in bucket.values() if iv.interval_start < end and iv.interval_end > start),
            key=lambda iv: iv.interval_start,
        )

    def interval_cost_usd(self, usage_point_id: str, interval_start: datetime) -> Optional[float]:
        accrued = self._accrued.get((usage_point_id, interval_start.astimezone(timezone.utc)))
        return None if accrued is None else accrued.interval_cost_usd
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_ledger.py -q`
Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add orion/energy/ledger.py orion/energy/tests/test_energy_ledger.py
git commit -m "feat(energy): usage ledger with cycle accrual and late-data repricing"
```

---

### Task 5: Run-cost join on settled power intents

**Files:**
- Create: `orion/energy/run_cost.py`
- Test: `orion/energy/tests/test_energy_run_cost.py`

**Interfaces:**
- Consumes: `UsageLedger` (Task 4), `PowerIntentSettledV1` (`orion/schemas/power.py`), `EnergyRunCostEstimatedV1` (Task 1)
- Produces: `estimate_run_cost(settled: PowerIntentSettledV1, *, ledger: UsageLedger, usage_point_id: Optional[str], computed_at: datetime) -> EnergyRunCostEstimatedV1`; constant `JOULES_PER_KWH = 3_600_000.0`

- [ ] **Step 1: Write failing tests**

Create `orion/energy/tests/test_energy_run_cost.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.run_cost import estimate_run_cost
from orion.energy.tariff import load_tariff
from orion.schemas.energy import EnergyUsageIntervalV1
from orion.schemas.power import PowerIntentSettledV1

ROOT = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
M = 1.071 * 1.0518
B1S = 0.098332 * M
T = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)


def _settled(**kw) -> PowerIntentSettledV1:
    base = dict(
        intent_id="i-1",
        workload_kind="reverie.diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=T,
        window_end=T + timedelta(hours=1),
        sample_count=3600,
        actual_mean_watts=250.0,
        actual_peak_watts=300.0,
        energy_joules=250.0 * 3600,
        baseline_watts=50.0,
    )
    base.update(kw)
    return PowerIntentSettledV1(**base)


def _iv(start: datetime, kwh: float) -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop", usage_point_id="UP123", interval_start=start,
        interval_end=start + timedelta(hours=1), energy_kwh=kwh, retrieved_at=NOW,
    )


def _ledger(*intervals: EnergyUsageIntervalV1) -> UsageLedger:
    led = UsageLedger(TARIFF, tz=ZoneInfo("America/Denver"), cycle_start_day=1)
    for iv in intervals:
        led.upsert(iv)
    for point, start in led.all_cycles():
        led.accrue_cycle(point, start, computed_at=NOW)
    return led


def test_blind_settlement_is_unknown_not_free() -> None:
    blind = _settled(outcome="no_samples", sample_count=0, actual_mean_watts=None,
                     actual_peak_watts=None, energy_joules=None)
    est = estimate_run_cost(blind, ledger=_ledger(_iv(T - timedelta(hours=2), 100.0)),
                            usage_point_id="UP123", computed_at=NOW)
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "settlement_not_measured"
    assert est.house_share_cost_usd is None and est.house_share_gap == "settlement_not_measured"
    assert est.energy_kwh is None


def test_incremental_cost_at_cycle_position() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.energy_basis == "incremental_over_baseline"
    assert est.energy_kwh == pytest.approx(0.2)
    assert est.estimated_run_cost_usd == pytest.approx(0.2 * B1S)
    assert est.marginal_usd_per_kwh == pytest.approx(B1S)
    assert est.cycle_kwh_basis == pytest.approx(100.0)
    assert est.cycle_kwh_basis_as_of == T - timedelta(hours=1)
    assert est.house_share_gap == "house_interval_missing"
    assert est.tariff_version == "rmp-ut-sch1-2026-08-10"


def test_gross_basis_without_baseline() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(_settled(baseline_watts=None), ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.energy_basis == "gross"
    assert est.energy_kwh == pytest.approx(0.25)


def test_no_cycle_usage_is_a_gap() -> None:
    est = estimate_run_cost(_settled(), ledger=_ledger(), usage_point_id="UP123", computed_at=NOW)
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "no_cycle_usage"


def test_unknown_usage_point_is_a_gap() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id=None, computed_at=NOW)
    assert est.run_cost_gap == "no_cycle_usage"
    assert est.house_share_gap == "house_interval_missing"


def test_house_share_when_interval_covers_window() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0), _iv(T, 1.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id="UP123", computed_at=NOW)
    interval_cost = 1.0 * B1S  # house position 100 kWh -> first block
    assert est.house_kwh_overlap == pytest.approx(1.0)
    assert est.house_share_cost_usd == pytest.approx(0.2 * interval_cost)
    assert est.house_share_gap is None
    assert est.estimated_run_cost_usd == pytest.approx(0.2 * B1S)


def test_house_share_needs_full_coverage() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0), _iv(T, 1.0))
    long_run = _settled(window_end=T + timedelta(hours=2), energy_joules=250.0 * 7200)
    est = estimate_run_cost(long_run, ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.house_share_cost_usd is None and est.house_share_gap == "house_interval_missing"
```

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_run_cost.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.energy.run_cost'`

- [ ] **Step 2: Implement**

Create `orion/energy/run_cost.py`:

```python
"""Price one settled power intent against the house tariff.

The meter-side estimate uses the INCREMENTAL draw (mean minus the card's baseline
just before the window) when a baseline exists: an idle card draws its baseline
whether or not the workload runs, so only the delta is the workload's cost.

The house share apportions each overlapping whole-house interval by the run's share
of that interval's kWh. It needs the utility data, which lags about a day, so it is
usually a gap at settlement time and filled in when the pipeline re-prices.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from orion.energy.ledger import UsageLedger
from orion.schemas.energy import EnergyRunCostEstimatedV1
from orion.schemas.power import PowerIntentSettledV1

JOULES_PER_KWH = 3_600_000.0
_COVERAGE_SLACK_SEC = 1.0


def _run_kwh(settled: PowerIntentSettledV1) -> tuple[Optional[float], Optional[str]]:
    if settled.outcome != "settled" or settled.energy_joules is None:
        return None, None
    elapsed = (settled.window_end - settled.window_start).total_seconds()
    if elapsed <= 0:
        return None, None
    if settled.baseline_watts is not None and settled.actual_mean_watts is not None:
        delta_w = max(0.0, settled.actual_mean_watts - settled.baseline_watts)
        return delta_w * elapsed / JOULES_PER_KWH, "incremental_over_baseline"
    return settled.energy_joules / JOULES_PER_KWH, "gross"


def _run_cost_fields(
    kwh: float, settled: PowerIntentSettledV1, ledger: UsageLedger, point: Optional[str]
) -> dict[str, Any]:
    before = ledger.cycle_kwh_before(point, settled.window_start) if point else None
    if before is None:
        return {"run_cost_gap": "no_cycle_usage"}
    cycle_kwh, as_of = before
    month = settled.window_start.astimezone(ledger.tz).month
    cost = ledger.tariff.energy_cost_usd(kwh, cycle_kwh_before=cycle_kwh, month=month)
    marginal = cost / kwh if kwh > 0 else ledger.tariff.marginal_usd_per_kwh(cycle_kwh=cycle_kwh, month=month)
    return {
        "estimated_run_cost_usd": cost,
        "marginal_usd_per_kwh": marginal,
        "cycle_kwh_basis": cycle_kwh,
        "cycle_kwh_basis_as_of": as_of,
    }


def _house_share_fields(
    kwh: float, settled: PowerIntentSettledV1, ledger: UsageLedger, point: Optional[str]
) -> dict[str, Any]:
    gap = {"house_share_gap": "house_interval_missing"}
    if not point:
        return gap
    start, end = settled.window_start, settled.window_end
    window_sec = (end - start).total_seconds()
    covered = share_cost = house_kwh = 0.0
    for iv in ledger.intervals_overlapping(point, start, end):
        interval_cost = ledger.interval_cost_usd(point, iv.interval_start)
        if interval_cost is None:
            return gap
        overlap = (min(end, iv.interval_end) - max(start, iv.interval_start)).total_seconds()
        interval_sec = float(iv.interval_seconds)
        run_in = kwh * overlap / window_sec
        house_in = iv.energy_kwh * overlap / interval_sec
        share = 1.0 if house_in <= 0 else min(1.0, run_in / house_in)
        share_cost += share * interval_cost * overlap / interval_sec
        house_kwh += house_in
        covered += overlap
    if covered + _COVERAGE_SLACK_SEC < window_sec:
        return gap
    return {"house_share_cost_usd": share_cost, "house_kwh_overlap": house_kwh}


def estimate_run_cost(
    settled: PowerIntentSettledV1,
    *,
    ledger: UsageLedger,
    usage_point_id: Optional[str],
    computed_at: datetime,
) -> EnergyRunCostEstimatedV1:
    base: dict[str, Any] = dict(
        intent_id=settled.intent_id,
        workload_kind=settled.workload_kind,
        node=settled.node,
        gpu_index=settled.gpu_index,
        window_start=settled.window_start,
        window_end=settled.window_end,
        settlement_outcome=settled.outcome,
        tariff_version=ledger.tariff.version,
        cost_basis=ledger.tariff.cost_basis,
        computed_at=computed_at,
    )
    kwh, basis = _run_kwh(settled)
    if kwh is None:
        return EnergyRunCostEstimatedV1(
            **base, run_cost_gap="settlement_not_measured", house_share_gap="settlement_not_measured"
        )
    return EnergyRunCostEstimatedV1(
        **base,
        energy_kwh=kwh,
        energy_basis=basis,
        **_run_cost_fields(kwh, settled, ledger, usage_point_id),
        **_house_share_fields(kwh, settled, ledger, usage_point_id),
    )
```

- [ ] **Step 3: Run the whole package suite**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests -q`
Expected: 35 passed (7 schema + 6 espi + 8 tariff + 7 ledger + 7 run cost)

- [ ] **Step 4: Commit**

```bash
git add orion/energy/run_cost.py orion/energy/tests/test_energy_run_cost.py
git commit -m "feat(energy): price settled power intents with dual labeled costs"
```

---

### Task 6: sql-writer persistence (newest-wins upserts)

**Files:**
- Create: `services/orion-sql-writer/app/models/energy.py`
- Create: `services/orion-sql-writer/app/energy_persist.py`
- Modify: `services/orion-sql-writer/app/models/__init__.py` (import ~line 19, `__all__` ~line 117)
- Modify: `services/orion-sql-writer/app/worker.py` (imports ~line 28 and ~line 147; `MODEL_MAP` ~line 468; `_write_row` just before `if sql_model_cls is HarnessTurnTraceSQL:` ~line 1763)
- Modify: `services/orion-sql-writer/app/settings.py` (`DEFAULT_ROUTE_MAP` ~line 36; default channels ~line 162; force-append ~line 667)
- Modify: `services/orion-sql-writer/.env_example` (`SQL_WRITER_SUBSCRIBE_CHANNELS`)
- Modify: `.github/workflows/orion-sql-writer-tests.yml`
- Test: `services/orion-sql-writer/tests/test_energy_sql_shape.py`

**Interfaces:**
- Consumes: the three schemas (Task 1)
- Produces: tables `energy_usage_interval`, `energy_cost_accrued`, `energy_run_cost`; `ENERGY_UPSERTS: dict[type, Callable[[Session, dict], bool]]`

- [ ] **Step 1: Write failing tests**

Create `services/orion-sql-writer/tests/test_energy_sql_shape.py`:

```python
"""Energy kinds -> tables, with newest-wins upserts on natural keys."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects import postgresql

from app.energy_persist import ENERGY_UPSERTS
from app.models.energy import EnergyCostAccruedSQL, EnergyRunCostSQL, EnergyUsageIntervalSQL
from app.settings import Settings
from app.worker import MODEL_MAP

from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)

T0 = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 10, 19, tzinfo=timezone.utc)

ROUTES = [
    ("energy.usage.observed.v1", "EnergyUsageIntervalSQL", EnergyUsageIntervalSQL, EnergyUsageIntervalV1, "orion:energy:usage:observed"),
    ("energy.cost.accrued.v1", "EnergyCostAccruedSQL", EnergyCostAccruedSQL, EnergyCostAccruedV1, "orion:energy:cost:accrued"),
    ("energy.run_cost.estimated.v1", "EnergyRunCostSQL", EnergyRunCostSQL, EnergyRunCostEstimatedV1, "orion:energy:run_cost:estimated"),
]


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_every_schema_field_has_a_column(kind, name, model, schema, channel) -> None:
    cols = {c.key for c in sa_inspect(model).columns}
    missing = [f for f in schema.model_fields if f not in cols]
    assert not missing, f"{model.__tablename__} missing {missing}"


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_kind_routes_to_table(kind, name, model, schema, channel) -> None:
    model_cls, schema_cls = MODEL_MAP[name]
    assert model_cls is model and schema_cls is schema
    assert Settings().route_map.get(kind) == name


@pytest.mark.parametrize("kind,name,model,schema,channel", ROUTES)
def test_channel_subscribed_even_with_stale_env(kind, name, model, schema, channel) -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert channel in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert channel in stale.effective_subscribe_channels


class _CapturingSession:
    def __init__(self) -> None:
        self.statements = []
        self.committed = False

    def execute(self, stmt):
        self.statements.append(stmt)

    def commit(self) -> None:
        self.committed = True


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_usage_upsert_newer_retrieval_wins() -> None:
    sess = _CapturingSession()
    row = EnergyUsageIntervalV1(
        source="file_drop", usage_point_id="UP123", interval_start=T0, interval_end=T1,
        energy_kwh=1.2, retrieved_at=T1,
    ).model_dump()
    assert ENERGY_UPSERTS[EnergyUsageIntervalSQL](sess, row) is True
    sql = _sql(sess.statements[0])
    assert "ON CONFLICT ON CONSTRAINT uq_energy_usage_interval_point_start DO UPDATE" in sql
    assert "energy_usage_interval.retrieved_at <= excluded.retrieved_at" in sql
    assert sess.committed


def test_accrued_upsert_newest_computation_wins_and_coerces_date() -> None:
    sess = _CapturingSession()
    row = EnergyCostAccruedV1(
        usage_point_id="UP123", interval_start=T0, interval_end=T1, energy_kwh=1.0,
        interval_cost_usd=0.11, marginal_usd_per_kwh=0.11, cycle_start=date(2026, 9, 1),
        cycle_accumulated_kwh=101.0, cycle_energy_cost_usd=11.2, cycle_to_date_total_usd=23.36,
        tariff_version="rmp-ut-sch1-2026-08-10", computed_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyCostAccruedSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_cost_accrued_point_start_tariff" in sql
    assert "energy_cost_accrued.computed_at <= excluded.computed_at" in sql
    assert stmt.compile(dialect=postgresql.dialect()).params["cycle_start"] == date(2026, 9, 1)


def test_run_cost_upsert_keyed_on_intent() -> None:
    sess = _CapturingSession()
    row = EnergyRunCostEstimatedV1(
        intent_id="i-1", workload_kind="reverie.diffusion", node="circe", window_start=T0,
        window_end=T1, settlement_outcome="settled", energy_kwh=0.2,
        energy_basis="incremental_over_baseline", estimated_run_cost_usd=0.022,
        house_share_gap="house_interval_missing", computed_at=T1,
    ).model_dump()
    ENERGY_UPSERTS[EnergyRunCostSQL](sess, row)
    sql = _sql(sess.statements[0])
    assert "ON CONFLICT (intent_id) DO UPDATE" in sql
    assert "energy_run_cost.computed_at <= excluded.computed_at" in sql
```

Run: `cd services/orion-sql-writer && $PY -m pytest tests/test_energy_sql_shape.py -q; cd ../..`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.energy_persist'`

- [ ] **Step 2: Add the models**

Create `services/orion-sql-writer/app/models/energy.py`:

```python
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)

from app.db import Base


class EnergyUsageIntervalSQL(Base):
    """Whole-house metered kWh per interval (``orion:energy:usage:observed``).

    One row per (usage point, interval start). Late utility corrections upsert in
    place; the newer ``retrieved_at`` wins (see ``app/energy_persist.py``).
    """

    __tablename__ = "energy_usage_interval"
    __table_args__ = (
        UniqueConstraint("usage_point_id", "interval_start", name="uq_energy_usage_interval_point_start"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=False)
    interval_start = Column(DateTime(timezone=True), nullable=False, index=True)
    interval_end = Column(DateTime(timezone=True), nullable=False)
    energy_kwh = Column(Float, nullable=False)
    quality = Column(String, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyCostAccruedSQL(Base):
    """Tariff-priced usage interval (``orion:energy:cost:accrued``), one row per tariff version."""

    __tablename__ = "energy_cost_accrued"
    __table_args__ = (
        UniqueConstraint(
            "usage_point_id", "interval_start", "tariff_version",
            name="uq_energy_cost_accrued_point_start_tariff",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    usage_point_id = Column(String, nullable=False)
    interval_start = Column(DateTime(timezone=True), nullable=False, index=True)
    interval_end = Column(DateTime(timezone=True), nullable=False)
    energy_kwh = Column(Float, nullable=False)
    interval_cost_usd = Column(Float, nullable=False)
    marginal_usd_per_kwh = Column(Float, nullable=False)
    cycle_start = Column(Date, nullable=False, index=True)
    cycle_accumulated_kwh = Column(Float, nullable=False)
    cycle_energy_cost_usd = Column(Float, nullable=False)
    cycle_to_date_total_usd = Column(Float, nullable=False)
    tariff_version = Column(String, nullable=False)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class EnergyRunCostSQL(Base):
    """Dollar cost of a settled power intent (``orion:energy:run_cost:estimated``).

    NULL cost columns mean UNKNOWN and always come with a ``*_gap`` reason; they
    must never be read as free.
    """

    __tablename__ = "energy_run_cost"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    intent_id = Column(String, nullable=False, unique=True)
    workload_kind = Column(String, nullable=False)
    node = Column(String, nullable=False)
    gpu_index = Column(Integer, nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_end = Column(DateTime(timezone=True), nullable=False)
    settlement_outcome = Column(String, nullable=False)
    energy_kwh = Column(Float, nullable=True)
    energy_basis = Column(String, nullable=True)
    estimated_run_cost_usd = Column(Float, nullable=True)
    marginal_usd_per_kwh = Column(Float, nullable=True)
    cycle_kwh_basis = Column(Float, nullable=True)
    cycle_kwh_basis_as_of = Column(DateTime(timezone=True), nullable=True)
    run_cost_gap = Column(String, nullable=True)
    house_share_cost_usd = Column(Float, nullable=True)
    house_kwh_overlap = Column(Float, nullable=True)
    house_share_gap = Column(String, nullable=True)
    tariff_version = Column(String, nullable=True)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)
```

In `services/orion-sql-writer/app/models/__init__.py`, next to the `HomeCoolingSampleSQL` import add:

```python
from .energy import EnergyCostAccruedSQL, EnergyRunCostSQL, EnergyUsageIntervalSQL
```

and to `__all__` next to `"HomeCoolingSampleSQL",` add:

```python
    "EnergyUsageIntervalSQL",
    "EnergyCostAccruedSQL",
    "EnergyRunCostSQL",
```

- [ ] **Step 3: Add the upserts**

Create `services/orion-sql-writer/app/energy_persist.py`:

```python
"""Newest-wins upserts for the energy tables.

A late utility correction, a re-priced cycle, or a run whose house share arrived a
day later all re-deliver the same natural key. The WHERE clause keeps a stale
redelivery (bus replay, out-of-order handler) from overwriting fresher data.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.energy import EnergyCostAccruedSQL, EnergyRunCostSQL, EnergyUsageIntervalSQL


def _columns(model: type, data: dict[str, Any]) -> dict[str, Any]:
    keys = {attr.key for attr in inspect(model).attrs} - {"id"}
    out = {k: v for k, v in data.items() if k in keys}
    for key, value in list(out.items()):
        if isinstance(value, str) and key.endswith(("_at", "_start", "_end", "_as_of")) and key != "cycle_start":
            out[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(out.get("cycle_start"), str):
        out["cycle_start"] = date.fromisoformat(out["cycle_start"])
    return out


def _upsert(
    sess: Session, model: type, data: dict[str, Any], *, conflict: dict[str, Any], newer_col: str
) -> bool:
    values = _columns(model, data)
    stmt = insert(model).values(**values)
    table = model.__table__
    immutable = set(conflict.get("index_elements") or []) | {"id"}
    stmt = stmt.on_conflict_do_update(
        **conflict,
        set_={k: stmt.excluded[k] for k in values if k not in immutable},
        where=table.c[newer_col] <= stmt.excluded[newer_col],
    )
    sess.execute(stmt)
    sess.commit()
    return True


def upsert_energy_usage_interval(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyUsageIntervalSQL, data,
        conflict={"constraint": "uq_energy_usage_interval_point_start"}, newer_col="retrieved_at",
    )


def upsert_energy_cost_accrued(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyCostAccruedSQL, data,
        conflict={"constraint": "uq_energy_cost_accrued_point_start_tariff"}, newer_col="computed_at",
    )


def upsert_energy_run_cost(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyRunCostSQL, data,
        conflict={"index_elements": ["intent_id"]}, newer_col="computed_at",
    )


ENERGY_UPSERTS: dict[type, Callable[[Session, dict[str, Any]], bool]] = {
    EnergyUsageIntervalSQL: upsert_energy_usage_interval,
    EnergyCostAccruedSQL: upsert_energy_cost_accrued,
    EnergyRunCostSQL: upsert_energy_run_cost,
}
```

- [ ] **Step 4: Wire worker and settings**

In `services/orion-sql-writer/app/worker.py`:

1. In the `from app.models import (...)` block next to `HomeCoolingSampleSQL,` add `EnergyCostAccruedSQL, EnergyRunCostSQL, EnergyUsageIntervalSQL,`.
2. Next to `from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1` add:
   ```python
   from orion.schemas.energy import EnergyCostAccruedV1, EnergyRunCostEstimatedV1, EnergyUsageIntervalV1
   from app.energy_persist import ENERGY_UPSERTS
   ```
3. In `MODEL_MAP` next to `"HomeCoolingSampleSQL": (...)` add:
   ```python
       "EnergyUsageIntervalSQL": (EnergyUsageIntervalSQL, EnergyUsageIntervalV1),
       "EnergyCostAccruedSQL": (EnergyCostAccruedSQL, EnergyCostAccruedV1),
       "EnergyRunCostSQL": (EnergyRunCostSQL, EnergyRunCostEstimatedV1),
   ```
4. In `_write_row`, immediately before `if sql_model_cls is HarnessTurnTraceSQL:` add (the enclosing `try/finally` already closes `sess`):
   ```python
           if sql_model_cls in ENERGY_UPSERTS:
               return ENERGY_UPSERTS[sql_model_cls](sess, filtered_data)
   ```

In `services/orion-sql-writer/app/settings.py`:

1. In `DEFAULT_ROUTE_MAP` after `"home.cooling.sample.v1": "HomeCoolingSampleSQL",` add:
   ```python
       "energy.usage.observed.v1": "EnergyUsageIntervalSQL",
       "energy.cost.accrued.v1": "EnergyCostAccruedSQL",
       "energy.run_cost.estimated.v1": "EnergyRunCostSQL",
   ```
2. In the default subscribe list after `"orion:home:cooling:sample",` add:
   ```python
               "orion:energy:usage:observed",
               "orion:energy:cost:accrued",
               "orion:energy:run_cost:estimated",
   ```
3. After the `orion:home:cooling:sample` force-append guard add:
   ```python
           # Same guarantee as cooling: SQL_WRITER_SUBSCRIBE_CHANNELS replaces rather
           # than merges, so a pre-energy operator .env would leave these routes inert.
           for energy_channel in (
               "orion:energy:usage:observed",
               "orion:energy:cost:accrued",
               "orion:energy:run_cost:estimated",
           ):
               if energy_channel not in channels:
                   channels.append(energy_channel)
   ```

Append the channels to the one-line JSON in `services/orion-sql-writer/.env_example` deterministically:

```bash
$PY - <<'EOF'
import json, pathlib
p = pathlib.Path("services/orion-sql-writer/.env_example")
lines = p.read_text().splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
        chans = json.loads(line.split("=", 1)[1])
        for c in ("orion:energy:usage:observed", "orion:energy:cost:accrued", "orion:energy:run_cost:estimated"):
            if c not in chans:
                chans.append(c)
        lines[i] = "SQL_WRITER_SUBSCRIBE_CHANNELS=" + json.dumps(chans, separators=(",", ":")) + "\n"
p.write_text("".join(lines))
EOF
git diff --stat services/orion-sql-writer/.env_example
```

Expected: 1 line changed.

In `.github/workflows/orion-sql-writer-tests.yml`, add `tests/test_energy_bus_catalog.py \` to the "Run transport reducer unit tests" list and `services/orion-sql-writer/tests/test_energy_sql_shape.py \` to the "Run sql-writer unit tests" list; add `"orion/schemas/energy.py"` and `"orion/bus/channels.yaml"` to both `paths:` filters if not already present.

- [ ] **Step 5: Run tests and sync env**

```bash
cd services/orion-sql-writer && $PY -m pytest tests/test_energy_sql_shape.py tests/test_route_map_completeness.py tests/test_home_cooling_sample_sql_shape.py -q; cd ../..
python3 scripts/sync_local_env_from_example.py
```

Expected: all pass. The sync script may report `SQL_WRITER_SUBSCRIBE_CHANNELS` as a differing (skipped) key on this host — record that in the PR; the force-append guard makes the live writer subscribe regardless.

- [ ] **Step 6: Commit**

```bash
git add services/orion-sql-writer/app/models/energy.py services/orion-sql-writer/app/energy_persist.py \
  services/orion-sql-writer/app/models/__init__.py services/orion-sql-writer/app/worker.py \
  services/orion-sql-writer/app/settings.py services/orion-sql-writer/.env_example \
  services/orion-sql-writer/tests/test_energy_sql_shape.py .github/workflows/orion-sql-writer-tests.yml
git diff --cached --check
git commit -m "feat(sql-writer): persist energy usage, accrual, and run cost"
```

---

### Task 7: `orion-energy` service — inbox, pipeline, bus wiring

**Files:**
- Create: `services/orion-energy/app/__init__.py` (empty)
- Create: `services/orion-energy/app/settings.py`
- Create: `services/orion-energy/app/inbox.py`
- Create: `services/orion-energy/app/pipeline.py`
- Create: `services/orion-energy/app/main.py`
- Create: `services/orion-energy/{requirements.txt,Dockerfile,docker-compose.yml,.env_example,README.md}`
- Create: `.github/workflows/orion-energy-tests.yml`
- Test: `services/orion-energy/tests/test_energy_inbox.py`, `test_energy_pipeline.py`, `test_energy_heartbeat.py`

**Interfaces:**
- Consumes: `parse_espi`, `EspiError` (Task 2); `load_tariff` (Task 3); `UsageLedger` (Task 4); `estimate_run_cost` (Task 5); kind constants (Task 1)
- Produces:
  - `scan_inbox(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[EnergyUsageIntervalV1]`
  - `load_processed(processed_dir: Path) -> list[EnergyUsageIntervalV1]`
  - `Outbound(channel: str, kind: str, payload: BaseModel)`; `EnergyChannels(usage: str, accrued: str, run_cost: str)`
  - `EnergyPipeline(*, ledger, channels, usage_point_id=None, pending_hours=96.0)` with `replay(intervals) -> None`, `ingest_intervals(intervals, *, now) -> list[Outbound]`, `on_settlement(settled, *, now) -> list[Outbound]`, `usage_point() -> Optional[str]`, `pending_count() -> int`

- [ ] **Step 1: Write failing inbox + pipeline tests**

```bash
mkdir -p services/orion-energy/app services/orion-energy/tests services/orion-energy/evals
touch services/orion-energy/app/__init__.py
```

Create `services/orion-energy/tests/test_energy_inbox.py`:

```python
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.inbox import load_processed, scan_inbox

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "orion/energy/tests/fixtures/espi_two_flows.xml"
NOW = datetime(2026, 9, 11, 12, 0, 5, tzinfo=timezone.utc)


def test_scan_parses_and_moves_to_processed(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    shutil.copy(FIXTURE, inbox / "sept.xml")
    rows = scan_inbox(inbox, processed, now=NOW)
    assert len(rows) == 3
    assert all(r.retrieved_at == NOW.replace(microsecond=0) for r in rows)
    assert not (inbox / "sept.xml").exists()
    assert [p.name for p in processed.iterdir()] == ["20260911T120005Z__sept.xml"]
    assert rows[0].source_file == "20260911T120005Z__sept.xml"


def test_unparseable_file_goes_to_failed_not_processed(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "junk.xml").write_bytes(b"<nope")
    assert scan_inbox(inbox, processed, now=NOW) == []
    assert (inbox / "failed" / "junk.xml").exists()
    assert not processed.exists() or not any(processed.iterdir())


def test_non_xml_files_ignored(tmp_path: Path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "notes.txt").write_text("hi")
    assert scan_inbox(inbox, processed, now=NOW) == []
    assert (inbox / "notes.txt").exists()


def test_replay_restores_retrieved_at_from_filename(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    shutil.copy(FIXTURE, processed / "20260911T120005Z__sept.xml")
    rows = load_processed(processed)
    assert len(rows) == 3
    assert rows[0].retrieved_at == datetime(2026, 9, 11, 12, 0, 5, tzinfo=timezone.utc)
```

Create `services/orion-energy/tests/test_energy_pipeline.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_USAGE_KIND,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1

REPO = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
CH = EnergyChannels(usage="u", accrued="a", run_cost="r")
T = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 10, 19, 5, tzinfo=timezone.utc)


def _pipeline(**kw) -> EnergyPipeline:
    led = UsageLedger(TARIFF, tz=ZoneInfo("America/Denver"), cycle_start_day=1)
    return EnergyPipeline(ledger=led, channels=CH, **kw)


def _iv(start: datetime, kwh: float, retrieved: datetime = NOW, point: str = "UP123") -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop", usage_point_id=point, interval_start=start,
        interval_end=start + timedelta(hours=1), energy_kwh=kwh, retrieved_at=retrieved,
    )


def _settled() -> PowerIntentSettledV1:
    return PowerIntentSettledV1(
        intent_id="i-1", workload_kind="reverie.diffusion", node="circe", gpu_index=2,
        outcome="settled", window_start=T, window_end=T + timedelta(hours=1), sample_count=3600,
        actual_mean_watts=250.0, actual_peak_watts=300.0, energy_joules=250.0 * 3600, baseline_watts=50.0,
    )


def test_ingest_publishes_usage_then_accrual() -> None:
    p = _pipeline()
    out = p.ingest_intervals([_iv(T - timedelta(hours=2), 100.0)], now=NOW)
    assert [(o.channel, o.kind) for o in out] == [("u", ENERGY_USAGE_KIND), ("a", ENERGY_ACCRUED_KIND)]


def test_stale_redelivery_publishes_nothing() -> None:
    p = _pipeline()
    p.ingest_intervals([_iv(T, 1.0)], now=NOW)
    assert p.ingest_intervals([_iv(T, 5.0, retrieved=NOW - timedelta(days=1))], now=NOW) == []


def test_settlement_before_house_data_is_repriced_when_it_arrives() -> None:
    p = _pipeline()
    p.ingest_intervals([_iv(T - timedelta(hours=2), 100.0)], now=NOW)
    first = p.on_settlement(_settled(), now=NOW)
    assert first[0].kind == ENERGY_RUN_COST_KIND
    assert first[0].payload.house_share_gap == "house_interval_missing"
    assert p.pending_count() == 1

    later = NOW + timedelta(days=1)
    out = p.ingest_intervals([_iv(T, 1.0, retrieved=later)], now=later)
    run_costs = [o.payload for o in out if o.kind == ENERGY_RUN_COST_KIND]
    assert len(run_costs) == 1
    assert run_costs[0].house_share_cost_usd == pytest.approx(0.2 * run_costs[0].marginal_usd_per_kwh)
    assert p.pending_count() == 0


def test_blind_settlement_is_not_kept_pending() -> None:
    p = _pipeline()
    blind = _settled().model_copy(update={"outcome": "no_samples", "energy_joules": None,
                                          "actual_mean_watts": None, "actual_peak_watts": None})
    p.on_settlement(blind, now=NOW)
    assert p.pending_count() == 0


def test_pending_expires_after_window() -> None:
    p = _pipeline(pending_hours=24)
    p.on_settlement(_settled(), now=NOW)
    assert p.pending_count() == 1
    much_later = NOW + timedelta(days=3)
    p.ingest_intervals([_iv(T + timedelta(days=2), 1.0, retrieved=much_later)], now=much_later)
    assert p.pending_count() == 0


def test_usage_point_resolution() -> None:
    p = _pipeline()
    assert p.usage_point() is None
    p.ingest_intervals([_iv(T, 1.0)], now=NOW)
    assert p.usage_point() == "UP123"
    p.ingest_intervals([_iv(T, 1.0, point="UP9")], now=NOW)
    assert p.usage_point() is None
    assert _pipeline(usage_point_id="UP9").usage_point() == "UP9"


def test_replay_warms_ledger_without_output() -> None:
    p = _pipeline()
    p.replay([_iv(T - timedelta(hours=2), 100.0)])
    est = p.on_settlement(_settled(), now=NOW)[0].payload
    assert est.estimated_run_cost_usd is not None
```

Create `services/orion-energy/tests/test_energy_heartbeat.py`:

```python
from __future__ import annotations

from app.main import build_heartbeat_chassis
from app.settings import Settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_heartbeat_chassis_builds_with_service_identity() -> None:
    chassis = build_heartbeat_chassis(Settings(ORION_BUS_ENABLED=False))
    assert isinstance(chassis, HeartbeatOnly)
```

Run: `cd services/orion-energy && PYTHONPATH=../..:. $PY -m pytest tests -q; cd ../..`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.inbox'`

- [ ] **Step 2: Settings and inbox**

Create `services/orion-energy/app/settings.py`:

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = Field(default="orion-energy")
    SERVICE_VERSION: str = Field(default="0.1.0")
    INSTANCE_ID: str = Field(default="athena")
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)

    ENERGY_INBOX_DIR: str = Field(default="/data/energy/inbox")
    ENERGY_PROCESSED_DIR: str = Field(default="/data/energy/processed")
    ENERGY_SCAN_INTERVAL_SEC: float = Field(default=60.0)
    ENERGY_TARIFF_PATH: str = Field(default="/app/config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
    ENERGY_TIMEZONE: str = Field(default="America/Denver")
    ENERGY_BILLING_CYCLE_START_DAY: int = Field(default=1, ge=1, le=31)
    # Empty = use the only usage point seen; required if the feed has several.
    ENERGY_USAGE_POINT_ID: str = Field(default="")
    ENERGY_RUN_COST_PENDING_HOURS: float = Field(default=96.0)
    # 0 disables pacing.
    ENERGY_PUBLISH_MAX_PER_SEC: float = Field(default=200.0, ge=0)

    ENERGY_USAGE_CHANNEL: str = Field(default="orion:energy:usage:observed")
    ENERGY_ACCRUED_CHANNEL: str = Field(default="orion:energy:cost:accrued")
    ENERGY_RUN_COST_CHANNEL: str = Field(default="orion:energy:run_cost:estimated")
    POWER_SETTLED_CHANNEL: str = Field(default="orion:power:intent:settled")

    HEARTBEAT_INTERVAL_SEC: float = Field(default=30.0)
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Create `services/orion-energy/app/inbox.py`:

```python
"""Drop directory for Green Button XML.

A parsed file moves to processed/ with its retrieval time as a filename prefix, so
a restart replays the exact same intervals with the exact same retrieved_at. A file
that fails to parse moves to inbox/failed/ and publishes nothing -- a broken export
must never read as a quiet day.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from orion.energy.espi import EspiError, parse_espi
from orion.schemas.energy import EnergyUsageIntervalV1

logger = logging.getLogger("orion-energy.inbox")
_STAMP = "%Y%m%dT%H%M%SZ"
_SEP = "__"


def scan_inbox(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[EnergyUsageIntervalV1]:
    if not inbox_dir.is_dir():
        return []
    stamp_time = now.astimezone(timezone.utc).replace(microsecond=0)
    rows: list[EnergyUsageIntervalV1] = []
    for path in sorted(p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml"):
        target_name = f"{stamp_time.strftime(_STAMP)}{_SEP}{path.name}"
        try:
            parsed = parse_espi(
                path.read_bytes(), retrieved_at=stamp_time, source="file_drop", source_file=target_name
            )
        except EspiError as exc:
            failed = inbox_dir / "failed"
            failed.mkdir(parents=True, exist_ok=True)
            path.rename(failed / path.name)
            logger.warning("energy_inbox_parse_failed file=%s error=%s", path.name, exc)
            continue
        processed_dir.mkdir(parents=True, exist_ok=True)
        path.rename(processed_dir / target_name)
        logger.info("energy_inbox_parsed file=%s intervals=%d", path.name, len(parsed))
        rows.extend(parsed)
    return rows


def load_processed(processed_dir: Path) -> list[EnergyUsageIntervalV1]:
    if not processed_dir.is_dir():
        return []
    rows: list[EnergyUsageIntervalV1] = []
    for path in sorted(processed_dir.glob("*.xml")):
        stamp, sep, _ = path.name.partition(_SEP)
        if not sep:
            logger.warning("energy_replay_skipped_unstamped file=%s", path.name)
            continue
        retrieved = datetime.strptime(stamp, _STAMP).replace(tzinfo=timezone.utc)
        try:
            rows.extend(parse_espi(path.read_bytes(), retrieved_at=retrieved, source="file_drop", source_file=path.name))
        except EspiError as exc:
            logger.warning("energy_replay_parse_failed file=%s error=%s", path.name, exc)
    return rows
```

- [ ] **Step 3: Pipeline**

Create `services/orion-energy/app/pipeline.py`:

```python
"""Pure orchestration: intervals and settlements in, bus messages out.

No I/O here, so every publish decision is testable. main.py only loops and publishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from pydantic import BaseModel

from orion.energy.ledger import UsageLedger
from orion.energy.run_cost import estimate_run_cost
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_USAGE_KIND,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1


@dataclass(frozen=True)
class Outbound:
    channel: str
    kind: str
    payload: BaseModel


@dataclass(frozen=True)
class EnergyChannels:
    usage: str
    accrued: str
    run_cost: str


class EnergyPipeline:
    def __init__(
        self,
        *,
        ledger: UsageLedger,
        channels: EnergyChannels,
        usage_point_id: Optional[str] = None,
        pending_hours: float = 96.0,
    ) -> None:
        self._ledger = ledger
        self._channels = channels
        self._configured_point = usage_point_id or None
        self._pending_hours = float(pending_hours)
        # Settlements whose cost is still partly unknown because utility data lags.
        self._pending: dict[str, PowerIntentSettledV1] = {}

    def usage_point(self) -> Optional[str]:
        if self._configured_point:
            return self._configured_point
        points = self._ledger.usage_points()
        return next(iter(points)) if len(points) == 1 else None

    def pending_count(self) -> int:
        return len(self._pending)

    def replay(self, intervals: Iterable[EnergyUsageIntervalV1]) -> None:
        for iv in intervals:
            self._ledger.upsert(iv)
        for point, cycle_start in sorted(self._ledger.all_cycles()):
            self._ledger.accrue_cycle(point, cycle_start, computed_at=datetime.now(timezone.utc))

    def ingest_intervals(self, intervals: Iterable[EnergyUsageIntervalV1], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        affected: set[tuple[str, datetime]] = set()
        for iv in intervals:
            if self._ledger.upsert(iv):
                out.append(Outbound(self._channels.usage, ENERGY_USAGE_KIND, iv))
                affected.add((iv.usage_point_id, self._ledger.cycle_bounds(iv.interval_start)[0]))
        for point, cycle_start in sorted(affected):
            for accrued in self._ledger.accrue_cycle(point, cycle_start, computed_at=now):
                out.append(Outbound(self._channels.accrued, ENERGY_ACCRUED_KIND, accrued))
        if affected:
            out.extend(self._reprice_pending(now=now))
        return out

    def on_settlement(self, settled: PowerIntentSettledV1, *, now: datetime) -> list[Outbound]:
        est = self._estimate(settled, now=now)
        self._track(settled, est, now=now)
        return [Outbound(self._channels.run_cost, ENERGY_RUN_COST_KIND, est)]

    def _estimate(self, settled: PowerIntentSettledV1, *, now: datetime) -> EnergyRunCostEstimatedV1:
        return estimate_run_cost(settled, ledger=self._ledger, usage_point_id=self.usage_point(), computed_at=now)

    def _track(self, settled: PowerIntentSettledV1, est: EnergyRunCostEstimatedV1, *, now: datetime) -> None:
        incomplete = est.run_cost_gap == "no_cycle_usage" or est.house_share_gap == "house_interval_missing"
        if incomplete and est.run_cost_gap != "settlement_not_measured":
            self._pending[settled.intent_id] = settled
        else:
            self._pending.pop(settled.intent_id, None)
        cutoff = now - timedelta(hours=self._pending_hours)
        for intent_id, pending in list(self._pending.items()):
            if pending.window_end < cutoff:
                del self._pending[intent_id]

    def _reprice_pending(self, *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for settled in list(self._pending.values()):
            est = self._estimate(settled, now=now)
            self._track(settled, est, now=now)
            if settled.intent_id in self._pending or est.house_share_cost_usd is not None:
                out.append(Outbound(self._channels.run_cost, ENERGY_RUN_COST_KIND, est))
        return out
```

> Behavior note for `_reprice_pending`: a pending run that has now expired past `pending_hours` is dropped without a final publish unless the reprice actually filled its house share. `test_pending_expires_after_window` pins the drop.

- [ ] **Step 4: Main loop**

Create `services/orion-energy/app/main.py`:

```python
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.codec import OrionCodec
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.power import PowerIntentSettledV1

from .inbox import load_processed, scan_inbox
from .pipeline import EnergyChannels, EnergyPipeline, Outbound
from .settings import Settings, get_settings

logger = logging.getLogger("orion-energy")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[ORION_ENERGY] %(asctime)s %(levelname)s - %(name)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_pipeline(settings: Settings) -> EnergyPipeline:
    ledger = UsageLedger(
        load_tariff(settings.ENERGY_TARIFF_PATH),
        tz=ZoneInfo(settings.ENERGY_TIMEZONE),
        cycle_start_day=settings.ENERGY_BILLING_CYCLE_START_DAY,
    )
    pipeline = EnergyPipeline(
        ledger=ledger,
        channels=EnergyChannels(
            usage=settings.ENERGY_USAGE_CHANNEL,
            accrued=settings.ENERGY_ACCRUED_CHANNEL,
            run_cost=settings.ENERGY_RUN_COST_CHANNEL,
        ),
        usage_point_id=settings.ENERGY_USAGE_POINT_ID or None,
        pending_hours=settings.ENERGY_RUN_COST_PENDING_HOURS,
    )
    replayed = load_processed(Path(settings.ENERGY_PROCESSED_DIR))
    pipeline.replay(replayed)
    logger.info("energy_ledger_replayed intervals=%d usage_point=%s", len(replayed), pipeline.usage_point())
    return pipeline


def _source(settings: Settings) -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.INSTANCE_ID)


async def publish_all(bus: OrionBusAsync, settings: Settings, outbound: Iterable[Outbound]) -> None:
    # A two-year backfill is ~35k messages; pub/sub drops what a slow sql-writer can't buffer.
    pause = 1.0 / settings.ENERGY_PUBLISH_MAX_PER_SEC if settings.ENERGY_PUBLISH_MAX_PER_SEC > 0 else 0.0
    for item in outbound:
        envelope = BaseEnvelope(
            kind=item.kind, source=_source(settings), payload=item.payload.model_dump(mode="json")
        )
        try:
            await bus.publish(item.channel, envelope)
        except Exception:
            logger.exception("energy_publish_failed channel=%s kind=%s", item.channel, item.kind)
        if pause:
            await asyncio.sleep(pause)


async def inbox_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    inbox, processed = Path(settings.ENERGY_INBOX_DIR), Path(settings.ENERGY_PROCESSED_DIR)
    while True:
        try:
            now = datetime.now(timezone.utc)
            rows = await asyncio.to_thread(scan_inbox, inbox, processed, now=now)
            if rows:
                async with lock:
                    outbound = pipeline.ingest_intervals(rows, now=now)
                await publish_all(bus, settings, outbound)
                logger.info("energy_ingested intervals=%d published=%d pending=%d",
                            len(rows), len(outbound), pipeline.pending_count())
        except Exception:
            logger.exception("energy_inbox_cycle_failed")
        await asyncio.sleep(settings.ENERGY_SCAN_INTERVAL_SEC)


async def settlement_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    codec = OrionCodec()
    async with bus.subscribe(settings.POWER_SETTLED_CHANNEL) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            try:
                decoded = codec.decode(msg.get("data"))
                if not decoded.ok:
                    logger.warning("energy_settlement_decode_failed error=%s", decoded.error)
                    continue
                settled = PowerIntentSettledV1.model_validate(decoded.envelope.payload)
                async with lock:
                    outbound = pipeline.on_settlement(settled, now=datetime.now(timezone.utc))
                await publish_all(bus, settings, outbound)
                est = outbound[0].payload
                logger.info(
                    "energy_run_cost intent_id=%s usd=%s gap=%s house_usd=%s house_gap=%s",
                    settled.intent_id, est.estimated_run_cost_usd, est.run_cost_gap,
                    est.house_share_cost_usd, est.house_share_gap,
                )
            except Exception:
                logger.exception("energy_settlement_handle_failed")


def build_heartbeat_chassis(settings: Optional[Settings] = None) -> HeartbeatOnly:
    s = settings if settings is not None else get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.SERVICE_NAME,
            service_version=s.SERVICE_VERSION,
            node_name=s.INSTANCE_ID,
            bus_url=s.ORION_BUS_URL,
            bus_enabled=s.ORION_BUS_ENABLED,
            heartbeat_interval_sec=s.HEARTBEAT_INTERVAL_SEC,
            health_channel=s.ORION_HEALTH_CHANNEL,
        )
    )


async def _main_async() -> None:
    settings = get_settings()
    heartbeat: Optional[HeartbeatOnly] = None
    try:
        heartbeat = build_heartbeat_chassis(settings)
        await heartbeat.start_background()
    except Exception:
        logger.exception("system_health_heartbeat_start_failed")
        heartbeat = None
    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED, codec=OrionCodec())
    try:
        if not bus.enabled:
            logger.warning("ORION_BUS_ENABLED=false; orion-energy has nothing to do")
            while True:
                await asyncio.sleep(3600)
        await bus.connect()
        pipeline = build_pipeline(settings)
        lock = asyncio.Lock()
        await asyncio.gather(
            inbox_loop(bus, settings, pipeline, lock),
            settlement_loop(bus, settings, pipeline, lock),
        )
    finally:
        if heartbeat is not None:
            try:
                await heartbeat.stop()
            except Exception:
                logger.exception("system_health_heartbeat_stop_error")


def main() -> None:
    setup_logging()
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        logger.info("orion-energy interrupted; exiting.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run service tests**

Run: `cd services/orion-energy && PYTHONPATH=../..:. $PY -m pytest tests -q; cd ../..`
Expected: 4 inbox + 7 pipeline + 1 heartbeat = 12 passed

- [ ] **Step 6: Packaging, env, docs, CI**

Create `services/orion-energy/requirements.txt`:

```text
pydantic==2.9.2
pydantic-settings==2.7.1
redis==5.0.8
orjson==3.10.7
pyyaml==6.0.3
tzdata==2024.2
```

Create `services/orion-energy/Dockerfile`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/orion-energy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-energy /app
COPY orion /app/orion
COPY config/energy /app/config/energy

CMD ["python", "-m", "app.main"]
```

Create `services/orion-energy/docker-compose.yml`:

```yaml
services:
  orion-energy:
    container_name: ${PROJECT}-orion-energy
    build:
      context: ../..
      dockerfile: services/orion-energy/Dockerfile
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ${ENERGY_HOST_DATA_DIR:-/mnt/storage-warm/orion-energy}:/data/energy
    networks:
      - app-net

networks:
  app-net:
    external: true
```

Create `services/orion-energy/.env_example`:

```text
SERVICE_NAME=orion-energy
SERVICE_VERSION=0.1.0
INSTANCE_ID=athena
ORION_BUS_URL=redis://100.92.216.81:6379/0
ORION_BUS_ENABLED=true
# Host directory mounted at /data/energy (absolute path; drop Green Button XML into <dir>/inbox).
ENERGY_HOST_DATA_DIR=/mnt/storage-warm/orion-energy
ENERGY_INBOX_DIR=/data/energy/inbox
ENERGY_PROCESSED_DIR=/data/energy/processed
ENERGY_SCAN_INTERVAL_SEC=60
ENERGY_TARIFF_PATH=/app/config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml
ENERGY_TIMEZONE=America/Denver
# Day of month your RMP billing cycle starts (meter read day). Blocks reset here.
ENERGY_BILLING_CYCLE_START_DAY=1
# Leave empty to use the only usage point in the feed.
ENERGY_USAGE_POINT_ID=
ENERGY_RUN_COST_PENDING_HOURS=96
# Bus publish pacing so a large backfill doesn't overrun sql-writer (0 = unpaced).
ENERGY_PUBLISH_MAX_PER_SEC=200
ENERGY_USAGE_CHANNEL=orion:energy:usage:observed
ENERGY_ACCRUED_CHANNEL=orion:energy:cost:accrued
ENERGY_RUN_COST_CHANNEL=orion:energy:run_cost:estimated
POWER_SETTLED_CHANNEL=orion:power:intent:settled
HEARTBEAT_INTERVAL_SEC=30
ORION_HEALTH_CHANNEL=orion:system:health
```

Create `services/orion-energy/README.md`:

````markdown
# orion-energy

Gives Orion the household electricity bill as a stake. Whole-house Green Button
usage from Rocky Mountain Power is priced with a versioned tariff, and every
settled GPU power intent gets a dollar estimate.

Spec: `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md`

## What it publishes

| Channel | Kind | Meaning |
|---|---|---|
| `orion:energy:usage:observed` | `energy.usage.observed.v1` | Metered house kWh per interval |
| `orion:energy:cost:accrued` | `energy.cost.accrued.v1` | Interval priced at its billing-cycle block position |
| `orion:energy:run_cost:estimated` | `energy.run_cost.estimated.v1` | Cost of one settled power intent |

It consumes `orion:power:intent:settled`.

Two run costs are published and never merged: `estimated_run_cost_usd` (Orion's
own meter, incremental over baseline, at the marginal tariff rate — the only
number autonomy may read) and `house_share_cost_usd` (share of the whole-house
interval; context only). A null cost always has a `*_gap` reason. Null is
unknown, not free. All costs are pre-tax (`cost_basis: pre_tax`).

## Feeding it (Plan 1: file drop)

1. On rockymountainpower.net: *Energy usage → Green Button → Download my data* (XML).
2. Copy the file into `${ENERGY_HOST_DATA_DIR}/inbox/`.
3. Within `ENERGY_SCAN_INTERVAL_SEC` the file moves to `processed/` (or `inbox/failed/` if unparseable).

Re-dropping an overlapping export is safe: the newer retrieval wins per interval.
Set `ENERGY_BILLING_CYCLE_START_DAY` to your bill's meter-read day.

## Debug queries

```sql
-- Cycle to date
SELECT cycle_start, max(cycle_accumulated_kwh) kwh, max(cycle_to_date_total_usd) usd
FROM energy_cost_accrued GROUP BY cycle_start ORDER BY cycle_start DESC LIMIT 3;

-- Recent run costs (null = unknown; read the gap column)
SELECT intent_id, workload_kind, energy_kwh, energy_basis, estimated_run_cost_usd,
       run_cost_gap, house_share_cost_usd, house_share_gap
FROM energy_run_cost ORDER BY window_start DESC LIMIT 20;
```

## Run

```bash
scripts/safe_docker_build.sh orion-energy up -d --build
docker logs --tail=100 orion-athena-orion-energy
```

## Tests

```bash
PYTHONPATH=. python -m pytest orion/energy/tests -q
cd services/orion-energy && PYTHONPATH=../..:. python -m pytest tests evals -q
```
````

Create `.github/workflows/orion-energy-tests.yml`:

```yaml
name: orion-energy-tests

on:
  pull_request:
    paths:
      - "services/orion-energy/**"
      - "orion/energy/**"
      - "orion/schemas/energy.py"
      - "orion/schemas/power.py"
      - "config/energy/**"
      - ".github/workflows/orion-energy-tests.yml"
  push:
    branches: [main]
    paths:
      - "services/orion-energy/**"
      - "orion/energy/**"
      - "orion/schemas/energy.py"
      - "orion/schemas/power.py"
      - "config/energy/**"
      - ".github/workflows/orion-energy-tests.yml"

jobs:
  energy-unit:
    name: Energy — package + service + eval
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r services/orion-energy/requirements.txt pytest
      - name: Package tests
        run: PYTHONPATH=. python -m pytest -q orion/energy/tests
      - name: Service tests + eval
        run: PYTHONPATH=.:services/orion-energy python -m pytest -q services/orion-energy/tests services/orion-energy/evals
```

Sync env and validate compose:

```bash
python3 scripts/sync_local_env_from_example.py
$PY scripts/check_env_template_parity.py
$PY scripts/check_compose_no_relative_mounts.py
scripts/safe_docker_build.sh orion-energy config >/dev/null && echo compose-ok
```

Expected: sync creates/extends `services/orion-energy/.env` in the primary checkout (report any skipped keys); parity and mount gates pass; `compose-ok`.

- [ ] **Step 7: Commit**

```bash
git add services/orion-energy .github/workflows/orion-energy-tests.yml
git status --short | grep -F '.env' | grep -v env_example && echo "STOP: .env staged" || true
git diff --cached --check
git commit -m "feat(energy): orion-energy service — Green Button drop, tariff accrual, run cost"
```

---

### Task 8: Bill-replay eval, spec status, live smoke, PR

**Files:**
- Create: `services/orion-energy/evals/test_energy_bill_replay_eval.py`
- Modify: `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md` (status + Plan 1 step 4 wording)
- Create: `docs/superpowers/pr-reports/2026-09-26-orion-energy-watcher-plan-1-pr.md`

**Interfaces:**
- Consumes: `EnergyPipeline`, `EnergyChannels` (Task 7); `UsageLedger` (Task 4); `load_tariff` (Task 3)

- [ ] **Step 1: Write the eval (independent oracle, hand arithmetic)**

Create `services/orion-energy/evals/test_energy_bill_replay_eval.py`:

```python
"""Replay a full synthetic September cycle and check against a hand-computed bill.

The oracle below is written out from the published Schedule 1 numbers, not by
calling the tariff code, so a bug in the tariff/ledger cannot pass by agreeing
with itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import ENERGY_ACCRUED_KIND, ENERGY_RUN_COST_KIND, EnergyUsageIntervalV1
from orion.schemas.power import PowerIntentSettledV1

REPO = Path(__file__).resolve().parents[3]
DENVER = ZoneInfo("America/Denver")
CYCLE_START = datetime(2026, 9, 1, tzinfo=DENVER)
HOURS = 30 * 24  # Sept 1 - Oct 1 local
RETRIEVED = datetime(2026, 10, 2, tzinfo=timezone.utc)

# Hand oracle: 720 kWh in a summer cycle.
MULT = (1 + (7.63 - 0.53) / 100) * (1 + (1.17 + 3.84 + 0.17) / 100)
ORACLE_ENERGY = (400 * 0.098332 + 320 * 0.125263) * MULT
ORACLE_TOTAL = ORACLE_ENERGY + 12.00 + 0.16


def _pipeline() -> EnergyPipeline:
    led = UsageLedger(
        load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"),
        tz=DENVER, cycle_start_day=1,
    )
    return EnergyPipeline(ledger=led, channels=EnergyChannels("u", "a", "r"))


def _month() -> list[EnergyUsageIntervalV1]:
    start = CYCLE_START.astimezone(timezone.utc)
    return [
        EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="UP123",
            interval_start=start + timedelta(hours=h), interval_end=start + timedelta(hours=h + 1),
            energy_kwh=1.0, retrieved_at=RETRIEVED,
        )
        for h in range(HOURS)
    ]


def test_cycle_total_matches_hand_bill() -> None:
    out = _pipeline().ingest_intervals(_month(), now=RETRIEVED)
    accrued = [o.payload for o in out if o.kind == ENERGY_ACCRUED_KIND]
    assert len(accrued) == HOURS
    last = accrued[-1]
    assert last.cycle_accumulated_kwh == pytest.approx(720.0)
    assert last.cycle_energy_cost_usd == pytest.approx(ORACLE_ENERGY, rel=1e-9)
    assert last.cycle_to_date_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert ORACLE_TOTAL == pytest.approx(101.6214, abs=1e-3)


def test_same_gpu_hour_costs_more_after_block_400() -> None:
    p = _pipeline()
    p.ingest_intervals(_month(), now=RETRIEVED)
    start = CYCLE_START.astimezone(timezone.utc)

    def run_at(hour: int) -> float:
        t = start + timedelta(hours=hour)
        settled = PowerIntentSettledV1(
            intent_id=f"run-{hour}", workload_kind="reverie.diffusion", node="circe", gpu_index=2,
            outcome="settled", window_start=t, window_end=t + timedelta(hours=1), sample_count=3600,
            actual_mean_watts=300.0, actual_peak_watts=320.0, energy_joules=300.0 * 3600,
            baseline_watts=50.0,
        )
        est = [o.payload for o in p.on_settlement(settled, now=RETRIEVED) if o.kind == ENERGY_RUN_COST_KIND][0]
        assert 0.0 <= est.house_share_cost_usd <= est.estimated_run_cost_usd + 1e-12
        return est.estimated_run_cost_usd

    early, late = run_at(100), run_at(600)  # 100 kWh vs 600 kWh into the cycle
    assert early == pytest.approx(0.25 * 0.098332 * MULT)
    assert late == pytest.approx(0.25 * 0.125263 * MULT)
    assert late > early
```

Run: `cd services/orion-energy && PYTHONPATH=../..:. $PY -m pytest evals -q; cd ../..`
Expected: 2 passed

> Why `house_share <= estimated`: each synthetic house hour is 1.0 kWh and the run's incremental 0.25 kWh sits inside it at the same block position, so the apportioned house cost equals the meter-side cost. If this ever fails, the two costs diverged in pricing — investigate, do not loosen.

- [ ] **Step 2: Update spec status and Plan 1 debug-surface wording**

In `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md`:

- Change the `**Status:**` line to: `**Status:** Plan 1 (cost primitive) implemented on feat/orion-energy-watcher — awaiting merge. Plan 2 (portal, bill reconcile, stakes, Hub) not started.`
- In "Recommended next patch" Plan 1 item 4, replace `Minimal Hub/debug read of accrued + run_cost (strip can wait for Plan 2).` with `Debug surface: Postgres tables + README queries (Hub strip is Plan 2).`

- [ ] **Step 3: Full focused gate run**

```bash
PYTHONPATH=. $PY -m pytest orion/energy/tests tests/test_energy_bus_catalog.py tests/test_home_cooling_bus_catalog.py -q
(cd services/orion-energy && PYTHONPATH=../..:. $PY -m pytest tests evals -q)
(cd services/orion-sql-writer && $PY -m pytest tests/test_energy_sql_shape.py tests/test_route_map_completeness.py tests/test_power_intent_settled_sql_shape.py -q)
$PY scripts/check_definition_drift.py --gate
$PY scripts/check_metric_lineage.py --gate
$PY scripts/check_env_template_parity.py
$PY scripts/check_compose_no_relative_mounts.py
git diff --check origin/main...HEAD
```

Expected: all green.

- [ ] **Step 4: Live smoke (runtime truth, or say UNVERIFIED)**

```bash
mkdir -p /mnt/storage-warm/orion-energy/inbox
scripts/safe_docker_build.sh orion-energy up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
cp orion/energy/tests/fixtures/espi_two_flows.xml /mnt/storage-warm/orion-energy/inbox/smoke.xml
sleep 70
docker logs --tail=50 orion-athena-orion-energy | grep -E "energy_(ingested|ledger_replayed|inbox)"
```

Then in Postgres (same DSN sql-writer uses):

```sql
SELECT count(*) FROM energy_usage_interval WHERE source_file LIKE '%smoke.xml';   -- expect 3
SELECT count(*) FROM energy_cost_accrued WHERE usage_point_id = 'UP123';           -- expect 3
```

After smoke, remove the fixture rows so they do not pollute the real ledger: move `/mnt/storage-warm/orion-energy/processed/*__smoke.xml` out of `processed/` (to `/tmp/orion-energy-smoke/`) and ask Juniper before deleting SQL rows (CLAUDE.md §13 — no `DELETE` without approval). If Docker or DB is unavailable, record the smoke as `UNVERIFIED` in the PR.

A real run-cost row needs a live `PowerIntentSettledV1` (diffusion host on circe) **and** a real Green Button export in the inbox; if neither has happened yet, state `energy_run_cost live path UNVERIFIED` in the PR.

- [ ] **Step 5: Graph refresh, review, commit, push, PR**

```bash
scripts/safe_graphify_update.sh
git add services/orion-energy/evals docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md
git commit -m "test(energy): bill-replay eval against hand-computed Schedule 1 bill"
```

Run the code-review skill in a subagent against `origin/main...HEAD`; fix every material finding with a test; re-run Step 3.

Write `docs/superpowers/pr-reports/2026-09-26-orion-energy-watcher-plan-1-pr.md` in the CLAUDE.md §18 template, including: the metric quality gate record below, env keys added (all of `services/orion-energy/.env_example`; sql-writer `SQL_WRITER_SUBSCRIBE_CHANNELS` extended), sync-script skipped keys, test/eval output, smoke result or `UNVERIFIED`, and restart commands:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-energy up -d --build
```

Metric quality gate record (paste into the PR):

1. **Provenance:** `energy_kwh` ← ESPI `IntervalReading.value × 10^powerOfTenMultiplier / 1000` (`orion/energy/espi.py:parse_espi`); run kWh ← `PowerIntentSettledV1.actual_mean_watts − baseline_watts` over the window (`services/orion-biometrics/app/power_intent.py:summarize`); dollars ← published Schedule 1 blocks × riders (`config/energy/tariff…yaml`).
2. **Independence:** house kWh is the utility meter (independent instrument); run kWh is the GPU sampler (independent of the utility). House share is derived from both and is explicitly labeled context, not a new independent signal.
3. **Theory anchor:** Marginal cost under increasing-block pricing — the price of the next kWh is set by cumulative cycle usage.
4. **Live data:** UNVERIFIED until a real Green Button export is dropped; the rest state is well-defined (no load ⇒ 0 kWh intervals with real $0 energy cost; unmeasured ⇒ null + gap), checked by tests.
5. **Existing mechanism:** none — no prior utility, tariff, or household-dollar signal in the repo (`rg -i "green.?button|tariff|kwh"`).
6. **Reversibility:** additive tables/channels; stop the service and nothing else depends on it until Plan 2 adds a consumer behind a default-off flag.

```bash
git add docs/superpowers/pr-reports/2026-09-26-orion-energy-watcher-plan-1-pr.md
git commit -m "docs(energy): Plan 1 PR report"
git push -u origin feat/orion-energy-watcher
gh pr create --title "feat(energy): house electricity cost primitive (Plan 1)" \
  --body-file docs/superpowers/pr-reports/2026-09-26-orion-energy-watcher-plan-1-pr.md
gh pr checks --watch
```

Expected: PR open, all checks green. Fix any failure at its cause (never `--no-verify`).
