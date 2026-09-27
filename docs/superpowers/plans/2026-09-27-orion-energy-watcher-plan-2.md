# Orion Energy Watcher — Plan 2 (portal, reconcile, stakes, Hub) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Orion compare its own tariff estimate to what Rocky Mountain Power actually billed or forecasts, fetch Green Button + bills from the RMP portal with a saved (MFA-kept) browser session, see whether the importer is healthy, let curiosity optionally hold a scheduled run when the house bill is trending at/over RMP's forecast (flag default off), and show all of it on a Hub Energy strip.

**Architecture:** Pure logic extends the shared package `orion/energy/` (reconcile, importer status, stakes snapshot). `services/orion-energy` gains a bill drop directory (JSON), reconcile on every bill and on late usage, and a 5-minute status tick that publishes importer status + stakes snapshot. A second compose service, `orion-energy-portal` (Playwright image, `profiles: ["portal"]`, not started by default), writes Green Button XML and bill JSON into the same drop directories plus a `status.json` — so pricing never depends on the scraper, and the portal code is the only thing that touches a browser. `orion-sql-writer` persists the five new kinds. Hub reads Postgres for a strip and for the curiosity gate.

**Tech Stack:** Python 3.12, pydantic 2.9, pydantic-settings, stdlib `json`/`re`/`zoneinfo`, Playwright 1.49 (portal image only, imported lazily), SQLAlchemy 2.0 + Postgres `ON CONFLICT`, asyncpg (Hub), FastAPI (Hub), vanilla JS + `node --test`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md` (Plan 2 = recommended-next-patch steps 5–8).

## Global Constraints

- Work only in worktree `/mnt/scripts/Orion-Sapienform-orion-energy-watcher-plan-2` on branch `feat/orion-energy-watcher-plan-2`.
- Local test interpreter: `PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python` (system python3 has no pydantic). CI uses plain `python`.
- **Unknown is never zero.** Missing / unmeasured → `None` with an explicit gap/reason. Never coerce to `0.0`. A stale importer never emits usage rows.
- **Two cost numbers, never merged.** `estimated_run_cost_usd` is the only run cost autonomy may read; nothing in this plan's curiosity gate reads `house_share_cost_usd`.
- Channels (exact): `orion:energy:bill:actual`, `orion:energy:bill:forecast`, `orion:energy:reconcile`, `orion:energy:stakes:snapshot`, `orion:energy:importer:status`.
- Kinds (exact): `energy.bill.actual.v1`, `energy.bill.forecast.v1`, `energy.reconcile.v1`, `energy.stakes.snapshot.v1`, `energy.importer.status.v1`.
- Importer states (exact): `healthy` / `stale` / `reauth_required` / `degraded`.
- Curiosity hold reason (exact): `held_off:energy_stakes`. Flag (exact): `ORION_ENERGY_STAKES_ENABLED`, default `false`. Flag off → curiosity behavior unchanged. Unknown / stale / unreadable snapshot never holds. A forced (operator) run is never held.
- **MFA stays on.** No RMP username/password in env, bus, Postgres, or logs. The headless fetcher only reuses a persistent Playwright profile; a dead session → `reauth_required`, no retry storm. Re-auth is a human, headed, one-time login.
- File drop and portal use the **same** parsers and event kinds (`source=file_drop` vs `rockymountain_power`).
- Portal URLs/selectors are **UNVERIFIED** until the first live spike; an empty download or empty bill table is an error state, never a silent success.
- Tariff fixes for systematic reconcile misses are config patches, not a second model.
- Bus URL: `ORION_BUS_URL=redis://100.92.216.81:6379/0`.
- Never commit `.env`. After any `.env_example` change run `python scripts/sync_local_env_from_example.py` from the worktree root and report skipped keys.
- Compose host paths must be absolute (`scripts/check_compose_no_relative_mounts.py`). Docker via `scripts/safe_docker_build.sh <service> <args>` only.
- No keyword/phrase triggers on user chat about bills or money.

## File Structure

| Path | Responsibility |
|---|---|
| `orion/schemas/energy.py` | Five new contracts + kinds + literals |
| `orion/schemas/registry.py`, `orion/bus/channels.yaml` | Register + catalog five channels |
| `tests/test_energy_bus_catalog.py` | Catalog ↔ registry for all eight energy channels |
| `config/metrics/metric_definitions.lock.json` | Re-locked by `check_definition_drift.py --update` |
| `services/orion-sql-writer/app/models/energy.py` | Five new tables |
| `services/orion-sql-writer/app/energy_persist.py` | Type-driven date coercion; upserts + insert-once |
| `services/orion-sql-writer/app/{models/__init__,worker,settings}.py`, `.env_example` | Routing + subscription |
| `orion/energy/testing.py` | Hand-checkable test tariff + hourly interval builder (tests only) |
| `orion/energy/ledger.py` | `window_prefix`, `latest_interval_end` |
| `orion/energy/reconcile.py` | Price a window, project a period, reconcile actual/forecast |
| `orion/energy/importer_status.py` | Portal status parse + importer state machine |
| `orion/energy/stakes.py` | Current forecast pick + stakes snapshot |
| `services/orion-energy/app/inbox.py` | Generic stamped drop-dir scan/replay; portal XML source |
| `services/orion-energy/app/bills.py` | Bill JSON drop directory |
| `services/orion-energy/app/portal_status.py` | Read `status.json` written by the portal |
| `services/orion-energy/app/pipeline.py` | Bills, reconcile, re-reconcile on late usage, status tick |
| `services/orion-energy/app/{main,settings}.py`, `.env_example`, `README.md` | Loops + config |
| `services/orion-energy/portal/` | Playwright adapter: settings, selectors, parse, driver, fetch, status, main, reauth |
| `services/orion-energy/{Dockerfile.portal,Dockerfile.portal.dockerignore,requirements-portal.txt,docker-compose.yml}` | Portal image + profiled compose service |
| `services/orion-hub/scripts/energy_stakes_gate.py` | Snapshot read + hold decision + attention row |
| `services/orion-hub/scripts/curiosity_investigation.py` | Flag-gated hold in `tick()` |
| `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example` | Wire flag + reader |
| `services/orion-hub/scripts/energy_routes.py`, `scripts/api_routes.py` | `/api/energy/latest`, `/api/energy/usage/daily` |
| `services/orion-hub/templates/index.html`, `static/js/energy-strip.js`, `static/js/biometrics-view.js` | Energy strip in the Cabinet subview |
| `services/orion-energy/evals/test_energy_reconcile_replay_eval.py` | Hand-oracle reconcile + end-to-end file-drop eval |

---

### Task 1: Contracts for bills, reconcile, stakes, importer status

**Files:**
- Modify: `orion/schemas/energy.py`
- Modify: `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- Modify: `config/metrics/metric_definitions.lock.json` (regenerated, never hand-edited)
- Test: `orion/energy/tests/test_energy_schemas.py`, `tests/test_energy_bus_catalog.py`

**Interfaces:**
- Produces: `EnergyBillActualV1`, `EnergyBillForecastV1`, `EnergyReconcileV1`, `EnergyStakesSnapshotV1`, `EnergyImporterStatusV1`; constants `ENERGY_BILL_ACTUAL_KIND`, `ENERGY_BILL_FORECAST_KIND`, `ENERGY_RECONCILE_KIND`, `ENERGY_STAKES_KIND`, `ENERGY_IMPORTER_STATUS_KIND`; literals `ReconcileKind`, `ReconcileMethod`, `ReconcileGap`, `UtilityBasis`, `ImporterState`, `ImporterSource`, `StakesPressure`.

- [ ] **Step 1: Write failing schema tests** — append to `orion/energy/tests/test_energy_schemas.py`:

```python
from datetime import date

from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyImporterStatusV1,
    EnergyReconcileV1,
    EnergyStakesSnapshotV1,
)

_T = datetime(2026, 9, 12, 6, tzinfo=timezone.utc)


def _bill(**over) -> dict:
    base = dict(
        source="file_drop", billing_period_start=date(2026, 8, 12), billing_period_end=date(2026, 9, 11),
        kwh_billed=712.0, current_charges=101.23, retrieved_at=_T,
    )
    base.update(over)
    return base


def test_bill_actual_missing_lines_are_unknown_not_zero() -> None:
    bill = EnergyBillActualV1(**_bill())
    assert bill.taxes is None and bill.energy_charge is None and bill.credits is None


def test_bill_actual_rejects_backwards_period() -> None:
    with pytest.raises(ValueError):
        EnergyBillActualV1(**_bill(billing_period_end=date(2026, 8, 12)))


def test_forecast_with_no_projection_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnergyBillForecastV1(
            source="file_drop", billing_period_start=date(2026, 9, 11), as_of=_T, retrieved_at=_T,
        )


def _rec(**over) -> dict:
    base = dict(
        reconcile_kind="actual", usage_point_id="UP1", billing_period_start=date(2026, 8, 12),
        billing_period_end=date(2026, 9, 11), utility_as_of=_T, utility_kwh=712.0,
        utility_total_usd=97.13, utility_basis="pre_tax", orion_method="metered_period",
        tariff_version="t1", computed_at=_T,
    )
    base.update(over)
    return base


def test_reconcile_needs_exactly_one_of_total_or_gap() -> None:
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec())
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec(orion_total_usd=99.0, reconcile_gap="no_usage"))
    assert EnergyReconcileV1(**_rec(reconcile_gap="no_usage")).orion_total_usd is None


def test_reconcile_gap_forbids_deltas() -> None:
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec(reconcile_gap="usage_incomplete", delta_usd=1.0))


def test_stakes_comparison_pressures_need_a_ratio() -> None:
    base = dict(as_of=_T, importer_state="healthy", pressure_reason="ratio=1.2")
    with pytest.raises(ValueError):
        EnergyStakesSnapshotV1(**base, pressure="over_forecast")
    ok = EnergyStakesSnapshotV1(**base, pressure="over_forecast", projected_to_forecast_ratio=1.2)
    assert ok.forecast_total_usd is None
    assert EnergyStakesSnapshotV1(**base, pressure="unknown").projected_to_forecast_ratio is None


def test_importer_status_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        EnergyImporterStatusV1(state="stale", reason="", source="file_drop", as_of=_T)
```

(`datetime`, `timezone`, `pytest` are already imported at the top of this file; if not, add them.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_schemas.py -q`
Expected: FAIL — `ImportError: cannot import name 'EnergyBillActualV1'`

- [ ] **Step 3: Implement contracts** — in `orion/schemas/energy.py`, update the module docstring's first line to `"""House electricity contracts: usage, accrual, run cost, bills, reconcile, stakes."""`, add after `ENERGY_RUN_COST_KIND`:

```python
ENERGY_BILL_ACTUAL_KIND = "energy.bill.actual.v1"
ENERGY_BILL_FORECAST_KIND = "energy.bill.forecast.v1"
ENERGY_RECONCILE_KIND = "energy.reconcile.v1"
ENERGY_STAKES_KIND = "energy.stakes.snapshot.v1"
ENERGY_IMPORTER_STATUS_KIND = "energy.importer.status.v1"
```

after `CostBasis = Literal["pre_tax"]`:

```python
ReconcileKind = Literal["actual", "forecast"]
ReconcileMethod = Literal["metered_period", "linear_run_rate"]
ReconcileGap = Literal["no_usage", "usage_incomplete"]
UtilityBasis = Literal["pre_tax", "tax_unknown"]
ImporterState = Literal["healthy", "stale", "reauth_required", "degraded"]
ImporterSource = Literal["portal", "file_drop"]
StakesPressure = Literal["unknown", "normal", "near_forecast", "over_forecast"]
```

and append at the end of the file:

```python
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
    """

    model_config = ConfigDict(extra="forbid")

    reconcile_kind: ReconcileKind
    usage_point_id: str = Field(min_length=1)
    billing_period_start: date
    billing_period_end: Optional[date] = None
    utility_as_of: datetime
    utility_kwh: Optional[float] = Field(default=None, ge=0.0)
    utility_total_usd: Optional[float] = None
    utility_basis: UtilityBasis
    orion_method: ReconcileMethod
    orion_covered_through: Optional[datetime] = None
    orion_kwh: Optional[float] = Field(default=None, ge=0.0)
    orion_energy_usd: Optional[float] = Field(default=None, ge=0.0)
    orion_fixed_usd: Optional[float] = Field(default=None, ge=0.0)
    orion_total_usd: Optional[float] = Field(default=None, ge=0.0)
    reconcile_gap: Optional[ReconcileGap] = None
    delta_kwh: Optional[float] = None
    delta_usd: Optional[float] = None
    delta_pct: Optional[float] = None
    bucket_deltas: dict[str, float] = Field(default_factory=dict)
    tariff_version: str = Field(min_length=1)
    cost_basis: CostBasis = "pre_tax"
    computed_at: datetime

    @field_validator("utility_as_of", "orion_covered_through", "computed_at")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _null_means_reason(self) -> "EnergyReconcileV1":
        if (self.orion_total_usd is None) == (self.reconcile_gap is None):
            raise ValueError("exactly one of orion_total_usd / reconcile_gap must be set")
        if self.reconcile_gap is not None and (
            self.delta_kwh is not None or self.delta_usd is not None
            or self.delta_pct is not None or self.bucket_deltas
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
    cycle_accumulated_kwh: Optional[float] = Field(default=None, ge=0.0)
    cycle_to_date_total_usd: Optional[float] = Field(default=None, ge=0.0)
    marginal_usd_per_kwh: Optional[float] = Field(default=None, ge=0.0)
    orion_projected_total_usd: Optional[float] = Field(default=None, ge=0.0)
    forecast_total_usd: Optional[float] = None
    forecast_as_of: Optional[datetime] = None
    projected_to_forecast_ratio: Optional[float] = None
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
```

- [ ] **Step 4: Run schema tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_schemas.py -q`
Expected: all pass (7 old + 7 new).

- [ ] **Step 5: Extend the catalog test (failing)** — in `tests/test_energy_bus_catalog.py` extend the import and `CASES`:

```python
from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyImporterStatusV1,
    EnergyReconcileV1,
    EnergyRunCostEstimatedV1,
    EnergyStakesSnapshotV1,
    EnergyUsageIntervalV1,
)
```

```python
CASES = [
    ("orion:energy:usage:observed", "EnergyUsageIntervalV1", "energy.usage.observed.v1", EnergyUsageIntervalV1),
    ("orion:energy:cost:accrued", "EnergyCostAccruedV1", "energy.cost.accrued.v1", EnergyCostAccruedV1),
    ("orion:energy:run_cost:estimated", "EnergyRunCostEstimatedV1", "energy.run_cost.estimated.v1", EnergyRunCostEstimatedV1),
    ("orion:energy:bill:actual", "EnergyBillActualV1", "energy.bill.actual.v1", EnergyBillActualV1),
    ("orion:energy:bill:forecast", "EnergyBillForecastV1", "energy.bill.forecast.v1", EnergyBillForecastV1),
    ("orion:energy:reconcile", "EnergyReconcileV1", "energy.reconcile.v1", EnergyReconcileV1),
    ("orion:energy:stakes:snapshot", "EnergyStakesSnapshotV1", "energy.stakes.snapshot.v1", EnergyStakesSnapshotV1),
    ("orion:energy:importer:status", "EnergyImporterStatusV1", "energy.importer.status.v1", EnergyImporterStatusV1),
]
```

Run: `PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py -q`
Expected: FAIL — `KeyError: 'orion:energy:bill:actual'`

- [ ] **Step 6: Register schemas and channels** — in `orion/schemas/registry.py`, extend the existing `from orion.schemas.energy import (...)` block with the five new models; add them to the plain name map next to `"EnergyRunCostEstimatedV1": EnergyRunCostEstimatedV1,`; add to `SCHEMA_REGISTRY` next to the `"EnergyRunCostEstimatedV1": SchemaRegistration(...)` block:

```python
    "EnergyBillActualV1": SchemaRegistration(model=EnergyBillActualV1, kind="energy.bill.actual.v1"),
    "EnergyBillForecastV1": SchemaRegistration(model=EnergyBillForecastV1, kind="energy.bill.forecast.v1"),
    "EnergyReconcileV1": SchemaRegistration(model=EnergyReconcileV1, kind="energy.reconcile.v1"),
    "EnergyStakesSnapshotV1": SchemaRegistration(model=EnergyStakesSnapshotV1, kind="energy.stakes.snapshot.v1"),
    "EnergyImporterStatusV1": SchemaRegistration(model=EnergyImporterStatusV1, kind="energy.importer.status.v1"),
```

In `orion/bus/channels.yaml`, directly after the `orion:energy:run_cost:estimated` entry:

```yaml
  - name: "orion:energy:bill:actual"
    kind: "event"
    schema_id: "EnergyBillActualV1"
    message_kind: "energy.bill.actual.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-27"
    description: "Closed Rocky Mountain Power billing period (portal scrape or hand-entered bill JSON)."

  - name: "orion:energy:bill:forecast"
    kind: "event"
    schema_id: "EnergyBillForecastV1"
    message_kind: "energy.bill.forecast.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-27"
    description: "RMP's in-cycle bill projection; rejected when it carries no projected number."

  - name: "orion:energy:reconcile"
    kind: "event"
    schema_id: "EnergyReconcileV1"
    message_kind: "energy.reconcile.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-27"
    description: "Orion tariff estimate vs RMP bill/forecast; deltas are Orion minus utility, gap when usage is incomplete."

  - name: "orion:energy:stakes:snapshot"
    kind: "event"
    schema_id: "EnergyStakesSnapshotV1"
    message_kind: "energy.stakes.snapshot.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-27"
    description: "Cycle-to-date cost, marginal rate, and projected-vs-forecast pressure for spend gates (Hub reads the SQL row)."

  - name: "orion:energy:importer:status"
    kind: "event"
    schema_id: "EnergyImporterStatusV1"
    message_kind: "energy.importer.status.v1"
    producer_services: ["orion-energy"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-09-27"
    description: "healthy / stale / reauth_required / degraded for house usage ingest."
```

- [ ] **Step 7: Run catalog + gates**

```bash
PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py orion/energy/tests/test_energy_schemas.py -q
$PY scripts/check_definition_drift.py --update
$PY scripts/check_definition_drift.py --gate
$PY scripts/check_metric_lineage.py --gate
```

Expected: all pass; drift `--update` adds the five new `metric://bus_channel/orion-energy/...` entries; both gates PASS. If the lineage gate fails, add exactly what its message names — do not suppress.

- [ ] **Step 8: Commit**

```bash
git add orion/schemas/energy.py orion/schemas/registry.py orion/bus/channels.yaml \
  config/metrics/metric_definitions.lock.json tests/test_energy_bus_catalog.py \
  orion/energy/tests/test_energy_schemas.py
git diff --cached --check
git commit -m "feat(energy): contracts for bills, reconcile, stakes, importer status"
```

---

### Task 2: sql-writer persistence for the five new kinds

**Files:**
- Modify: `services/orion-sql-writer/app/models/energy.py`, `app/models/__init__.py`, `app/energy_persist.py`, `app/worker.py`, `app/settings.py`, `.env_example`
- Test: `services/orion-sql-writer/tests/test_energy_sql_shape.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: tables `energy_bill_actual`, `energy_bill_forecast`, `energy_reconcile`, `energy_stakes_snapshot`, `energy_importer_status`; SQL classes `EnergyBillActualSQL`, `EnergyBillForecastSQL`, `EnergyReconcileSQL`, `EnergyStakesSnapshotSQL`, `EnergyImporterStatusSQL`. Hub (Tasks 7–8) reads these table/column names.

- [ ] **Step 1: Write failing tests** — in `services/orion-sql-writer/tests/test_energy_sql_shape.py` extend imports:

```python
from app.models.energy import (
    EnergyBillActualSQL,
    EnergyBillForecastSQL,
    EnergyCostAccruedSQL,
    EnergyImporterStatusSQL,
    EnergyReconcileSQL,
    EnergyRunCostSQL,
    EnergyStakesSnapshotSQL,
    EnergyUsageIntervalSQL,
)
from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyImporterStatusV1,
    EnergyReconcileV1,
    EnergyRunCostEstimatedV1,
    EnergyStakesSnapshotV1,
    EnergyUsageIntervalV1,
)
```

append to `ROUTES`:

```python
    ("energy.bill.actual.v1", "EnergyBillActualSQL", EnergyBillActualSQL, EnergyBillActualV1, "orion:energy:bill:actual"),
    ("energy.bill.forecast.v1", "EnergyBillForecastSQL", EnergyBillForecastSQL, EnergyBillForecastV1, "orion:energy:bill:forecast"),
    ("energy.reconcile.v1", "EnergyReconcileSQL", EnergyReconcileSQL, EnergyReconcileV1, "orion:energy:reconcile"),
    ("energy.stakes.snapshot.v1", "EnergyStakesSnapshotSQL", EnergyStakesSnapshotSQL, EnergyStakesSnapshotV1, "orion:energy:stakes:snapshot"),
    ("energy.importer.status.v1", "EnergyImporterStatusSQL", EnergyImporterStatusSQL, EnergyImporterStatusV1, "orion:energy:importer:status"),
```

and append tests:

```python
def test_bill_actual_upsert_by_period_and_dates_stay_dates() -> None:
    sess = _CapturingSession()
    row = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 8, 12), billing_period_end=date(2026, 9, 11),
        kwh_billed=712.0, current_charges=101.23, due_date=date(2026, 10, 2), retrieved_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyBillActualSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_bill_actual_period" in sql
    assert "energy_bill_actual.retrieved_at <= excluded.retrieved_at" in sql
    params = stmt.compile(dialect=postgresql.dialect()).params
    # Regression: the old suffix rule parsed any *_start/*_end string as a datetime.
    assert params["billing_period_start"] == date(2026, 8, 12)
    assert params["billing_period_end"] == date(2026, 9, 11)
    assert params["due_date"] == date(2026, 10, 2)
    assert params["retrieved_at"] == T1


def test_forecast_upsert_by_period_and_as_of() -> None:
    sess = _CapturingSession()
    row = EnergyBillForecastV1(
        source="rockymountain_power", billing_period_start=date(2026, 9, 11), as_of=T0,
        projected_total_usd=96.0, retrieved_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyBillForecastSQL](sess, row)
    sql = _sql(sess.statements[0])
    assert "uq_energy_bill_forecast_period_as_of" in sql
    assert "energy_bill_forecast.retrieved_at <= excluded.retrieved_at" in sql


def test_reconcile_upsert_keeps_bucket_deltas() -> None:
    sess = _CapturingSession()
    row = EnergyReconcileV1(
        reconcile_kind="actual", usage_point_id="UP1", billing_period_start=date(2026, 8, 12),
        billing_period_end=date(2026, 9, 11), utility_as_of=T0, utility_kwh=712.0,
        utility_total_usd=97.13, utility_basis="pre_tax", orion_method="metered_period",
        orion_kwh=710.0, orion_energy_usd=85.0, orion_fixed_usd=12.16, orion_total_usd=97.16,
        delta_kwh=-2.0, delta_usd=0.03, delta_pct=0.0003, bucket_deltas={"customer_charge": 0.16},
        tariff_version="t1", computed_at=T1,
    ).model_dump(mode="json")
    ENERGY_UPSERTS[EnergyReconcileSQL](sess, row)
    stmt = sess.statements[0]
    sql = _sql(stmt)
    assert "uq_energy_reconcile_key" in sql
    assert "energy_reconcile.computed_at <= excluded.computed_at" in sql
    assert stmt.compile(dialect=postgresql.dialect()).params["bucket_deltas"] == {"customer_charge": 0.16}


@pytest.mark.parametrize(
    "model,payload,constraint",
    [
        (
            EnergyStakesSnapshotSQL,
            EnergyStakesSnapshotV1(as_of=T1, importer_state="stale", pressure="unknown", pressure_reason="importer_stale"),
            "uq_energy_stakes_snapshot_as_of",
        ),
        (
            EnergyImporterStatusSQL,
            EnergyImporterStatusV1(state="stale", reason="no_usage_yet", source="file_drop", as_of=T1),
            "uq_energy_importer_status_as_of",
        ),
    ],
)
def test_snapshots_insert_once(model, payload, constraint) -> None:
    sess = _CapturingSession()
    ENERGY_UPSERTS[model](sess, payload.model_dump(mode="json"))
    sql = _sql(sess.statements[0])
    assert f"ON CONFLICT ON CONSTRAINT {constraint} DO NOTHING" in sql
    assert sess.committed
```

Run: `cd services/orion-sql-writer && PYTHONPATH=.:../.. $PY -m pytest tests/test_energy_sql_shape.py -q` (use whatever invocation `.github/workflows/orion-sql-writer-tests.yml` uses for this file if it differs)
Expected: FAIL — `ImportError: cannot import name 'EnergyBillActualSQL'`

- [ ] **Step 2: Add tables** — append to `services/orion-sql-writer/app/models/energy.py` (add `from sqlalchemy.dialects.postgresql import JSONB` to imports):

```python
class EnergyBillActualSQL(Base):
    """Closed RMP billing period (``orion:energy:bill:actual``). NULL money = not on the bill."""

    __tablename__ = "energy_bill_actual"
    __table_args__ = (
        UniqueConstraint("billing_period_start", "billing_period_end", name="uq_energy_bill_actual_period"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=True)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=False)
    kwh_billed = Column(Float, nullable=False)
    energy_charge = Column(Float, nullable=True)
    customer_charge = Column(Float, nullable=True)
    adjustments = Column(Float, nullable=True)
    fees = Column(Float, nullable=True)
    taxes = Column(Float, nullable=True)
    credits = Column(Float, nullable=True)
    current_charges = Column(Float, nullable=False)
    amount_due = Column(Float, nullable=True)
    due_date = Column(Date, nullable=True)
    statement_artifact_id = Column(String, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyBillForecastSQL(Base):
    """RMP in-cycle projection (``orion:energy:bill:forecast``), one row per (period, as_of)."""

    __tablename__ = "energy_bill_forecast"
    __table_args__ = (
        UniqueConstraint("billing_period_start", "as_of", name="uq_energy_bill_forecast_period_as_of"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=True)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=True)
    as_of = Column(DateTime(timezone=True), nullable=False)
    days_into_cycle = Column(Integer, nullable=True)
    projected_kwh = Column(Float, nullable=True)
    projected_total_usd = Column(Float, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    source_file = Column(String, nullable=True)


class EnergyReconcileSQL(Base):
    """Orion vs RMP (``orion:energy:reconcile``). NULL Orion total always has a ``reconcile_gap``."""

    __tablename__ = "energy_reconcile"
    __table_args__ = (
        UniqueConstraint(
            "reconcile_kind", "billing_period_start", "utility_as_of", "tariff_version",
            name="uq_energy_reconcile_key",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    reconcile_kind = Column(String, nullable=False)
    usage_point_id = Column(String, nullable=False)
    billing_period_start = Column(Date, nullable=False, index=True)
    billing_period_end = Column(Date, nullable=True)
    utility_as_of = Column(DateTime(timezone=True), nullable=False)
    utility_kwh = Column(Float, nullable=True)
    utility_total_usd = Column(Float, nullable=True)
    utility_basis = Column(String, nullable=False)
    orion_method = Column(String, nullable=False)
    orion_covered_through = Column(DateTime(timezone=True), nullable=True)
    orion_kwh = Column(Float, nullable=True)
    orion_energy_usd = Column(Float, nullable=True)
    orion_fixed_usd = Column(Float, nullable=True)
    orion_total_usd = Column(Float, nullable=True)
    reconcile_gap = Column(String, nullable=True)
    delta_kwh = Column(Float, nullable=True)
    delta_usd = Column(Float, nullable=True)
    delta_pct = Column(Float, nullable=True)
    bucket_deltas = Column(JSONB, nullable=False, default=dict)
    tariff_version = Column(String, nullable=False)
    cost_basis = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class EnergyStakesSnapshotSQL(Base):
    """Stakes snapshot (``orion:energy:stakes:snapshot``); Hub + curiosity read the latest row."""

    __tablename__ = "energy_stakes_snapshot"
    __table_args__ = (UniqueConstraint("as_of", name="uq_energy_stakes_snapshot_as_of"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
    usage_point_id = Column(String, nullable=True)
    cycle_start = Column(Date, nullable=True)
    cycle_end = Column(Date, nullable=True)
    covered_through = Column(DateTime(timezone=True), nullable=True)
    cycle_accumulated_kwh = Column(Float, nullable=True)
    cycle_to_date_total_usd = Column(Float, nullable=True)
    marginal_usd_per_kwh = Column(Float, nullable=True)
    orion_projected_total_usd = Column(Float, nullable=True)
    forecast_total_usd = Column(Float, nullable=True)
    forecast_as_of = Column(DateTime(timezone=True), nullable=True)
    projected_to_forecast_ratio = Column(Float, nullable=True)
    importer_state = Column(String, nullable=False)
    pressure = Column(String, nullable=False)
    pressure_reason = Column(String, nullable=False)
    tariff_version = Column(String, nullable=True)


class EnergyImporterStatusSQL(Base):
    """Importer health (``orion:energy:importer:status``)."""

    __tablename__ = "energy_importer_status"
    __table_args__ = (UniqueConstraint("as_of", name="uq_energy_importer_status_as_of"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    state = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    source = Column(String, nullable=False)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    latest_interval_end = Column(DateTime(timezone=True), nullable=True)
    usage_lag_hours = Column(Float, nullable=True)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
```

- [ ] **Step 3: Persist** — in `services/orion-sql-writer/app/energy_persist.py`, replace the imports and `_columns` and add insert-once + the new functions:

```python
from datetime import date, datetime
from typing import Any, Callable

from sqlalchemy import Date, DateTime, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.energy import (
    EnergyBillActualSQL,
    EnergyBillForecastSQL,
    EnergyCostAccruedSQL,
    EnergyImporterStatusSQL,
    EnergyReconcileSQL,
    EnergyRunCostSQL,
    EnergyStakesSnapshotSQL,
    EnergyUsageIntervalSQL,
)


def _columns(model: type, data: dict[str, Any]) -> dict[str, Any]:
    # Coerce by the column's SQL type, not its name: billing_period_start is a DATE.
    types = {col.key: col.type for col in inspect(model).columns if col.key != "id"}
    out = {k: v for k, v in data.items() if k in types}
    for key, value in list(out.items()):
        if not isinstance(value, str):
            continue
        if isinstance(types[key], DateTime):
            out[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(types[key], Date):
            out[key] = date.fromisoformat(value)
    return out
```

(keep `_upsert` and the three existing upsert functions unchanged), then add:

```python
def _insert_once(sess: Session, model: type, data: dict[str, Any], *, constraint: str) -> bool:
    stmt = insert(model).values(**_columns(model, data)).on_conflict_do_nothing(constraint=constraint)
    sess.execute(stmt)
    sess.commit()
    return True


def upsert_energy_bill_actual(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyBillActualSQL, data,
        conflict={"constraint": "uq_energy_bill_actual_period"}, newer_col="retrieved_at",
    )


def upsert_energy_bill_forecast(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyBillForecastSQL, data,
        conflict={"constraint": "uq_energy_bill_forecast_period_as_of"}, newer_col="retrieved_at",
    )


def upsert_energy_reconcile(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyReconcileSQL, data,
        conflict={"constraint": "uq_energy_reconcile_key"}, newer_col="computed_at",
    )


def insert_energy_stakes_snapshot(sess: Session, data: dict[str, Any]) -> bool:
    return _insert_once(sess, EnergyStakesSnapshotSQL, data, constraint="uq_energy_stakes_snapshot_as_of")


def insert_energy_importer_status(sess: Session, data: dict[str, Any]) -> bool:
    return _insert_once(sess, EnergyImporterStatusSQL, data, constraint="uq_energy_importer_status_as_of")
```

and extend `ENERGY_UPSERTS`:

```python
ENERGY_UPSERTS: dict[type, Callable[[Session, dict[str, Any]], bool]] = {
    EnergyUsageIntervalSQL: upsert_energy_usage_interval,
    EnergyCostAccruedSQL: upsert_energy_cost_accrued,
    EnergyRunCostSQL: upsert_energy_run_cost,
    EnergyBillActualSQL: upsert_energy_bill_actual,
    EnergyBillForecastSQL: upsert_energy_bill_forecast,
    EnergyReconcileSQL: upsert_energy_reconcile,
    EnergyStakesSnapshotSQL: insert_energy_stakes_snapshot,
    EnergyImporterStatusSQL: insert_energy_importer_status,
}
```

Note: `_upsert`'s `immutable` set only covers `index_elements`; for constraint conflicts the natural-key columns are re-set to their identical values, which is harmless (existing behavior).

- [ ] **Step 4: Route + subscribe**
  - `app/models/__init__.py`: import and export the five new SQL classes next to `EnergyRunCostSQL` (same style as the existing lines ~L20 and ~L119–121).
  - `app/worker.py` `MODEL_MAP` (next to `"EnergyRunCostSQL": ...` ~L476), importing the five schemas from `orion.schemas.energy`:

```python
    "EnergyBillActualSQL": (EnergyBillActualSQL, EnergyBillActualV1),
    "EnergyBillForecastSQL": (EnergyBillForecastSQL, EnergyBillForecastV1),
    "EnergyReconcileSQL": (EnergyReconcileSQL, EnergyReconcileV1),
    "EnergyStakesSnapshotSQL": (EnergyStakesSnapshotSQL, EnergyStakesSnapshotV1),
    "EnergyImporterStatusSQL": (EnergyImporterStatusSQL, EnergyImporterStatusV1),
```

  - `app/settings.py` `DEFAULT_ROUTE_MAP` (after `"energy.run_cost.estimated.v1": "EnergyRunCostSQL",`):

```python
    "energy.bill.actual.v1": "EnergyBillActualSQL",
    "energy.bill.forecast.v1": "EnergyBillForecastSQL",
    "energy.reconcile.v1": "EnergyReconcileSQL",
    "energy.stakes.snapshot.v1": "EnergyStakesSnapshotSQL",
    "energy.importer.status.v1": "EnergyImporterStatusSQL",
```

  - Run `rg -n "orion:energy:run_cost:estimated" services/orion-sql-writer` and at **every** hit in `app/settings.py` (default channel list ~L166 and the list `effective_subscribe_channels` merges ~L675) and `.env_example` (`SQL_WRITER_SUBSCRIBE_CHANNELS`, and `SQL_WRITER_ROUTE_MAP_JSON` if it lists energy kinds) add the five new channels (and kinds→models in the route-map JSON). Then run `python scripts/sync_local_env_from_example.py` from the worktree root.

- [ ] **Step 5: Run tests**

Run: the same command as Step 1.
Expected: all pass (parametrized cases now 8 routes; 4 new tests).

- [ ] **Step 6: Commit**

```bash
git add services/orion-sql-writer/app services/orion-sql-writer/.env_example services/orion-sql-writer/tests/test_energy_sql_shape.py
git diff --cached --check
git commit -m "feat(sql-writer): persist energy bills, reconcile, stakes, importer status"
```

---

### Task 3: Reconcile — Orion's estimate vs RMP bill and forecast

**Files:**
- Create: `orion/energy/testing.py`, `orion/energy/reconcile.py`
- Modify: `orion/energy/ledger.py`
- Test: `orion/energy/tests/test_energy_reconcile.py`, `orion/energy/tests/test_energy_ledger.py`

**Interfaces:**
- Consumes: `UsageLedger` (`tariff`, `tz`, `cycle_bounds`, `_in_cycle`, `_contiguous_from_cycle_start`), `Tariff.energy_cost_usd`, `Tariff.fixed_monthly_usd`, `Tariff.version`, `Tariff.cost_basis`; Task 1 models.
- Produces:
  - `UsageLedger.window_prefix(usage_point_id: str, start: datetime, end: datetime) -> tuple[list[EnergyUsageIntervalV1], Optional[datetime]]`
  - `UsageLedger.latest_interval_end(usage_point_id: str) -> Optional[datetime]`
  - `orion.energy.reconcile`: `MIN_RUN_RATE_HOURS: float`, `local_midnight(day: date, tz: ZoneInfo) -> datetime`, `period_bounds(start_day: date, end_day: Optional[date], ledger: UsageLedger) -> tuple[datetime, datetime]`, `price_intervals(tariff, intervals, *, tz) -> tuple[float, float]`, `PeriodProjection` (fields `observed_kwh, observed_energy_usd, projected_kwh, projected_energy_usd, fixed_usd, covered_through`; property `projected_total_usd`), `project_period(ledger, usage_point_id, start, end) -> tuple[Optional[PeriodProjection], Optional[ReconcileGap], Optional[datetime]]`, `reconcile_actual(bill, *, ledger, usage_point_id, computed_at) -> EnergyReconcileV1`, `reconcile_forecast(forecast, *, ledger, usage_point_id, computed_at) -> EnergyReconcileV1`
  - `orion.energy.testing`: `UTC`, `flat_test_tariff() -> Tariff`, `make_test_ledger(cycle_start_day: int = 1) -> UsageLedger`, `hourly(start, hours, *, kwh=1.0, point="UP1", retrieved_at=None, skip=frozenset()) -> list[EnergyUsageIntervalV1]`

- [ ] **Step 1: Create the test helper** `orion/energy/testing.py`:

```python
"""Hand-checkable fixtures shared by orion/energy and services/orion-energy tests.

Not used at runtime. The flat tariff makes every oracle doable on paper:
400 kWh at $0.10, then $0.12, $10 fixed, no riders, one season.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import Block, Season, Tariff
from orion.schemas.energy import EnergyUsageIntervalV1

UTC = ZoneInfo("UTC")


def flat_test_tariff() -> Tariff:
    return Tariff(
        version="test-flat-v1",
        cost_basis="pre_tax",
        seasons=(
            Season(
                name="all",
                months=frozenset(range(1, 13)),
                blocks=(Block(up_to_kwh=400.0, usd_per_kwh=0.10), Block(up_to_kwh=None, usd_per_kwh=0.12)),
            ),
        ),
        energy_multiplier=1.0,
        fixed_monthly_usd=10.0,
    )


def make_test_ledger(cycle_start_day: int = 1) -> UsageLedger:
    return UsageLedger(flat_test_tariff(), tz=UTC, cycle_start_day=cycle_start_day)


def hourly(
    start: datetime,
    hours: int,
    *,
    kwh: float = 1.0,
    point: str = "UP1",
    retrieved_at: Optional[datetime] = None,
    skip: frozenset[int] = frozenset(),
) -> list[EnergyUsageIntervalV1]:
    got = retrieved_at or (start + timedelta(days=60))
    return [
        EnergyUsageIntervalV1(
            source="file_drop", usage_point_id=point,
            interval_start=start + timedelta(hours=h), interval_end=start + timedelta(hours=h + 1),
            energy_kwh=kwh, retrieved_at=got,
        )
        for h in range(hours)
        if h not in skip
    ]
```

- [ ] **Step 2: Write failing tests**

Append to `orion/energy/tests/test_energy_ledger.py`:

```python
from orion.energy.testing import hourly, make_test_ledger

_S = datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_window_prefix_stops_at_the_first_hole() -> None:
    led = make_test_ledger()
    for iv in hourly(_S, 10, skip=frozenset({4})):
        led.upsert(iv)
    prefix, covered = led.window_prefix("UP1", _S, _S + timedelta(hours=10))
    assert len(prefix) == 4
    assert covered == _S + timedelta(hours=4)


def test_window_prefix_empty_when_start_missing() -> None:
    led = make_test_ledger()
    for iv in hourly(_S + timedelta(hours=1), 3):
        led.upsert(iv)
    assert led.window_prefix("UP1", _S, _S + timedelta(hours=5)) == ([], None)


def test_latest_interval_end() -> None:
    led = make_test_ledger()
    assert led.latest_interval_end("UP1") is None
    for iv in hourly(_S, 5, skip=frozenset({2})):
        led.upsert(iv)
    assert led.latest_interval_end("UP1") == _S + timedelta(hours=5)
```

(add `timedelta` to the file's `datetime` import if missing.)

Create `orion/energy/tests/test_energy_reconcile.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from orion.energy.reconcile import reconcile_actual, reconcile_forecast
from orion.energy.testing import hourly, make_test_ledger
from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def _bill(**over) -> EnergyBillActualV1:
    base = dict(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 9, 3),
        kwh_billed=470.0, energy_charge=48.0, customer_charge=10.0, taxes=3.0,
        current_charges=64.6, retrieved_at=NOW,
    )
    base.update(over)
    return EnergyBillActualV1(**base)


def _ledger_with(intervals):
    led = make_test_ledger()
    for iv in intervals:
        led.upsert(iv)
    return led


def test_actual_full_period_hand_oracle() -> None:
    # 48 h x 10 kWh = 480 kWh: 400 @ 0.10 + 80 @ 0.12 = 49.60, + 10 fixed = 59.60.
    rec = reconcile_actual(_bill(), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap is None
    assert rec.orion_kwh == pytest.approx(480.0)
    assert rec.orion_energy_usd == pytest.approx(49.6)
    assert rec.orion_total_usd == pytest.approx(59.6)
    assert rec.utility_basis == "pre_tax"
    assert rec.utility_total_usd == pytest.approx(61.6)  # 64.60 - 3.00 tax
    assert rec.delta_usd == pytest.approx(-2.0)
    assert rec.delta_kwh == pytest.approx(10.0)
    assert rec.delta_pct == pytest.approx(-2.0 / 61.6)
    assert rec.bucket_deltas == pytest.approx({"energy_charge": 1.6, "customer_charge": 0.0})
    assert rec.orion_method == "metered_period"
    assert rec.tariff_version == "test-flat-v1"


def test_actual_bucket_with_adjustments() -> None:
    rec = reconcile_actual(
        _bill(adjustments=1.5), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW,
    )
    assert rec.bucket_deltas["energy_charge_plus_adjustments"] == pytest.approx(49.6 - 49.5)


def test_actual_without_tax_line_is_labeled_tax_unknown() -> None:
    rec = reconcile_actual(
        _bill(taxes=None), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW,
    )
    assert rec.utility_basis == "tax_unknown"
    assert rec.utility_total_usd == pytest.approx(64.6)


def test_actual_with_a_hole_is_a_gap_not_a_number() -> None:
    rec = reconcile_actual(
        _bill(), ledger=_ledger_with(hourly(S, 48, kwh=10.0, skip=frozenset({30}))), usage_point_id="UP1", computed_at=NOW,
    )
    assert rec.reconcile_gap == "usage_incomplete"
    assert rec.orion_total_usd is None and rec.delta_usd is None
    assert rec.orion_covered_through == S + timedelta(hours=30)


def test_actual_with_no_usage() -> None:
    rec = reconcile_actual(_bill(), ledger=make_test_ledger(), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap == "no_usage"
    assert rec.orion_covered_through is None


def _forecast(**over) -> EnergyBillForecastV1:
    base = dict(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        as_of=S + timedelta(hours=72), projected_kwh=700.0, projected_total_usd=80.0, retrieved_at=S + timedelta(hours=72),
    )
    base.update(over)
    return EnergyBillForecastV1(**base)


def test_forecast_linear_run_rate_hand_oracle() -> None:
    # 72 kWh in 72 of 720 h -> 720 kWh: 400 @ .10 + 320 @ .12 = 78.40, + 10 = 88.40.
    rec = reconcile_forecast(_forecast(), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW)
    assert rec.orion_method == "linear_run_rate"
    assert rec.orion_kwh == pytest.approx(720.0)
    assert rec.orion_total_usd == pytest.approx(88.4)
    assert rec.delta_kwh == pytest.approx(20.0)
    assert rec.delta_usd == pytest.approx(8.4)
    assert rec.delta_pct == pytest.approx(8.4 / 80.0)
    assert rec.utility_basis == "tax_unknown"


def test_forecast_needs_a_day_of_usage() -> None:
    rec = reconcile_forecast(_forecast(), ledger=_ledger_with(hourly(S, 12)), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap == "usage_incomplete"


def test_forecast_without_end_uses_ledger_cycle() -> None:
    rec = reconcile_forecast(
        _forecast(billing_period_end=None), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW,
    )
    assert rec.billing_period_end == date(2026, 10, 1)
    assert rec.orion_total_usd == pytest.approx(88.4)


def test_forecast_kwh_only_leaves_usd_delta_unknown() -> None:
    rec = reconcile_forecast(
        _forecast(projected_total_usd=None), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW,
    )
    assert rec.delta_usd is None and rec.delta_pct is None
    assert rec.delta_kwh == pytest.approx(20.0)
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_ledger.py orion/energy/tests/test_energy_reconcile.py -q`
Expected: FAIL — `AttributeError: 'UsageLedger' object has no attribute 'window_prefix'` / `ModuleNotFoundError: orion.energy.reconcile`

- [ ] **Step 4: Ledger methods** — add to `UsageLedger` in `orion/energy/ledger.py` (after `cycle_coverage`):

```python
    def window_prefix(
        self, usage_point_id: str, start: datetime, end: datetime
    ) -> tuple[list[EnergyUsageIntervalV1], Optional[datetime]]:
        """Intervals contiguous from `start` inside [start, end), and where coverage stops."""
        prefix = self._contiguous_from_cycle_start(self._in_cycle(usage_point_id, start, end), start)
        return prefix, (prefix[-1].interval_end if prefix else None)

    def latest_interval_end(self, usage_point_id: str) -> Optional[datetime]:
        return max((iv.interval_end for iv in self._intervals.get(usage_point_id, {}).values()), default=None)
```

- [ ] **Step 5: Implement** `orion/energy/reconcile.py`:

```python
"""Reconcile Orion's tariff estimate against what Rocky Mountain Power billed or projects.

Bill periods come from RMP (meter-read dates), not from ENERGY_BILLING_CYCLE_START_DAY,
and blocks reset at the bill's own start. A period Orion cannot fully see is a gap,
never a partial number dressed up as a total.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import Tariff
from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyReconcileV1,
    EnergyUsageIntervalV1,
    ReconcileGap,
    UtilityBasis,
)

# A run rate from less than a day of data is mostly time-of-day noise.
MIN_RUN_RATE_HOURS = 24.0


def local_midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=tz)


def period_bounds(start_day: date, end_day: Optional[date], ledger: UsageLedger) -> tuple[datetime, datetime]:
    start = local_midnight(start_day, ledger.tz)
    end = local_midnight(end_day, ledger.tz) if end_day is not None else ledger.cycle_bounds(start)[1]
    return start, end


def price_intervals(
    tariff: Tariff, intervals: Iterable[EnergyUsageIntervalV1], *, tz: ZoneInfo
) -> tuple[float, float]:
    kwh = 0.0
    cost = 0.0
    for iv in intervals:
        month = iv.interval_start.astimezone(tz).month
        cost += tariff.energy_cost_usd(iv.energy_kwh, cycle_kwh_before=kwh, month=month)
        kwh += iv.energy_kwh
    return kwh, cost


@dataclass(frozen=True)
class PeriodProjection:
    observed_kwh: float
    observed_energy_usd: float
    projected_kwh: float
    projected_energy_usd: float
    fixed_usd: float
    covered_through: datetime

    @property
    def projected_total_usd(self) -> float:
        return self.projected_energy_usd + self.fixed_usd


def project_period(
    ledger: UsageLedger, usage_point_id: str, start: datetime, end: datetime
) -> tuple[Optional[PeriodProjection], Optional[ReconcileGap], Optional[datetime]]:
    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    if not prefix or covered is None:
        return None, "no_usage", None
    covered_sec = (covered - start).total_seconds()
    if covered_sec < MIN_RUN_RATE_HOURS * 3600.0:
        return None, "usage_incomplete", covered
    tariff, tz = ledger.tariff, ledger.tz
    kwh, energy = price_intervals(tariff, prefix, tz=tz)
    period_sec = (end - start).total_seconds()
    projected_kwh = kwh if covered >= end else kwh * period_sec / covered_sec
    last_month = (end - timedelta(seconds=1)).astimezone(tz).month
    remaining = max(0.0, projected_kwh - kwh)
    projected_energy = energy + tariff.energy_cost_usd(remaining, cycle_kwh_before=kwh, month=last_month)
    return (
        PeriodProjection(
            observed_kwh=kwh, observed_energy_usd=energy, projected_kwh=projected_kwh,
            projected_energy_usd=projected_energy, fixed_usd=tariff.fixed_monthly_usd, covered_through=covered,
        ),
        None,
        covered,
    )


def _pct(delta: Optional[float], base: Optional[float]) -> Optional[float]:
    if delta is None or base is None or base == 0.0:
        return None
    return delta / base


def _utility_pretax(bill: EnergyBillActualV1) -> tuple[float, UtilityBasis]:
    if bill.taxes is None:
        return bill.current_charges, "tax_unknown"
    return bill.current_charges - bill.taxes, "pre_tax"


def reconcile_actual(
    bill: EnergyBillActualV1, *, ledger: UsageLedger, usage_point_id: str, computed_at: datetime
) -> EnergyReconcileV1:
    tariff, tz = ledger.tariff, ledger.tz
    start, end = period_bounds(bill.billing_period_start, bill.billing_period_end, ledger)
    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    utility_total, basis = _utility_pretax(bill)
    common = dict(
        reconcile_kind="actual", usage_point_id=usage_point_id,
        billing_period_start=bill.billing_period_start, billing_period_end=bill.billing_period_end,
        utility_as_of=bill.retrieved_at, utility_kwh=bill.kwh_billed, utility_total_usd=utility_total,
        utility_basis=basis, orion_method="metered_period", orion_covered_through=covered,
        tariff_version=tariff.version, cost_basis=tariff.cost_basis, computed_at=computed_at,
    )
    if not prefix or covered is None:
        return EnergyReconcileV1(**common, reconcile_gap="no_usage")
    if covered < end:
        return EnergyReconcileV1(**common, reconcile_gap="usage_incomplete")
    kwh, energy = price_intervals(tariff, prefix, tz=tz)
    fixed = tariff.fixed_monthly_usd
    total = energy + fixed
    buckets: dict[str, float] = {}
    if bill.energy_charge is not None:
        buckets["energy_charge"] = energy - bill.energy_charge
        if bill.adjustments is not None:
            buckets["energy_charge_plus_adjustments"] = energy - (bill.energy_charge + bill.adjustments)
    if bill.customer_charge is not None:
        buckets["customer_charge"] = fixed - bill.customer_charge
    delta_usd = total - utility_total
    return EnergyReconcileV1(
        **common, orion_kwh=kwh, orion_energy_usd=energy, orion_fixed_usd=fixed, orion_total_usd=total,
        delta_kwh=kwh - bill.kwh_billed, delta_usd=delta_usd, delta_pct=_pct(delta_usd, utility_total),
        bucket_deltas=buckets,
    )


def reconcile_forecast(
    forecast: EnergyBillForecastV1, *, ledger: UsageLedger, usage_point_id: str, computed_at: datetime
) -> EnergyReconcileV1:
    tariff = ledger.tariff
    start, end = period_bounds(forecast.billing_period_start, forecast.billing_period_end, ledger)
    projection, gap, covered = project_period(ledger, usage_point_id, start, end)
    common = dict(
        reconcile_kind="forecast", usage_point_id=usage_point_id,
        billing_period_start=forecast.billing_period_start,
        billing_period_end=forecast.billing_period_end or end.astimezone(ledger.tz).date(),
        utility_as_of=forecast.as_of, utility_kwh=forecast.projected_kwh,
        utility_total_usd=forecast.projected_total_usd, utility_basis="tax_unknown",
        orion_method="linear_run_rate", orion_covered_through=covered,
        tariff_version=tariff.version, cost_basis=tariff.cost_basis, computed_at=computed_at,
    )
    if projection is None:
        return EnergyReconcileV1(**common, reconcile_gap=gap)
    total = projection.projected_total_usd
    delta_kwh = None if forecast.projected_kwh is None else projection.projected_kwh - forecast.projected_kwh
    delta_usd = None if forecast.projected_total_usd is None else total - forecast.projected_total_usd
    return EnergyReconcileV1(
        **common, orion_kwh=projection.projected_kwh, orion_energy_usd=projection.projected_energy_usd,
        orion_fixed_usd=projection.fixed_usd, orion_total_usd=total, delta_kwh=delta_kwh,
        delta_usd=delta_usd, delta_pct=_pct(delta_usd, forecast.projected_total_usd),
    )
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add orion/energy/testing.py orion/energy/reconcile.py orion/energy/ledger.py \
  orion/energy/tests/test_energy_reconcile.py orion/energy/tests/test_energy_ledger.py
git diff --cached --check
git commit -m "feat(energy): reconcile Orion estimate against RMP bill and forecast"
```

---

### Task 4: Importer status and stakes snapshot (pure)

**Files:**
- Create: `orion/energy/importer_status.py`, `orion/energy/stakes.py`
- Test: `orion/energy/tests/test_energy_importer_status.py`, `orion/energy/tests/test_energy_stakes.py`

**Interfaces:**
- Consumes: Task 3 `period_bounds`, `price_intervals`, `project_period`, `UsageLedger.window_prefix`; testing helpers.
- Produces:
  - `orion.energy.importer_status`: `PortalState = Literal["ok","reauth_required","error"]`, `PortalStatus(state, reason, last_attempt_at, last_success_at)` frozen dataclass, `parse_portal_status(raw: Mapping[str, Any]) -> PortalStatus` (raises `ValueError`), `portal_status_dict(status: PortalStatus) -> dict[str, Any]`, `compute_importer_status(*, portal_enabled: bool, portal: Optional[PortalStatus], portal_interval_hours: float, latest_interval_end: Optional[datetime], last_file_at: Optional[datetime], now: datetime, stale_after_hours: float) -> EnergyImporterStatusV1`
  - `orion.energy.stakes`: `current_forecast(forecasts: Iterable[EnergyBillForecastV1], *, now: datetime, tz: ZoneInfo) -> Optional[EnergyBillForecastV1]`, `build_stakes_snapshot(*, ledger, usage_point_id: Optional[str], forecast: Optional[EnergyBillForecastV1], importer: EnergyImporterStatusV1, now: datetime, near_ratio: float, over_ratio: float) -> EnergyStakesSnapshotV1`

- [ ] **Step 1: Write failing tests**

`orion/energy/tests/test_energy_importer_status.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.energy.importer_status import (
    PortalStatus,
    compute_importer_status,
    parse_portal_status,
    portal_status_dict,
)

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def _status(**over):
    base = dict(
        portal_enabled=False, portal=None, portal_interval_hours=24.0,
        latest_interval_end=NOW - timedelta(hours=30), last_file_at=NOW - timedelta(hours=2),
        now=NOW, stale_after_hours=48.0,
    )
    base.update(over)
    return compute_importer_status(**base)


def test_file_drop_fresh_usage_is_healthy() -> None:
    s = _status()
    assert (s.state, s.reason, s.source) == ("healthy", "usage_fresh", "file_drop")
    assert s.usage_lag_hours == pytest.approx(30.0)
    assert s.last_success_at == NOW - timedelta(hours=2)


def test_old_usage_is_stale() -> None:
    s = _status(latest_interval_end=NOW - timedelta(hours=50))
    assert s.state == "stale" and s.reason.startswith("usage_lag_hours=50.0")


def test_no_usage_is_stale_not_zero() -> None:
    s = _status(latest_interval_end=None)
    assert (s.state, s.reason, s.usage_lag_hours) == ("stale", "no_usage_yet", None)


def _portal(state="ok", attempt_hours_ago=1.0, reason=None):
    return PortalStatus(
        state=state, reason=reason or state,
        last_attempt_at=NOW - timedelta(hours=attempt_hours_ago), last_success_at=NOW - timedelta(hours=26),
    )


def test_reauth_beats_fresh_usage() -> None:
    s = _status(portal_enabled=True, portal=_portal("reauth_required", reason="session_expired"))
    assert (s.state, s.reason, s.source) == ("reauth_required", "session_expired", "portal")


def test_portal_missing_status_is_degraded() -> None:
    assert _status(portal_enabled=True, portal=None).reason == "portal_status_missing"


def test_portal_error_is_degraded_with_its_reason() -> None:
    s = _status(portal_enabled=True, portal=_portal("error", reason="empty_download"))
    assert (s.state, s.reason) == ("degraded", "empty_download")


def test_portal_not_running_is_degraded() -> None:
    s = _status(portal_enabled=True, portal=_portal("ok", attempt_hours_ago=72.0))
    assert (s.state, s.reason) == ("degraded", "portal_not_running")


def test_portal_ok_uses_portal_times() -> None:
    s = _status(portal_enabled=True, portal=_portal("ok"))
    assert s.state == "healthy"
    assert s.last_success_at == NOW - timedelta(hours=26)
    assert s.last_attempt_at == NOW - timedelta(hours=1)


def test_parse_portal_status_roundtrip_and_rejects_unknown() -> None:
    p = _portal("error", reason="timeout")
    assert parse_portal_status(portal_status_dict(p)) == p
    with pytest.raises(ValueError):
        parse_portal_status({"state": "fine", "last_attempt_at": NOW.isoformat()})
    with pytest.raises(ValueError):
        parse_portal_status({"state": "ok"})
```

`orion/energy/tests/test_energy_stakes.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from orion.energy.stakes import build_stakes_snapshot, current_forecast
from orion.energy.testing import UTC, hourly, make_test_ledger
from orion.schemas.energy import EnergyBillForecastV1, EnergyImporterStatusV1

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = S + timedelta(hours=72)


def _importer(state="healthy") -> EnergyImporterStatusV1:
    return EnergyImporterStatusV1(state=state, reason="x", source="file_drop", as_of=NOW)


def _forecast(total=80.0, **over) -> EnergyBillForecastV1:
    base = dict(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        as_of=NOW, projected_total_usd=total, retrieved_at=NOW,
    )
    base.update(over)
    return EnergyBillForecastV1(**base)


def _snap(forecast, importer="healthy", point="UP1"):
    led = make_test_ledger()
    for iv in hourly(S, 72):
        led.upsert(iv)
    return build_stakes_snapshot(
        ledger=led, usage_point_id=point, forecast=forecast, importer=_importer(importer),
        now=NOW, near_ratio=1.0, over_ratio=1.10,
    )


def test_over_forecast_hand_oracle() -> None:
    # Projected 88.40 (see reconcile oracle) / 80.00 = 1.105 >= 1.10.
    s = _snap(_forecast(80.0))
    assert s.pressure == "over_forecast"
    assert s.projected_to_forecast_ratio == pytest.approx(1.105)
    assert s.orion_projected_total_usd == pytest.approx(88.4)
    assert s.cycle_accumulated_kwh == pytest.approx(72.0)
    assert s.cycle_to_date_total_usd == pytest.approx(17.2)
    assert s.marginal_usd_per_kwh == pytest.approx(0.10)
    assert (s.cycle_start, s.cycle_end) == (date(2026, 9, 1), date(2026, 10, 1))
    assert s.covered_through == NOW


@pytest.mark.parametrize("total,pressure", [(85.0, "near_forecast"), (90.0, "normal")])
def test_near_and_normal(total, pressure) -> None:
    assert _snap(_forecast(total)).pressure == pressure


def test_unhealthy_importer_is_unknown_but_keeps_numbers() -> None:
    s = _snap(_forecast(80.0), importer="stale")
    assert (s.pressure, s.pressure_reason) == ("unknown", "importer_stale")
    assert s.projected_to_forecast_ratio is None
    assert s.cycle_to_date_total_usd == pytest.approx(17.2)


def test_no_forecast_is_unknown() -> None:
    s = _snap(None)
    assert (s.pressure, s.pressure_reason) == ("unknown", "no_forecast_total")
    assert s.orion_projected_total_usd == pytest.approx(88.4)
    assert (s.cycle_start, s.cycle_end) == (date(2026, 9, 1), date(2026, 10, 1))


def test_kwh_only_forecast_is_unknown() -> None:
    s = _snap(_forecast(None, projected_kwh=700.0))
    assert s.pressure_reason == "no_forecast_total"


def test_no_usage_point_is_unknown() -> None:
    s = _snap(_forecast(80.0), point=None)
    assert (s.pressure, s.pressure_reason) == ("unknown", "no_usage_point")


def test_current_forecast_picks_latest_live_period() -> None:
    old = _forecast(70.0, billing_period_start=date(2026, 8, 1), billing_period_end=date(2026, 9, 1))
    early = _forecast(75.0, as_of=NOW - timedelta(days=1), retrieved_at=NOW - timedelta(days=1))
    late = _forecast(80.0)
    assert current_forecast([old, early, late], now=NOW, tz=UTC) is late
    assert current_forecast([old], now=NOW, tz=UTC) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests/test_energy_importer_status.py orion/energy/tests/test_energy_stakes.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** `orion/energy/importer_status.py`:

```python
"""Is house usage actually arriving?

Precedence: reauth_required > degraded > stale > healthy. The portal fetcher only
reports what it tried; freshness is judged from the usage the ledger really holds,
so a portal that says "ok" but delivers nothing still reads stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from orion.schemas.energy import EnergyImporterStatusV1

PortalState = Literal["ok", "reauth_required", "error"]
_PORTAL_STATES = ("ok", "reauth_required", "error")


@dataclass(frozen=True)
class PortalStatus:
    state: PortalState
    reason: str
    last_attempt_at: datetime
    last_success_at: Optional[datetime]


def _ts(raw: Any) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def parse_portal_status(raw: Mapping[str, Any]) -> PortalStatus:
    state = raw.get("state")
    if state not in _PORTAL_STATES:
        raise ValueError(f"unknown portal state {state!r}")
    attempt = _ts(raw.get("last_attempt_at"))
    if attempt is None:
        raise ValueError("portal status missing last_attempt_at")
    return PortalStatus(
        state=state, reason=str(raw.get("reason") or state),
        last_attempt_at=attempt, last_success_at=_ts(raw.get("last_success_at")),
    )


def portal_status_dict(status: PortalStatus) -> dict[str, Any]:
    return {
        "state": status.state,
        "reason": status.reason,
        "last_attempt_at": status.last_attempt_at.isoformat(),
        "last_success_at": status.last_success_at.isoformat() if status.last_success_at else None,
    }


def compute_importer_status(
    *,
    portal_enabled: bool,
    portal: Optional[PortalStatus],
    portal_interval_hours: float,
    latest_interval_end: Optional[datetime],
    last_file_at: Optional[datetime],
    now: datetime,
    stale_after_hours: float,
) -> EnergyImporterStatusV1:
    lag = None if latest_interval_end is None else max(0.0, (now - latest_interval_end).total_seconds() / 3600.0)
    state: Optional[str] = None
    reason: Optional[str] = None
    if portal_enabled:
        if portal is None:
            state, reason = "degraded", "portal_status_missing"
        elif portal.state == "reauth_required":
            state, reason = "reauth_required", portal.reason
        elif portal.state == "error":
            state, reason = "degraded", portal.reason
        elif (now - portal.last_attempt_at).total_seconds() > 2 * portal_interval_hours * 3600.0:
            state, reason = "degraded", "portal_not_running"
    if state is None:
        if lag is None:
            state, reason = "stale", "no_usage_yet"
        elif lag > stale_after_hours:
            state, reason = "stale", f"usage_lag_hours={lag:.1f}"
        else:
            state, reason = "healthy", "usage_fresh"
    if portal_enabled:
        success = portal.last_success_at if portal else None
        attempt = portal.last_attempt_at if portal else None
    else:
        success = attempt = last_file_at
    return EnergyImporterStatusV1(
        state=state, reason=reason, source="portal" if portal_enabled else "file_drop",
        last_success_at=success, last_attempt_at=attempt,
        latest_interval_end=latest_interval_end, usage_lag_hours=lag, as_of=now,
    )
```

`orion/energy/stakes.py`:

```python
"""Stakes snapshot: what the house bill looks like right now, for spend gates.

Compares Orion's pre-tax run-rate projection with RMP's forecast (which may include
tax), so the ratio leans low -- a gate reading it holds less often, not more.
Any missing or unhealthy input makes pressure `unknown`, and unknown never holds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import period_bounds, price_intervals, project_period
from orion.schemas.energy import EnergyBillForecastV1, EnergyImporterStatusV1, EnergyStakesSnapshotV1


def current_forecast(
    forecasts: Iterable[EnergyBillForecastV1], *, now: datetime, tz: ZoneInfo
) -> Optional[EnergyBillForecastV1]:
    today = now.astimezone(tz).date()
    live = [
        f for f in forecasts
        if f.billing_period_start <= today and (f.billing_period_end is None or today < f.billing_period_end)
    ]
    return max(live, key=lambda f: (f.billing_period_start, f.as_of), default=None)


def build_stakes_snapshot(
    *,
    ledger: UsageLedger,
    usage_point_id: Optional[str],
    forecast: Optional[EnergyBillForecastV1],
    importer: EnergyImporterStatusV1,
    now: datetime,
    near_ratio: float,
    over_ratio: float,
) -> EnergyStakesSnapshotV1:
    tz, tariff = ledger.tz, ledger.tariff
    snap: dict[str, Any] = dict(
        as_of=now, usage_point_id=usage_point_id, importer_state=importer.state, tariff_version=tariff.version,
    )
    if forecast is not None:
        snap.update(forecast_total_usd=forecast.projected_total_usd, forecast_as_of=forecast.as_of)
    if usage_point_id is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="no_usage_point")

    if forecast is not None:
        start, end = period_bounds(forecast.billing_period_start, forecast.billing_period_end, ledger)
    else:
        start, end = ledger.cycle_bounds(now)
    snap.update(cycle_start=start.astimezone(tz).date(), cycle_end=end.astimezone(tz).date())

    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    if prefix and covered is not None:
        kwh, energy = price_intervals(tariff, prefix, tz=tz)
        month = (covered - timedelta(seconds=1)).astimezone(tz).month
        snap.update(
            covered_through=covered, cycle_accumulated_kwh=kwh,
            cycle_to_date_total_usd=energy + tariff.fixed_monthly_usd,
            marginal_usd_per_kwh=tariff.marginal_usd_per_kwh(cycle_kwh=kwh, month=month),
        )
    projection, gap, _ = project_period(ledger, usage_point_id, start, end)
    if projection is not None:
        snap["orion_projected_total_usd"] = projection.projected_total_usd

    if importer.state != "healthy":
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason=f"importer_{importer.state}")
    if projection is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason=f"projection_{gap}")
    if forecast is None or forecast.projected_total_usd is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="no_forecast_total")
    if forecast.projected_total_usd <= 0.0:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="forecast_nonpositive")

    ratio = projection.projected_total_usd / forecast.projected_total_usd
    pressure = "over_forecast" if ratio >= over_ratio else "near_forecast" if ratio >= near_ratio else "normal"
    return EnergyStakesSnapshotV1(
        **snap, projected_to_forecast_ratio=ratio, pressure=pressure, pressure_reason=f"ratio={ratio:.3f}",
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/energy/tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add orion/energy/importer_status.py orion/energy/stakes.py \
  orion/energy/tests/test_energy_importer_status.py orion/energy/tests/test_energy_stakes.py
git diff --cached --check
git commit -m "feat(energy): importer status state machine and stakes snapshot"
```

---

### Task 5: orion-energy service — bill drop, reconcile, status tick

**Files:**
- Modify: `services/orion-energy/app/inbox.py`, `app/pipeline.py`, `app/main.py`, `app/settings.py`, `.env_example`, `README.md`
- Create: `services/orion-energy/app/bills.py`, `app/portal_status.py`
- Test: `services/orion-energy/tests/test_energy_inbox.py`, `tests/test_energy_bills.py`, `tests/test_energy_pipeline_bills.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `app.inbox`: `STAMP_FORMAT`, `SEP`, `PORTAL_PREFIX = "rmp-portal"`, `scan_dir(inbox_dir, processed_dir, *, now, suffix, parse) -> list`, `replay_dir(processed_dir, *, suffix, parse) -> list`, `latest_processed_at(processed_dir: Path, suffix: str) -> Optional[datetime]`, existing `scan_inbox` / `load_processed` (same signatures; portal-named files get `source="rockymountain_power"`).
  - `app.bills`: `Bill = Union[EnergyBillActualV1, EnergyBillForecastV1]`, `parse_bill(raw: bytes, *, retrieved_at: datetime, source_file: str) -> Bill`, `scan_bills(inbox_dir, processed_dir, *, now) -> list[Bill]`, `load_processed_bills(processed_dir) -> list[Bill]`.
  - `app.portal_status`: `read_portal_status(path: Path) -> Optional[PortalStatus]`.
  - `app.pipeline`: `EnergyChannels` gains defaulted fields `bill_actual`, `bill_forecast`, `reconcile`, `stakes`, `importer_status`; new `StakesConfig(near_ratio=1.0, over_ratio=1.10, stale_after_hours=48.0, portal_enabled=False, portal_interval_hours=24.0)`; `EnergyPipeline(..., stakes: StakesConfig = StakesConfig())`; methods `replay_bills(bills)`, `ingest_bills(bills, *, now) -> list[Outbound]`, `status_tick(*, now, portal, last_file_at) -> list[Outbound]` (always `[importer_status, stakes]`).
  - Portal (Task 6) writes files named `rmp-portal-*.xml` into `ENERGY_INBOX_DIR`, `rmp-portal-*.json` into `ENERGY_BILL_INBOX_DIR`, and `status.json` at `ENERGY_PORTAL_STATUS_PATH`.

- [ ] **Step 1: Write failing tests**

Append to `services/orion-energy/tests/test_energy_inbox.py` (reuse that file's existing fixture-copy pattern for the ESPI XML; the fixture is `orion/energy/tests/fixtures/espi_two_flows.xml`):

```python
from app.inbox import latest_processed_at

_FIXTURE = Path(__file__).resolve().parents[3] / "orion/energy/tests/fixtures/espi_two_flows.xml"


def test_portal_named_file_is_labeled_rockymountain_power(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    inbox.mkdir()
    (inbox / "rmp-portal-20260927T060000Z.xml").write_bytes(_FIXTURE.read_bytes())
    (inbox / "manual.xml").write_bytes(_FIXTURE.read_bytes())
    now = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
    rows = scan_inbox(inbox, processed, now=now)
    sources = {r.source_file.split("__", 1)[1]: r.source for r in rows}
    assert sources["rmp-portal-20260927T060000Z.xml"] == "rockymountain_power"
    assert sources["manual.xml"] == "file_drop"
    replayed = {r.source for r in load_processed(processed)}
    assert replayed == {"rockymountain_power", "file_drop"}
    assert latest_processed_at(processed, ".xml") == now


def test_latest_processed_at_empty(tmp_path) -> None:
    assert latest_processed_at(tmp_path / "missing", ".xml") is None
```

(Add `from datetime import datetime, timezone`, `from pathlib import Path`, and `load_processed`/`scan_inbox` to the imports if the file doesn't already have them.)

`services/orion-energy/tests/test_energy_bills.py`:

```python
from __future__ import annotations

import json
from datetime import date, datetime, timezone

from app.bills import load_processed_bills, scan_bills
from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)

ACTUAL = {
    "kind": "energy.bill.actual.v1",
    "billing_period_start": "2026-08-12",
    "billing_period_end": "2026-09-11",
    "kwh_billed": 712,
    "current_charges": 101.23,
    "taxes": 4.10,
}


def _drop(inbox, name, obj) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_hand_entered_bill_is_parsed_stamped_and_replayed(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    _drop(inbox, "aug.json", ACTUAL)
    [bill] = scan_bills(inbox, processed, now=NOW)
    assert isinstance(bill, EnergyBillActualV1)
    assert bill.source == "file_drop"
    assert bill.retrieved_at == NOW
    assert bill.billing_period_start == date(2026, 8, 12)
    assert bill.energy_charge is None
    assert not list(inbox.glob("*.json"))
    [again] = load_processed_bills(processed)
    assert again == bill


def test_portal_bill_keeps_its_own_source_and_time(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    got = "2026-09-26T05:00:00+00:00"
    _drop(inbox, "rmp-portal-x-00.json", {
        "kind": "energy.bill.forecast.v1", "source": "rockymountain_power",
        "billing_period_start": "2026-09-11", "as_of": got, "projected_total_usd": 96.0, "retrieved_at": got,
    })
    [fc] = scan_bills(inbox, processed, now=NOW)
    assert isinstance(fc, EnergyBillForecastV1)
    assert fc.source == "rockymountain_power"
    assert fc.retrieved_at == datetime(2026, 9, 26, 5, tzinfo=timezone.utc)


def test_bad_bills_go_to_failed_and_publish_nothing(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    _drop(inbox, "nokind.json", {k: v for k, v in ACTUAL.items() if k != "kind"})
    _drop(inbox, "empty_forecast.json", {"kind": "energy.bill.forecast.v1", "billing_period_start": "2026-09-11", "as_of": "2026-09-26T05:00:00Z"})
    _drop(inbox, "garbage.json", "{not json")
    assert scan_bills(inbox, processed, now=NOW) == []
    assert sorted(p.name for p in (inbox / "failed").iterdir()) == ["empty_forecast.json", "garbage.json", "nokind.json"]
```

`services/orion-energy/tests/test_energy_pipeline_bills.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline, StakesConfig
from orion.energy.importer_status import PortalStatus
from orion.energy.testing import hourly, make_test_ledger
from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_IMPORTER_STATUS_KIND,
    ENERGY_RECONCILE_KIND,
    ENERGY_STAKES_KIND,
    ENERGY_USAGE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _pipe(**kw) -> EnergyPipeline:
    return EnergyPipeline(ledger=make_test_ledger(), channels=EnergyChannels("u", "a", "r"), usage_point_id="UP1", **kw)


def _bill(retrieved=NOW) -> EnergyBillActualV1:
    return EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 9, 3),
        kwh_billed=470.0, taxes=3.0, current_charges=64.6, retrieved_at=retrieved,
    )


def test_bill_after_usage_publishes_bill_and_priced_reconcile() -> None:
    p = _pipe()
    p.ingest_intervals(hourly(S, 48, kwh=10.0), now=NOW)
    out = p.ingest_bills([_bill()], now=NOW)
    assert [o.kind for o in out] == [ENERGY_BILL_ACTUAL_KIND, ENERGY_RECONCILE_KIND]
    assert out[0].channel == "orion:energy:bill:actual"
    assert out[1].channel == "orion:energy:reconcile"
    assert out[1].payload.orion_total_usd == pytest.approx(59.6)


def test_bill_before_usage_reconciles_again_when_usage_lands() -> None:
    p = _pipe()
    first = p.ingest_bills([_bill()], now=NOW)
    assert first[1].payload.reconcile_gap == "no_usage"
    out = p.ingest_intervals(hourly(S, 48, kwh=10.0), now=NOW)
    recs = [o.payload for o in out if o.kind == ENERGY_RECONCILE_KIND]
    assert len(recs) == 1 and recs[0].orion_total_usd == pytest.approx(59.6)


def test_stale_bill_redelivery_is_ignored() -> None:
    p = _pipe()
    p.ingest_bills([_bill()], now=NOW)
    assert p.ingest_bills([_bill(retrieved=NOW - timedelta(days=1))], now=NOW) == []


def test_status_tick_with_no_data_is_stale_and_emits_no_usage() -> None:
    p = EnergyPipeline(ledger=make_test_ledger(), channels=EnergyChannels("u", "a", "r"))
    out = p.status_tick(now=NOW, portal=None, last_file_at=None)
    assert [o.kind for o in out] == [ENERGY_IMPORTER_STATUS_KIND, ENERGY_STAKES_KIND]
    assert ENERGY_USAGE_KIND not in {o.kind for o in out}
    assert out[0].payload.state == "stale" and out[0].payload.reason == "no_usage_yet"
    assert out[1].payload.pressure == "unknown"


def test_status_tick_over_forecast() -> None:
    p = _pipe()
    p.ingest_intervals(hourly(S, 72), now=NOW)
    p.replay_bills([
        EnergyBillForecastV1(
            source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
            as_of=NOW, projected_total_usd=80.0, retrieved_at=NOW,
        )
    ])
    out = p.status_tick(now=NOW, portal=None, last_file_at=NOW)
    assert out[0].payload.state == "healthy"
    assert out[1].payload.pressure == "over_forecast"
    assert out[1].channel == "orion:energy:stakes:snapshot"


def test_status_tick_portal_reauth() -> None:
    p = _pipe(stakes=StakesConfig(portal_enabled=True))
    p.ingest_intervals(hourly(S, 72), now=NOW)
    portal = PortalStatus(state="reauth_required", reason="session_expired", last_attempt_at=NOW, last_success_at=None)
    out = p.status_tick(now=NOW, portal=portal, last_file_at=None)
    assert out[0].payload.state == "reauth_required"
    assert out[1].payload.pressure_reason == "importer_reauth_required"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/tests -q`
Expected: FAIL — `ImportError` for `latest_processed_at`, `app.bills`, `StakesConfig`.

- [ ] **Step 3: Generic drop directory** — replace `services/orion-energy/app/inbox.py` with:

```python
"""Stamped drop directories (Green Button XML, bill JSON).

A parsed file moves to processed/ with its retrieval time as a filename prefix, so
a restart replays the exact same rows with the exact same retrieved_at. A file that
fails to parse moves to <inbox>/failed/ and publishes nothing -- a broken export must
never read as a quiet day. Files the portal fetcher wrote start with `rmp-portal`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

from orion.energy.espi import parse_espi
from orion.schemas.energy import EnergySource, EnergyUsageIntervalV1

logger = logging.getLogger("orion-energy.inbox")
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
SEP = "__"
PORTAL_PREFIX = "rmp-portal"

T = TypeVar("T")
Parse = Callable[[bytes, datetime, str], list[T]]


def _source_for(stamped_name: str) -> EnergySource:
    original = stamped_name.partition(SEP)[2] or stamped_name
    return "rockymountain_power" if original.startswith(PORTAL_PREFIX) else "file_drop"


def _parse_xml(raw: bytes, retrieved_at: datetime, stamped_name: str) -> list[EnergyUsageIntervalV1]:
    return parse_espi(raw, retrieved_at=retrieved_at, source=_source_for(stamped_name), source_file=stamped_name)


def scan_dir(inbox_dir: Path, processed_dir: Path, *, now: datetime, suffix: str, parse: Parse) -> list:
    if not inbox_dir.is_dir():
        return []
    stamp_time = now.astimezone(timezone.utc).replace(microsecond=0)
    rows: list = []
    for path in sorted(p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix):
        target_name = f"{stamp_time.strftime(STAMP_FORMAT)}{SEP}{path.name}"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=read error=%s", path.name, exc)
            continue
        try:
            parsed = parse(raw, stamp_time, target_name)
        except ValueError as exc:
            failed = inbox_dir / "failed"
            failed.mkdir(parents=True, exist_ok=True)
            try:
                path.rename(failed / path.name)
            except OSError as rename_exc:
                logger.warning("energy_inbox_io_failed file=%s op=rename_to_failed error=%s", path.name, rename_exc)
                continue
            logger.warning("energy_inbox_parse_failed file=%s error=%s", path.name, exc)
            continue
        processed_dir.mkdir(parents=True, exist_ok=True)
        try:
            path.rename(processed_dir / target_name)
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=rename_to_processed error=%s", path.name, exc)
            continue
        logger.info("energy_inbox_parsed file=%s rows=%d", path.name, len(parsed))
        rows.extend(parsed)
    return rows


def _stamp_of(path: Path) -> Optional[datetime]:
    stamp, sep, _ = path.name.partition(SEP)
    if not sep:
        return None
    try:
        return datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def replay_dir(processed_dir: Path, *, suffix: str, parse: Parse) -> list:
    if not processed_dir.is_dir():
        return []
    rows: list = []
    for path in sorted(processed_dir.glob(f"*{suffix}")):
        retrieved = _stamp_of(path)
        if retrieved is None:
            logger.warning("energy_replay_skipped_unstamped file=%s", path.name)
            continue
        try:
            rows.extend(parse(path.read_bytes(), retrieved, path.name))
        except ValueError as exc:
            logger.warning("energy_replay_parse_failed file=%s error=%s", path.name, exc)
    return rows


def latest_processed_at(processed_dir: Path, suffix: str) -> Optional[datetime]:
    if not processed_dir.is_dir():
        return None
    stamps = [s for s in (_stamp_of(p) for p in processed_dir.glob(f"*{suffix}")) if s is not None]
    return max(stamps, default=None)


def scan_inbox(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[EnergyUsageIntervalV1]:
    return scan_dir(inbox_dir, processed_dir, now=now, suffix=".xml", parse=_parse_xml)


def load_processed(processed_dir: Path) -> list[EnergyUsageIntervalV1]:
    return replay_dir(processed_dir, suffix=".xml", parse=_parse_xml)
```

(`EspiError` subclasses `ValueError`; confirm with `rg -n "class EspiError" orion/energy/espi.py`. If any existing inbox test asserts the old log key `intervals=`, update it to `rows=`.)

- [ ] **Step 4: Bills** — create `services/orion-energy/app/bills.py`:

```python
"""Bill drop directory: one JSON object per file.

Same contract whether Juniper types a bill in or the portal fetcher writes it:
`kind` is energy.bill.actual.v1 or energy.bill.forecast.v1, the other keys are that
schema's fields. `source` defaults to file_drop and `retrieved_at` to the drop time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Union

from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

from .inbox import replay_dir, scan_dir

Bill = Union[EnergyBillActualV1, EnergyBillForecastV1]
_MODELS = {ENERGY_BILL_ACTUAL_KIND: EnergyBillActualV1, ENERGY_BILL_FORECAST_KIND: EnergyBillForecastV1}


def parse_bill(raw: bytes, *, retrieved_at: datetime, source_file: str) -> Bill:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("bill file must be a JSON object")
    model = _MODELS.get(data.pop("kind", None))
    if model is None:
        raise ValueError("bill file needs kind energy.bill.actual.v1 or energy.bill.forecast.v1")
    data.setdefault("source", "file_drop")
    data.setdefault("retrieved_at", retrieved_at.isoformat())
    data["source_file"] = source_file
    return model.model_validate(data)


def _parse(raw: bytes, retrieved_at: datetime, name: str) -> list[Bill]:
    return [parse_bill(raw, retrieved_at=retrieved_at, source_file=name)]


def scan_bills(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[Bill]:
    return scan_dir(inbox_dir, processed_dir, now=now, suffix=".json", parse=_parse)


def load_processed_bills(processed_dir: Path) -> list[Bill]:
    return replay_dir(processed_dir, suffix=".json", parse=_parse)
```

(`json.JSONDecodeError` and pydantic `ValidationError` both subclass `ValueError`, so `scan_dir` routes them to `failed/`.)

Create `services/orion-energy/app/portal_status.py`:

```python
"""Read the status.json the portal fetcher writes. Missing or unreadable -> None."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from orion.energy.importer_status import PortalStatus, parse_portal_status

logger = logging.getLogger("orion-energy.portal_status")


def read_portal_status(path: Path) -> Optional[PortalStatus]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("energy_portal_status_unreadable path=%s error=%s", path, exc)
        return None
    try:
        return parse_portal_status(raw)
    except ValueError as exc:
        logger.warning("energy_portal_status_invalid path=%s error=%s", path, exc)
        return None
```

- [ ] **Step 5: Pipeline** — in `services/orion-energy/app/pipeline.py`:

Replace the imports block's energy/schema imports with:

```python
from orion.energy.importer_status import PortalStatus, compute_importer_status
from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import period_bounds, reconcile_actual, reconcile_forecast
from orion.energy.run_cost import estimate_run_cost
from orion.energy.stakes import build_stakes_snapshot, current_forecast
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    ENERGY_IMPORTER_STATUS_KIND,
    ENERGY_RECONCILE_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_STAKES_KIND,
    ENERGY_USAGE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1
```

and add `from datetime import date` plus `Union` to the typing import. Replace `EnergyChannels` and add `StakesConfig`:

```python
@dataclass(frozen=True)
class EnergyChannels:
    usage: str
    accrued: str
    run_cost: str
    bill_actual: str = "orion:energy:bill:actual"
    bill_forecast: str = "orion:energy:bill:forecast"
    reconcile: str = "orion:energy:reconcile"
    stakes: str = "orion:energy:stakes:snapshot"
    importer_status: str = "orion:energy:importer:status"


@dataclass(frozen=True)
class StakesConfig:
    near_ratio: float = 1.0
    over_ratio: float = 1.10
    stale_after_hours: float = 48.0
    portal_enabled: bool = False
    portal_interval_hours: float = 24.0


Bill = Union[EnergyBillActualV1, EnergyBillForecastV1]
```

In `EnergyPipeline.__init__` add parameter `stakes: StakesConfig = StakesConfig(),` (after `pending_hours`) and state:

```python
        self._stakes = stakes
        # Bills keyed by natural key; a newer retrieval of the same bill wins.
        self._actuals: dict[tuple[date, date], EnergyBillActualV1] = {}
        self._forecasts: dict[tuple[date, datetime], EnergyBillForecastV1] = {}
```

Replace `ingest_intervals` with:

```python
    def ingest_intervals(self, intervals: Iterable[EnergyUsageIntervalV1], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        affected: set[tuple[str, datetime]] = set()
        lo: Optional[datetime] = None
        hi: Optional[datetime] = None
        for iv in intervals:
            if self._ledger.upsert(iv):
                out.append(Outbound(self._channels.usage, ENERGY_USAGE_KIND, iv))
                affected.add((iv.usage_point_id, self._ledger.cycle_bounds(iv.interval_start)[0]))
                lo = iv.interval_start if lo is None else min(lo, iv.interval_start)
                hi = iv.interval_end if hi is None else max(hi, iv.interval_end)
        for point, cycle_start in sorted(affected):
            for accrued in self._accrue_cycle(point, cycle_start, computed_at=now):
                out.append(Outbound(self._channels.accrued, ENERGY_ACCRUED_KIND, accrued))
        if affected and lo is not None and hi is not None:
            out.extend(self._reprice_pending(now=now))
            out.extend(self._rereconcile(lo, hi, now=now))
        return out
```

Add methods:

```python
    def _store_bill(self, bill: Bill) -> bool:
        if isinstance(bill, EnergyBillActualV1):
            key = (bill.billing_period_start, bill.billing_period_end)
            old = self._actuals.get(key)
            if old is not None and old.retrieved_at > bill.retrieved_at:
                return False
            self._actuals[key] = bill
            return True
        fkey = (bill.billing_period_start, bill.as_of)
        old_fc = self._forecasts.get(fkey)
        if old_fc is not None and old_fc.retrieved_at > bill.retrieved_at:
            return False
        self._forecasts[fkey] = bill
        return True

    def replay_bills(self, bills: Iterable[Bill]) -> None:
        for bill in bills:
            self._store_bill(bill)

    def ingest_bills(self, bills: Iterable[Bill], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for bill in bills:
            if not self._store_bill(bill):
                continue
            if isinstance(bill, EnergyBillActualV1):
                out.append(Outbound(self._channels.bill_actual, ENERGY_BILL_ACTUAL_KIND, bill))
            else:
                out.append(Outbound(self._channels.bill_forecast, ENERGY_BILL_FORECAST_KIND, bill))
            rec = self._reconcile(bill, now=now)
            if rec is not None:
                out.append(rec)
        return out

    def _reconcile(self, bill: Bill, *, now: datetime) -> Optional[Outbound]:
        point = bill.usage_point_id or self.usage_point()
        if point is None:
            logger.warning("energy_reconcile_skipped reason=no_usage_point period_start=%s", bill.billing_period_start)
            return None
        fn = reconcile_actual if isinstance(bill, EnergyBillActualV1) else reconcile_forecast
        rec = fn(bill, ledger=self._ledger, usage_point_id=point, computed_at=now)
        return Outbound(self._channels.reconcile, ENERGY_RECONCILE_KIND, rec)

    def _latest_forecasts(self) -> list[EnergyBillForecastV1]:
        latest: dict[date, EnergyBillForecastV1] = {}
        for fc in self._forecasts.values():
            cur = latest.get(fc.billing_period_start)
            if cur is None or fc.as_of > cur.as_of:
                latest[fc.billing_period_start] = fc
        return list(latest.values())

    def _rereconcile(self, lo: datetime, hi: datetime, *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for bill in [*self._actuals.values(), *self._latest_forecasts()]:
            start, end = period_bounds(bill.billing_period_start, bill.billing_period_end, self._ledger)
            if start < hi and lo < end:
                rec = self._reconcile(bill, now=now)
                if rec is not None:
                    out.append(rec)
        return out

    def status_tick(
        self, *, now: datetime, portal: Optional[PortalStatus], last_file_at: Optional[datetime]
    ) -> list[Outbound]:
        point = self.usage_point()
        cfg = self._stakes
        importer = compute_importer_status(
            portal_enabled=cfg.portal_enabled, portal=portal, portal_interval_hours=cfg.portal_interval_hours,
            latest_interval_end=None if point is None else self._ledger.latest_interval_end(point),
            last_file_at=last_file_at, now=now, stale_after_hours=cfg.stale_after_hours,
        )
        snapshot = build_stakes_snapshot(
            ledger=self._ledger, usage_point_id=point,
            forecast=current_forecast(self._forecasts.values(), now=now, tz=self._ledger.tz),
            importer=importer, now=now, near_ratio=cfg.near_ratio, over_ratio=cfg.over_ratio,
        )
        return [
            Outbound(self._channels.importer_status, ENERGY_IMPORTER_STATUS_KIND, importer),
            Outbound(self._channels.stakes, ENERGY_STAKES_KIND, snapshot),
        ]
```

- [ ] **Step 6: Settings + env** — add to `Settings` in `services/orion-energy/app/settings.py` (after `ENERGY_PUBLISH_MAX_PER_SEC`):

```python
    ENERGY_BILL_INBOX_DIR: str = Field(default="/data/energy/bills/inbox")
    ENERGY_BILL_PROCESSED_DIR: str = Field(default="/data/energy/bills/processed")
    ENERGY_STATUS_INTERVAL_SEC: float = Field(default=300.0, gt=0)
    # ~24h utility lag is normal; past this the importer reads stale.
    ENERGY_STALE_AFTER_HOURS: float = Field(default=48.0, gt=0)
    ENERGY_PORTAL_ENABLED: bool = Field(default=False)
    ENERGY_PORTAL_STATUS_PATH: str = Field(default="/data/energy/portal/status.json")
    ENERGY_PORTAL_INTERVAL_HOURS: float = Field(default=24.0, gt=0)
    ENERGY_STAKES_NEAR_RATIO: float = Field(default=1.0, gt=0)
    ENERGY_STAKES_OVER_RATIO: float = Field(default=1.10, gt=0)
```

and next to the channel settings:

```python
    ENERGY_BILL_ACTUAL_CHANNEL: str = Field(default="orion:energy:bill:actual")
    ENERGY_BILL_FORECAST_CHANNEL: str = Field(default="orion:energy:bill:forecast")
    ENERGY_RECONCILE_CHANNEL: str = Field(default="orion:energy:reconcile")
    ENERGY_STAKES_CHANNEL: str = Field(default="orion:energy:stakes:snapshot")
    ENERGY_IMPORTER_STATUS_CHANNEL: str = Field(default="orion:energy:importer:status")
```

In `services/orion-energy/.env_example` after `ENERGY_PUBLISH_MAX_PER_SEC=200`:

```text
# Drop bill JSON here (see README "Bills"); the portal fetcher writes here too.
ENERGY_BILL_INBOX_DIR=/data/energy/bills/inbox
ENERGY_BILL_PROCESSED_DIR=/data/energy/bills/processed
ENERGY_STATUS_INTERVAL_SEC=300
# Utility data lags ~24h; older than this reads stale (never as $0 usage).
ENERGY_STALE_AFTER_HOURS=48
# true once the orion-energy-portal compose profile is running.
ENERGY_PORTAL_ENABLED=false
ENERGY_PORTAL_STATUS_PATH=/data/energy/portal/status.json
ENERGY_PORTAL_INTERVAL_HOURS=24
# Projected cycle total / RMP forecast total at which pressure reads near / over.
ENERGY_STAKES_NEAR_RATIO=1.0
ENERGY_STAKES_OVER_RATIO=1.10
```

and after `ENERGY_RUN_COST_CHANNEL=...`:

```text
ENERGY_BILL_ACTUAL_CHANNEL=orion:energy:bill:actual
ENERGY_BILL_FORECAST_CHANNEL=orion:energy:bill:forecast
ENERGY_RECONCILE_CHANNEL=orion:energy:reconcile
ENERGY_STAKES_CHANNEL=orion:energy:stakes:snapshot
ENERGY_IMPORTER_STATUS_CHANNEL=orion:energy:importer:status
```

Then run `python scripts/sync_local_env_from_example.py` from the worktree root.

- [ ] **Step 7: main.py** — in `services/orion-energy/app/main.py`:

Imports: replace `from .inbox import load_processed, scan_inbox` and `from .pipeline import ...` with:

```python
from .bills import load_processed_bills, scan_bills
from .inbox import latest_processed_at, load_processed, scan_inbox
from .pipeline import EnergyChannels, EnergyPipeline, Outbound, StakesConfig
from .portal_status import read_portal_status
```

`build_pipeline`:

```python
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
            bill_actual=settings.ENERGY_BILL_ACTUAL_CHANNEL,
            bill_forecast=settings.ENERGY_BILL_FORECAST_CHANNEL,
            reconcile=settings.ENERGY_RECONCILE_CHANNEL,
            stakes=settings.ENERGY_STAKES_CHANNEL,
            importer_status=settings.ENERGY_IMPORTER_STATUS_CHANNEL,
        ),
        usage_point_id=settings.ENERGY_USAGE_POINT_ID or None,
        pending_hours=settings.ENERGY_RUN_COST_PENDING_HOURS,
        stakes=StakesConfig(
            near_ratio=settings.ENERGY_STAKES_NEAR_RATIO,
            over_ratio=settings.ENERGY_STAKES_OVER_RATIO,
            stale_after_hours=settings.ENERGY_STALE_AFTER_HOURS,
            portal_enabled=settings.ENERGY_PORTAL_ENABLED,
            portal_interval_hours=settings.ENERGY_PORTAL_INTERVAL_HOURS,
        ),
    )
    replayed = load_processed(Path(settings.ENERGY_PROCESSED_DIR))
    pipeline.replay(replayed)
    bills = load_processed_bills(Path(settings.ENERGY_BILL_PROCESSED_DIR))
    pipeline.replay_bills(bills)
    logger.info(
        "energy_ledger_replayed intervals=%d bills=%d usage_point=%s",
        len(replayed), len(bills), pipeline.usage_point(),
    )
    return pipeline
```

`inbox_loop` body (inside `try:`), replacing the current body:

```python
            scan_at = datetime.now(timezone.utc)
            rows = await asyncio.to_thread(scan_inbox, inbox, processed, now=scan_at)
            bills = await asyncio.to_thread(
                scan_bills, Path(settings.ENERGY_BILL_INBOX_DIR), Path(settings.ENERGY_BILL_PROCESSED_DIR), now=scan_at
            )
            if rows or bills:
                async with lock:
                    now = datetime.now(timezone.utc)
                    outbound = pipeline.ingest_intervals(rows, now=now) if rows else []
                    outbound += pipeline.ingest_bills(bills, now=now) if bills else []
                await publish_all(bus, settings, outbound)
                logger.info(
                    "energy_ingested intervals=%d bills=%d published=%d pending=%d",
                    len(rows), len(bills), len(outbound), pipeline.pending_count(),
                )
```

Add a status loop:

```python
async def status_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    processed = Path(settings.ENERGY_PROCESSED_DIR)
    status_path = Path(settings.ENERGY_PORTAL_STATUS_PATH)
    while True:
        try:
            portal = await asyncio.to_thread(read_portal_status, status_path) if settings.ENERGY_PORTAL_ENABLED else None
            last_file_at = await asyncio.to_thread(latest_processed_at, processed, ".xml")
            async with lock:
                outbound = pipeline.status_tick(now=datetime.now(timezone.utc), portal=portal, last_file_at=last_file_at)
            await publish_all(bus, settings, outbound)
            importer, stakes = outbound[0].payload, outbound[1].payload
            logger.info(
                "energy_status state=%s reason=%s pressure=%s pressure_reason=%s",
                importer.state, importer.reason, stakes.pressure, stakes.pressure_reason,
            )
        except Exception:
            logger.exception("energy_status_cycle_failed")
        await asyncio.sleep(settings.ENERGY_STATUS_INTERVAL_SEC)
```

and add `status_loop(bus, settings, pipeline, lock),` to the `asyncio.gather(...)` in `_main_async`.

- [ ] **Step 8: README** — in `services/orion-energy/README.md`, add the five new channels to "What it publishes", and add a section:

````markdown
## Bills (file drop)

Drop one JSON file per bill into `${ENERGY_HOST_DATA_DIR}/bills/inbox/`. Lines the bill
does not show are simply omitted (unknown, not $0):

```json
{"kind": "energy.bill.actual.v1", "billing_period_start": "2026-08-12",
 "billing_period_end": "2026-09-11", "kwh_billed": 712, "energy_charge": 85.10,
 "customer_charge": 12.00, "taxes": 4.10, "current_charges": 101.23}
```

RMP's in-cycle estimate uses `"kind": "energy.bill.forecast.v1"` with `billing_period_start`,
`as_of`, and `projected_total_usd` and/or `projected_kwh`. The period is
`[billing_period_start, billing_period_end)` at local midnight. Each bill publishes a
`energy.reconcile.v1` row (Orion minus RMP); late usage re-reconciles automatically.

## Status and stakes

Every `ENERGY_STATUS_INTERVAL_SEC` the service publishes `energy.importer.status.v1`
(`healthy` / `stale` / `reauth_required` / `degraded`) and `energy.stakes.snapshot.v1`
(cycle-to-date cost, next-kWh price, projected total vs RMP forecast). Pressure is
`unknown` whenever the importer isn't healthy or a forecast is missing.
````

Add debug queries to the existing "Debug queries" section:

```sql
SELECT reconcile_kind, billing_period_start, orion_total_usd, utility_total_usd, utility_basis, delta_usd, reconcile_gap
FROM energy_reconcile ORDER BY computed_at DESC LIMIT 5;
SELECT as_of, state, reason, usage_lag_hours FROM energy_importer_status ORDER BY as_of DESC LIMIT 3;
SELECT as_of, pressure, pressure_reason, cycle_to_date_total_usd, orion_projected_total_usd, forecast_total_usd
FROM energy_stakes_snapshot ORDER BY as_of DESC LIMIT 3;
```

- [ ] **Step 9: Run tests**

Run: `PYTHONPATH=.:services/orion-energy $PY -m pytest orion/energy/tests services/orion-energy/tests services/orion-energy/evals -q`
Expected: all pass (existing Plan 1 tests unchanged).

- [ ] **Step 10: Commit**

```bash
git add services/orion-energy/app services/orion-energy/.env_example services/orion-energy/README.md \
  services/orion-energy/tests
git diff --cached --check
git commit -m "feat(energy): bill drop, reconcile on bills and late usage, status + stakes tick"
```

---

### Task 6: RMP portal adapter (Playwright, profiled compose service)

**Files:**
- Create: `services/orion-energy/portal/{__init__,settings,selectors,parse,driver,fetch,status,main,reauth}.py`
- Create: `services/orion-energy/Dockerfile.portal`, `Dockerfile.portal.dockerignore`, `requirements-portal.txt`
- Modify: `services/orion-energy/docker-compose.yml`, `.env_example`, `README.md`
- Test: `services/orion-energy/tests/test_portal_parse.py`, `tests/test_portal_fetch.py`

**Interfaces:**
- Consumes: `orion.energy.espi.parse_espi`, `EspiError`; `orion.energy.importer_status.PortalStatus`, `parse_portal_status`, `portal_status_dict`; Task 1 bill models; file-name contract from Task 5 (`rmp-portal-*.xml` / `rmp-portal-*.json`, written atomically so the service never reads a partial file).
- Produces:
  - `portal.parse`: `parse_money(text) -> Optional[float]`, `parse_kwh(text) -> Optional[float]`, `parse_date(text) -> date`, `parse_period(text) -> tuple[date, date]`, `is_login_url(url) -> bool`, `bill_payload_from_fields(fields: Mapping[str,str], *, retrieved_at) -> dict`, `forecast_payload_from_fields(fields, *, retrieved_at) -> dict`
  - `portal.driver`: `PortalDriver` Protocol (`open_usage() -> str`, `download_green_button(*, days: int) -> bytes`, `billing_rows() -> list[dict[str,str]]`, `forecast_fields() -> Optional[dict[str,str]]`, `page_html() -> str`), `open_playwright_driver(*, profile_dir, base_url, headless=True)` async context manager
  - `portal.fetch`: `PortalOutcome(state, reason, xml_file=None, bill_files=())`, `run_once(driver, *, inbox_dir, bill_inbox_dir, raw_dir, backfill_days, now) -> PortalOutcome`
  - `portal.status`: `write_status(path, outcome, *, now) -> PortalStatus`, `read_status(path) -> Optional[PortalStatus]`

- [ ] **Step 1: Write failing tests**

`services/orion-energy/tests/test_portal_parse.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1
from portal.parse import (
    bill_payload_from_fields,
    forecast_payload_from_fields,
    is_login_url,
    parse_date,
    parse_kwh,
    parse_money,
    parse_period,
)

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text,value",
    [("$1,234.56", 1234.56), ("-$5.00", -5.0), ("($5.00)", -5.0), ("$3.20 CR", -3.2), ("", None), ("--", None)],
)
def test_parse_money(text, value) -> None:
    assert parse_money(text) == (None if value is None else pytest.approx(value))


def test_parse_money_rejects_text_without_digits() -> None:
    with pytest.raises(ValueError):
        parse_money("pending")


def test_parse_kwh_and_dates() -> None:
    assert parse_kwh("1,712 kWh") == pytest.approx(1712.0)
    assert parse_kwh("") is None
    assert parse_date("Aug 12, 2026") == date(2026, 8, 12)
    assert parse_date("08/12/2026") == date(2026, 8, 12)
    assert parse_date("2026-08-12") == date(2026, 8, 12)
    assert parse_period("Aug 12, 2026 - Sep 11, 2026") == (date(2026, 8, 12), date(2026, 9, 11))
    assert parse_period("2026-08-12 to 2026-09-11") == (date(2026, 8, 12), date(2026, 9, 11))
    with pytest.raises(ValueError):
        parse_period("Aug 12, 2026")


def test_login_detection() -> None:
    assert is_login_url("https://pacificorpb2c.b2clogin.com/x/B2C_1A_PAC_SIGNIN/oauth2")
    assert not is_login_url("https://csapps.rockymountainpower.net/secure/my-account/energy-usage")


def test_bill_row_to_payload() -> None:
    payload = bill_payload_from_fields(
        {
            "billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh",
            "current_charges": "$101.23", "taxes": "$4.10", "due_date": "Oct 2, 2026",
        },
        retrieved_at=NOW,
    )
    assert payload["kind"] == "energy.bill.actual.v1"
    bill = EnergyBillActualV1.model_validate({k: v for k, v in payload.items() if k != "kind"})
    assert bill.source == "rockymountain_power"
    assert bill.kwh_billed == 712.0 and bill.taxes == pytest.approx(4.10)
    assert bill.energy_charge is None
    assert bill.due_date == date(2026, 10, 2)


@pytest.mark.parametrize("missing", ["billing_period", "kwh", "current_charges"])
def test_bill_row_missing_required_field_raises(missing) -> None:
    fields = {"billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh", "current_charges": "$101.23"}
    del fields[missing]
    with pytest.raises(ValueError):
        bill_payload_from_fields(fields, retrieved_at=NOW)


def test_forecast_fields_to_payload_and_empty_rejected() -> None:
    payload = forecast_payload_from_fields(
        {"billing_period": "Sep 11, 2026 - Oct 12, 2026", "projected_total": "$96.00"}, retrieved_at=NOW,
    )
    fc = EnergyBillForecastV1.model_validate({k: v for k, v in payload.items() if k != "kind"})
    assert fc.projected_total_usd == pytest.approx(96.0) and fc.projected_kwh is None
    with pytest.raises(ValueError):
        forecast_payload_from_fields({"billing_period": "Sep 11, 2026 - Oct 12, 2026"}, retrieved_at=NOW)
```

`services/orion-energy/tests/test_portal_fetch.py`:

```python
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.bills import parse_bill
from portal.fetch import PortalOutcome, run_once
from portal.status import read_status, write_status

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
FIXTURE = Path(__file__).resolve().parents[3] / "orion/energy/tests/fixtures/espi_two_flows.xml"
USAGE_URL = "https://csapps.rockymountainpower.net/secure/my-account/energy-usage"
ROW = {"billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh", "current_charges": "$101.23"}
FORECAST = {"billing_period": "Sep 11, 2026 - Oct 12, 2026", "projected_total": "$96.00"}


class FakeDriver:
    def __init__(self, *, url=USAGE_URL, xml=b"", rows=None, forecast=None, raise_on=None):
        self.url, self.xml, self.rows, self.forecast, self.raise_on = url, xml, rows, forecast, raise_on
        self.days = None

    async def open_usage(self):
        if self.raise_on == "open":
            raise RuntimeError("boom")
        return self.url

    async def download_green_button(self, *, days):
        self.days = days
        return self.xml

    async def billing_rows(self):
        if self.raise_on == "billing":
            raise TimeoutError("selector")
        return list(self.rows or [])

    async def forecast_fields(self):
        return self.forecast

    async def page_html(self):
        return "<html>snapshot</html>"


def _run(tmp_path, driver) -> PortalOutcome:
    return asyncio.run(run_once(
        driver, inbox_dir=tmp_path / "inbox", bill_inbox_dir=tmp_path / "bills",
        raw_dir=tmp_path / "raw", backfill_days=3, now=NOW,
    ))


def test_login_redirect_is_reauth_and_writes_nothing(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(url="https://pacificorpb2c.b2clogin.com/B2C_1A_PAC_SIGNIN"))
    assert (out.state, out.reason) == ("reauth_required", "session_expired")
    assert not (tmp_path / "inbox").exists() and not (tmp_path / "bills").exists()


def test_good_fetch_delivers_xml_and_bills(tmp_path) -> None:
    driver = FakeDriver(xml=FIXTURE.read_bytes(), rows=[ROW], forecast=FORECAST)
    out = _run(tmp_path, driver)
    assert (out.state, out.reason) == ("ok", "fetched")
    assert driver.days == 3
    assert out.xml_file.name.startswith("rmp-portal-") and out.xml_file.suffix == ".xml"
    assert out.xml_file.read_bytes() == FIXTURE.read_bytes()
    assert len(out.bill_files) == 2
    kinds = [json.loads(p.read_text())["kind"] for p in out.bill_files]
    assert kinds == ["energy.bill.actual.v1", "energy.bill.forecast.v1"]
    bill = parse_bill(out.bill_files[0].read_bytes(), retrieved_at=NOW, source_file="x")
    assert bill.source == "rockymountain_power"
    assert not list((tmp_path / "inbox").glob(".*"))  # no leftover partial files


def test_garbage_download_is_error_and_kept_for_debugging(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=b"<html>not espi</html>", rows=[ROW]))
    assert out.state == "error" and out.reason.startswith("espi_invalid")
    assert not (tmp_path / "inbox").exists()
    assert list((tmp_path / "raw").glob("*green_button.xml"))


def test_empty_bill_table_is_error_but_usage_still_delivered(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), rows=[]))
    assert (out.state, out.reason) == ("error", "bill_rows_empty")
    assert out.xml_file is not None and out.xml_file.exists()
    assert list((tmp_path / "raw").glob("*billing.html"))


def test_billing_scrape_exception_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), raise_on="billing"))
    assert (out.state, out.reason) == ("error", "billing_scrape_failed:TimeoutError")


def test_unparseable_bill_row_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), rows=[{"billing_period": "soon"}]))
    assert out.state == "error" and out.reason.startswith("bill_parse_failed")
    assert out.bill_files == ()


def test_driver_crash_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(raise_on="open"))
    assert (out.state, out.reason) == ("error", "RuntimeError")


def test_status_file_keeps_last_success_across_failures(tmp_path) -> None:
    path = tmp_path / "portal" / "status.json"
    assert read_status(path) is None
    write_status(path, PortalOutcome("ok", "fetched"), now=NOW)
    later = NOW + timedelta(days=1)
    s = write_status(path, PortalOutcome("reauth_required", "session_expired"), now=later)
    assert (s.state, s.last_attempt_at, s.last_success_at) == ("reauth_required", later, NOW)
    assert read_status(path) == s
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/tests/test_portal_parse.py services/orion-energy/tests/test_portal_fetch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'portal'`

- [ ] **Step 3: Implement the portal package**

`services/orion-energy/portal/__init__.py`:

```python
"""Rocky Mountain Power portal fetcher. Writes into orion-energy's drop directories."""
```

`portal/settings.py`:

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PortalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENERGY_PORTAL_BASE_URL: str = Field(default="https://csapps.rockymountainpower.net")
    ENERGY_PORTAL_PROFILE_DIR: str = Field(default="/data/energy/portal/profile")
    ENERGY_PORTAL_STATUS_PATH: str = Field(default="/data/energy/portal/status.json")
    ENERGY_PORTAL_RAW_DIR: str = Field(default="/data/energy/portal/raw")
    ENERGY_INBOX_DIR: str = Field(default="/data/energy/inbox")
    ENERGY_BILL_INBOX_DIR: str = Field(default="/data/energy/bills/inbox")
    ENERGY_PORTAL_INTERVAL_HOURS: float = Field(default=24.0, gt=0)
    # Rolling re-fetch window; late AMI intervals arrive for ~3 days.
    ENERGY_PORTAL_BACKFILL_DAYS: int = Field(default=3, ge=1, le=730)
    ENERGY_PORTAL_TIMEOUT_SEC: float = Field(default=300.0, gt=0)


@lru_cache
def get_portal_settings() -> PortalSettings:
    return PortalSettings()
```

`portal/selectors.py`:

```python
"""Every RMP portal URL and DOM selector in one place.

UNVERIFIED: written from the PacifiCorp portal shape (nburns/pacificpower-import) and
not yet checked against a live rockymountainpower.net session. The first live spike
edits only this file. A selector that stops matching surfaces as an error status
(empty download / bill_rows_empty), never as a quiet success.
"""

USAGE_PATH = "/secure/my-account/energy-usage"
BILLING_PATH = "/secure/my-account/billing-payment-history"

LOGIN_URL_MARKERS = ("b2clogin.com", "b2c_1a_pac_signin", "/signin", "/login")
LOGGED_IN_MARKER = "text=Sign Out"

GREEN_BUTTON_OPEN = "text=Green Button"
GREEN_BUTTON_FROM = "input[name='startDate']"
GREEN_BUTTON_TO = "input[name='endDate']"
GREEN_BUTTON_DOWNLOAD = "button:has-text('Download')"
GREEN_BUTTON_DATE_FORMAT = "%m/%d/%Y"

BILL_ROW = "[data-testid='billing-history-row']"
BILL_ROW_FIELDS = {
    "billing_period": "[data-testid='billing-period']",
    "kwh": "[data-testid='kwh-used']",
    "current_charges": "[data-testid='current-charges']",
    "amount_due": "[data-testid='amount-due']",
    "due_date": "[data-testid='due-date']",
}

FORECAST_PANEL = "[data-testid='bill-projection']"
FORECAST_FIELDS = {
    "billing_period": "[data-testid='projection-period']",
    "projected_total": "[data-testid='projected-amount']",
    "projected_kwh": "[data-testid='projected-kwh']",
}
```

`portal/parse.py`:

```python
"""Portal text -> bill contracts. Pure, so it is tested without a browser."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Mapping, Optional

from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

from .selectors import LOGIN_URL_MARKERS

_BLANK = {"", "-", "--", "—", "n/a"}
_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d")
_PERIOD_SPLIT = re.compile(r"\s+(?:-|–|—|to)\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_OPTIONAL_MONEY = ("energy_charge", "customer_charge", "adjustments", "fees", "taxes", "credits", "amount_due")


def parse_money(text: str) -> Optional[float]:
    s = (text or "").strip()
    if s.lower() in _BLANK:
        return None
    match = _NUMBER.search(s)
    if match is None:
        raise ValueError(f"no amount in {text!r}")
    value = float(match.group(0).replace(",", ""))
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or s.upper().endswith("CR")
    return -value if negative else value


def parse_kwh(text: str) -> Optional[float]:
    s = (text or "").strip()
    if s.lower() in _BLANK:
        return None
    match = _NUMBER.search(s)
    if match is None:
        raise ValueError(f"no kWh in {text!r}")
    return float(match.group(0).replace(",", ""))


def parse_date(text: str) -> date:
    s = (text or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {text!r}")


def parse_period(text: str) -> tuple[date, date]:
    parts = _PERIOD_SPLIT.split((text or "").strip())
    if len(parts) != 2:
        raise ValueError(f"unrecognized billing period {text!r}")
    return parse_date(parts[0]), parse_date(parts[1])


def is_login_url(url: str) -> bool:
    low = (url or "").lower()
    return any(marker in low for marker in LOGIN_URL_MARKERS)


def _required(fields: Mapping[str, str], key: str) -> str:
    value = (fields.get(key) or "").strip()
    if not value:
        raise ValueError(f"portal row has no {key}")
    return value


def bill_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = parse_period(_required(fields, "billing_period"))
    kwh = parse_kwh(_required(fields, "kwh"))
    charges = parse_money(_required(fields, "current_charges"))
    if kwh is None or charges is None:
        raise ValueError("portal row has blank kWh or current charges")
    money = {k: parse_money(fields[k]) for k in _OPTIONAL_MONEY if fields.get(k)}
    due = fields.get("due_date")
    bill = EnergyBillActualV1(
        source="rockymountain_power", billing_period_start=start, billing_period_end=end,
        kwh_billed=kwh, current_charges=charges, due_date=parse_date(due) if due else None,
        retrieved_at=retrieved_at, **money,
    )
    return {"kind": ENERGY_BILL_ACTUAL_KIND, **bill.model_dump(mode="json", exclude={"source_file"})}


def forecast_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = parse_period(_required(fields, "billing_period"))
    forecast = EnergyBillForecastV1(
        source="rockymountain_power", billing_period_start=start, billing_period_end=end,
        as_of=retrieved_at, projected_kwh=parse_kwh(fields.get("projected_kwh", "")),
        projected_total_usd=parse_money(fields.get("projected_total", "")), retrieved_at=retrieved_at,
    )
    return {"kind": ENERGY_BILL_FORECAST_KIND, **forecast.model_dump(mode="json", exclude={"source_file"})}
```

`portal/driver.py`:

```python
"""The only module that touches a browser. UNVERIFIED against the live portal."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Protocol

from . import selectors


class PortalDriver(Protocol):
    async def open_usage(self) -> str: ...
    async def download_green_button(self, *, days: int) -> bytes: ...
    async def billing_rows(self) -> list[dict[str, str]]: ...
    async def forecast_fields(self) -> Optional[dict[str, str]]: ...
    async def page_html(self) -> str: ...


class PlaywrightDriver:
    def __init__(self, page: Any, base_url: str) -> None:
        self._page = page
        self._base = base_url.rstrip("/")

    async def open_usage(self) -> str:
        await self._page.goto(self._base + selectors.USAGE_PATH, wait_until="networkidle")
        return self._page.url

    async def download_green_button(self, *, days: int) -> bytes:
        page = self._page
        end = date.today()
        start = end - timedelta(days=days)
        await page.click(selectors.GREEN_BUTTON_OPEN)
        await page.fill(selectors.GREEN_BUTTON_FROM, start.strftime(selectors.GREEN_BUTTON_DATE_FORMAT))
        await page.fill(selectors.GREEN_BUTTON_TO, end.strftime(selectors.GREEN_BUTTON_DATE_FORMAT))
        async with page.expect_download() as info:
            await page.click(selectors.GREEN_BUTTON_DOWNLOAD)
        download = await info.value
        return Path(await download.path()).read_bytes()

    async def _fields(self, scope: Any, spec: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, sel in spec.items():
            loc = scope.locator(sel)
            if await loc.count():
                out[name] = (await loc.first.inner_text()).strip()
        return out

    async def billing_rows(self) -> list[dict[str, str]]:
        await self._page.goto(self._base + selectors.BILLING_PATH, wait_until="networkidle")
        return [await self._fields(row, selectors.BILL_ROW_FIELDS) for row in await self._page.locator(selectors.BILL_ROW).all()]

    async def forecast_fields(self) -> Optional[dict[str, str]]:
        panel = self._page.locator(selectors.FORECAST_PANEL)
        if not await panel.count():
            return None
        return await self._fields(panel.first, selectors.FORECAST_FIELDS)

    async def page_html(self) -> str:
        return await self._page.content()


@asynccontextmanager
async def open_playwright_driver(*, profile_dir: str, base_url: str, headless: bool = True) -> AsyncIterator[PlaywrightDriver]:
    from playwright.async_api import async_playwright  # portal image only; tests use fakes

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(profile_dir, headless=headless, accept_downloads=True)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            yield PlaywrightDriver(page, base_url)
        finally:
            await context.close()
```

`portal/fetch.py`:

```python
"""One fetch attempt: usage XML + bills into the drop directories.

No credentials anywhere: a login redirect means the saved session died, and the
answer is `reauth_required` -- a human logs in once (MFA stays on), never a retry.
Anything empty or unparseable is an error with the raw artifact kept for debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from orion.energy.espi import EspiError, parse_espi

from .driver import PortalDriver
from .parse import bill_payload_from_fields, forecast_payload_from_fields, is_login_url

_STAMP = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class PortalOutcome:
    state: Literal["ok", "reauth_required", "error"]
    reason: str
    xml_file: Optional[Path] = None
    bill_files: tuple[Path, ...] = ()


def _atomic_write(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / f".{name}.part"
    part.write_bytes(data)
    final = directory / name
    part.rename(final)
    return final


def _save_raw(raw_dir: Path, now: datetime, name: str, data: bytes) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{now.strftime(_STAMP)}-{name}").write_bytes(data)


async def _snapshot_html(driver: PortalDriver, raw_dir: Path, now: datetime) -> None:
    try:
        html = await driver.page_html()
    except Exception:  # noqa: BLE001 -- best effort debugging artifact
        return
    _save_raw(raw_dir, now, "billing.html", html.encode())


async def run_once(
    driver: PortalDriver,
    *,
    inbox_dir: Path,
    bill_inbox_dir: Path,
    raw_dir: Path,
    backfill_days: int,
    now: datetime,
) -> PortalOutcome:
    stamp = now.strftime(_STAMP)
    try:
        if is_login_url(await driver.open_usage()):
            return PortalOutcome("reauth_required", "session_expired")
        xml = await driver.download_green_button(days=backfill_days)
        try:
            intervals = parse_espi(xml, retrieved_at=now, source="rockymountain_power")
        except EspiError as exc:
            _save_raw(raw_dir, now, "green_button.xml", xml)
            return PortalOutcome("error", f"espi_invalid:{exc}"[:200])
        if not intervals:
            _save_raw(raw_dir, now, "green_button.xml", xml)
            return PortalOutcome("error", "empty_download")
        xml_file = _atomic_write(inbox_dir, f"rmp-portal-{stamp}.xml", xml)

        try:
            rows = await driver.billing_rows()
            forecast = await driver.forecast_fields()
        except Exception as exc:  # noqa: BLE001 -- any scrape failure is a visible error state
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", f"billing_scrape_failed:{type(exc).__name__}", xml_file=xml_file)
        if not rows:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", "bill_rows_empty", xml_file=xml_file)
        try:
            payloads = [bill_payload_from_fields(row, retrieved_at=now) for row in rows]
            if forecast:
                payloads.append(forecast_payload_from_fields(forecast, retrieved_at=now))
        except ValueError as exc:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", f"bill_parse_failed:{exc}"[:200], xml_file=xml_file)
        bill_files = tuple(
            _atomic_write(bill_inbox_dir, f"rmp-portal-{stamp}-{i:02d}.json", json.dumps(p).encode())
            for i, p in enumerate(payloads)
        )
        return PortalOutcome("ok", "fetched", xml_file=xml_file, bill_files=bill_files)
    except Exception as exc:  # noqa: BLE001 -- the loop must survive and report
        return PortalOutcome("error", type(exc).__name__)
```

`portal/status.py`:

```python
"""status.json: what the fetcher last tried. orion-energy turns it into importer status."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from orion.energy.importer_status import PortalStatus, parse_portal_status, portal_status_dict

from .fetch import PortalOutcome


def read_status(path: Path) -> Optional[PortalStatus]:
    try:
        return parse_portal_status(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def write_status(path: Path, outcome: PortalOutcome, *, now: datetime) -> PortalStatus:
    previous = read_status(path)
    status = PortalStatus(
        state=outcome.state, reason=outcome.reason, last_attempt_at=now,
        last_success_at=now if outcome.state == "ok" else (previous.last_success_at if previous else None),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(portal_status_dict(status)))
    part.rename(path)
    return status
```

`portal/main.py`:

```python
"""Headless fetch loop. `--once --days 730` does the two-year backfill."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .driver import open_playwright_driver
from .fetch import PortalOutcome, run_once
from .settings import PortalSettings, get_portal_settings
from .status import write_status

logger = logging.getLogger("orion-energy-portal")


async def attempt(settings: PortalSettings, *, days: int) -> PortalOutcome:
    now = datetime.now(timezone.utc)
    try:
        async with open_playwright_driver(
            profile_dir=settings.ENERGY_PORTAL_PROFILE_DIR, base_url=settings.ENERGY_PORTAL_BASE_URL,
        ) as driver:
            outcome = await asyncio.wait_for(
                run_once(
                    driver, inbox_dir=Path(settings.ENERGY_INBOX_DIR),
                    bill_inbox_dir=Path(settings.ENERGY_BILL_INBOX_DIR),
                    raw_dir=Path(settings.ENERGY_PORTAL_RAW_DIR), backfill_days=days, now=now,
                ),
                timeout=settings.ENERGY_PORTAL_TIMEOUT_SEC,
            )
    except (TimeoutError, asyncio.TimeoutError):
        outcome = PortalOutcome("error", "timeout")
    except Exception as exc:  # noqa: BLE001
        outcome = PortalOutcome("error", f"browser_failed:{type(exc).__name__}")
    write_status(Path(settings.ENERGY_PORTAL_STATUS_PATH), outcome, now=now)
    logger.info(
        "energy_portal_fetch state=%s reason=%s xml=%s bills=%d",
        outcome.state, outcome.reason, outcome.xml_file.name if outcome.xml_file else None, len(outcome.bill_files),
    )
    return outcome


async def loop(settings: PortalSettings) -> None:
    while True:
        await attempt(settings, days=settings.ENERGY_PORTAL_BACKFILL_DAYS)
        await asyncio.sleep(settings.ENERGY_PORTAL_INTERVAL_HOURS * 3600.0)


def main() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="[ORION_ENERGY_PORTAL] %(asctime)s %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    settings = get_portal_settings()
    if args.once:
        outcome = asyncio.run(attempt(settings, days=args.days or settings.ENERGY_PORTAL_BACKFILL_DAYS))
        sys.exit(0 if outcome.state == "ok" else 1)
    asyncio.run(loop(settings))


if __name__ == "__main__":
    main()
```

`portal/reauth.py`:

```python
"""One-time headed login into the persistent profile. MFA stays on: you complete it.

Run on a host with a display, against the same profile dir the container mounts:
  python -m portal.reauth --profile /mnt/storage-warm/orion-energy/portal/profile \
      --status /mnt/storage-warm/orion-energy/portal/status.json
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from . import selectors
from .fetch import PortalOutcome
from .settings import get_portal_settings
from .status import write_status


async def reauth(*, profile_dir: str, base_url: str, timeout_sec: float) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(profile_dir, headless=False)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(base_url.rstrip("/") + selectors.USAGE_PATH)
            print(f"Log in (including MFA) in the browser window. Waiting up to {int(timeout_sec)}s ...")
            await page.wait_for_selector(selectors.LOGGED_IN_MARKER, timeout=timeout_sec * 1000)
        finally:
            await context.close()


def main() -> None:
    settings = get_portal_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=settings.ENERGY_PORTAL_PROFILE_DIR)
    parser.add_argument("--status", default=settings.ENERGY_PORTAL_STATUS_PATH)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    asyncio.run(reauth(profile_dir=args.profile, base_url=settings.ENERGY_PORTAL_BASE_URL, timeout_sec=args.timeout))
    write_status(Path(args.status), PortalOutcome("ok", "reauth_completed"), now=datetime.now(timezone.utc))
    print("Session saved. The next scheduled fetch will use it.")


if __name__ == "__main__":
    main()
```

Note on `write_status` after reauth: it records `state=ok` and sets `last_success_at` to now; importer health then comes from real usage freshness, so a reauth alone never makes the importer read healthy with no new data.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/tests -q`
Expected: all pass.

- [ ] **Step 5: Image + compose + env + README**

`services/orion-energy/requirements-portal.txt`:

```text
playwright==1.49.0
pydantic==2.9.2
pydantic-settings==2.7.1
tzdata==2024.2
```

`services/orion-energy/Dockerfile.portal`:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/orion-energy/requirements-portal.txt ./requirements-portal.txt
RUN pip install --no-cache-dir -r requirements-portal.txt

COPY services/orion-energy/portal /app/portal
COPY orion /app/orion

CMD ["python", "-m", "portal.main"]
```

`Dockerfile.portal.dockerignore`: copy `Dockerfile.dockerignore` verbatim (`cp services/orion-energy/Dockerfile.dockerignore services/orion-energy/Dockerfile.portal.dockerignore`).

Append to `services/orion-energy/docker-compose.yml` under `services:` (before `networks:`):

```yaml
  orion-energy-portal:
    container_name: ${PROJECT}-orion-energy-portal
    # Not started by a plain `up`: needs a one-time headed reauth first (README "Portal").
    profiles: ["portal"]
    build:
      context: ../..
      dockerfile: services/orion-energy/Dockerfile.portal
    restart: unless-stopped
    env_file:
      - .env
    shm_size: "1gb"
    volumes:
      - ${ENERGY_HOST_DATA_DIR:-/mnt/storage-warm/orion-energy}:/data/energy
    networks:
      - app-net
```

Append to `services/orion-energy/.env_example`:

```text
# --- orion-energy-portal (compose profile "portal"); no RMP credentials live here ---
ENERGY_PORTAL_BASE_URL=https://csapps.rockymountainpower.net
ENERGY_PORTAL_PROFILE_DIR=/data/energy/portal/profile
ENERGY_PORTAL_RAW_DIR=/data/energy/portal/raw
ENERGY_PORTAL_BACKFILL_DAYS=3
ENERGY_PORTAL_TIMEOUT_SEC=300
```

Run `python scripts/sync_local_env_from_example.py` and `python scripts/check_compose_no_relative_mounts.py` from the worktree root.

Add to `services/orion-energy/README.md`:

````markdown
## Portal (optional, compose profile `portal`)

`orion-energy-portal` reuses a saved browser session to download Green Button XML and
scrape bills into the same drop directories. It stores **no** RMP password; MFA stays on.
Selectors are UNVERIFIED until the first live run (`portal/selectors.py`).

1. One-time login on a host with a display (same profile dir the container mounts):
   ```bash
   pip install playwright==1.49.0 pydantic-settings==2.7.1 && python -m playwright install chromium
   cd services/orion-energy && PYTHONPATH=../..:. python -m portal.reauth \
     --profile /mnt/storage-warm/orion-energy/portal/profile \
     --status /mnt/storage-warm/orion-energy/portal/status.json
   ```
2. Two-year backfill once, then the daily loop:
   ```bash
   scripts/safe_docker_build.sh orion-energy --profile portal run --rm orion-energy-portal python -m portal.main --once --days 730
   scripts/safe_docker_build.sh orion-energy --profile portal up -d --build orion-energy-portal
   ```
3. Set `ENERGY_PORTAL_ENABLED=true` for `orion-energy` and restart it.

When the session dies the importer reads `reauth_required`; repeat step 1. Failed
downloads/scrapes keep the raw artifact in `${ENERGY_HOST_DATA_DIR}/portal/raw/` (may
contain account details — local disk only).
````

- [ ] **Step 6: Commit**

```bash
git add services/orion-energy/portal services/orion-energy/Dockerfile.portal \
  services/orion-energy/Dockerfile.portal.dockerignore services/orion-energy/requirements-portal.txt \
  services/orion-energy/docker-compose.yml services/orion-energy/.env_example services/orion-energy/README.md \
  services/orion-energy/tests/test_portal_parse.py services/orion-energy/tests/test_portal_fetch.py
git diff --cached --check
git commit -m "feat(energy): RMP portal fetcher with saved session, reauth, and status file"
```

---

### Task 7: Curiosity energy-stakes hold (flag default off)

**Files:**
- Create: `services/orion-hub/scripts/energy_stakes_gate.py`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (`__init__` ~L548, `tick()` ~L1437), `services/orion-hub/scripts/main.py` (~L591), `services/orion-hub/app/settings.py` (~L902), `services/orion-hub/.env_example` (~L564)
- Test: `services/orion-hub/tests/test_energy_stakes_gate.py`, `services/orion-hub/tests/test_curiosity_investigation.py` (append)

**Interfaces:**
- Consumes: SQL table `energy_stakes_snapshot` (Task 2 column names); `AttentionSchemaV1`, `bind_correlation`, `ATTENTION_SCHEMA_CHANNEL`, `ATTENTION_SCHEMA_KIND` (already imported in `curiosity_investigation.py` L207).
- Produces:
  - `scripts.energy_stakes_gate`: `HOLD_REASON = "held_off:energy_stakes"`, `HOLD_PRESSURES`, `EnergyHold` dataclass, `energy_stakes_hold(snapshot: Optional[Mapping[str, Any]], *, now: datetime, max_age_sec: float) -> Optional[EnergyHold]`, `read_latest_energy_stakes(database_url: str) -> Optional[dict[str, Any]]`, `hold_attention_row(hold: EnergyHold, *, now: datetime) -> AttentionSchemaV1`
  - `CuriosityInvestigation(..., energy_stakes_enabled: bool = False, energy_stakes_reader: Optional[Callable[[], Awaitable[Optional[Mapping[str, Any]]]]] = None, energy_stakes_max_age_sec: float = 1800.0)`; `tick()` returns `"held_off:energy_stakes"` when held.
  - Hub settings `ORION_ENERGY_STAKES_ENABLED` (default `False`), `ORION_ENERGY_STAKES_MAX_AGE_SEC` (default `1800.0`).

- [ ] **Step 1: Write failing gate tests** — `services/orion-hub/tests/test_energy_stakes_gate.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.energy_stakes_gate import HOLD_REASON, energy_stakes_hold, hold_attention_row

NOW = datetime(2026, 9, 27, 18, tzinfo=timezone.utc)


def _snap(pressure="over_forecast", age_sec=60.0, **over) -> dict:
    base = {
        "as_of": NOW - timedelta(seconds=age_sec), "pressure": pressure, "pressure_reason": "ratio=1.105",
        "projected_to_forecast_ratio": 1.105, "orion_projected_total_usd": 88.4, "forecast_total_usd": 80.0,
        "marginal_usd_per_kwh": 0.12, "importer_state": "healthy",
    }
    base.update(over)
    return base


@pytest.mark.parametrize("pressure", ["over_forecast", "near_forecast"])
def test_fresh_pressure_holds(pressure) -> None:
    hold = energy_stakes_hold(_snap(pressure), now=NOW, max_age_sec=1800)
    assert hold is not None and hold.pressure == pressure


@pytest.mark.parametrize(
    "snapshot",
    [None, {}, _snap("normal"), _snap("unknown"), _snap(age_sec=7200.0), _snap(as_of=None)],
)
def test_never_holds_without_fresh_evidence(snapshot) -> None:
    assert energy_stakes_hold(snapshot, now=NOW, max_age_sec=1800) is None


def test_iso_string_as_of_is_accepted() -> None:
    snap = _snap(as_of=(NOW - timedelta(seconds=30)).isoformat())
    assert energy_stakes_hold(snap, now=NOW, max_age_sec=1800) is not None


def test_hold_row_is_a_curiosity_attention_row() -> None:
    hold = energy_stakes_hold(_snap(), now=NOW, max_age_sec=1800)
    row = hold_attention_row(hold, now=NOW)
    assert row.process == "curiosity"
    assert row.attention_reason == HOLD_REASON == "held_off:energy_stakes"
    assert row.attended_id is None
    assert row.narrative_kind == "computed"
    assert "$88.40" in row.reason_narrative and "$80.00" in row.reason_narrative
    assert row.entry_id == f"curiosity:{HOLD_REASON}:{hold.as_of.isoformat()}"


def test_unknown_numbers_render_as_unknown_not_zero() -> None:
    hold = energy_stakes_hold(_snap(marginal_usd_per_kwh=None), now=NOW, max_age_sec=1800)
    assert "unknown" in hold_attention_row(hold, now=NOW).reason_narrative
```

(If `scripts` import collides with the repo-root `scripts` package when run together with root tests, copy the `_ensure_hub_scripts_import_path()` preamble from `services/orion-hub/tests/test_cabinet_cooling_routes.py` to the top of this file.)

- [ ] **Step 2: Write failing loop tests** — append to `services/orion-hub/tests/test_curiosity_investigation.py`:

```python
# --- energy stakes hold (ORION_ENERGY_STAKES_ENABLED) ------------------------

from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL  # noqa: E402


def _energy_snapshot(pressure="over_forecast", age_sec=60.0) -> dict:
    return {
        "as_of": datetime.now(timezone.utc) - timedelta(seconds=age_sec), "pressure": pressure,
        "pressure_reason": "ratio=1.105", "projected_to_forecast_ratio": 1.105,
        "orion_projected_total_usd": 88.4, "forecast_total_usd": 80.0,
        "marginal_usd_per_kwh": 0.12, "importer_state": "healthy",
    }


def _energy_reader(snapshot, calls: list):
    async def read():
        calls.append(1)
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot
    return read


def test_energy_flag_off_never_reads_the_snapshot() -> None:
    bus, calls = _FakeBus(), []
    loop = _loop(bus, energy_stakes_enabled=False, energy_stakes_reader=_energy_reader(_energy_snapshot(), calls))
    assert asyncio.run(loop.tick()) is None
    assert calls == []
    assert len(bus.journal) == 1


def test_energy_over_forecast_holds_and_leaves_an_attention_row() -> None:
    bus = _FakeBus()
    loop = _loop(bus, energy_stakes_enabled=True, energy_stakes_reader=_energy_reader(_energy_snapshot(), []))
    assert asyncio.run(loop.tick()) == "held_off:energy_stakes"
    assert bus.journal == []
    assert loop._done_today == 0
    rows = [e.payload for c, e in bus.published if c == ATTENTION_SCHEMA_CHANNEL]
    assert len(rows) == 1
    assert rows[0]["attention_reason"] == "held_off:energy_stakes"
    assert rows[0]["process"] == "curiosity"


def test_energy_hold_never_blocks_a_forced_run() -> None:
    bus, calls = _FakeBus(), []
    loop = _loop(bus, energy_stakes_enabled=True, energy_stakes_reader=_energy_reader(_energy_snapshot(), calls))
    assert asyncio.run(loop.tick(force=True)) is None
    assert calls == []
    assert len(bus.journal) == 1


@pytest.mark.parametrize(
    "snapshot",
    [_energy_snapshot("normal"), _energy_snapshot("unknown"), _energy_snapshot(age_sec=7200.0), None, RuntimeError("pg down")],
)
def test_energy_no_fresh_pressure_never_holds(snapshot) -> None:
    bus = _FakeBus()
    loop = _loop(bus, energy_stakes_enabled=True, energy_stakes_reader=_energy_reader(snapshot, []))
    assert asyncio.run(loop.tick()) is None
    assert len(bus.journal) == 1
```

(`timedelta` is already imported at the top of this file.)

- [ ] **Step 3: Run to verify failure**

Run: `cd services/orion-hub && PYTHONPATH=../..:. $PY -m pytest tests/test_energy_stakes_gate.py tests/test_curiosity_investigation.py -q -k "energy"` (match the invocation in `.github/workflows/orion-reading-tests.yml` if it differs)
Expected: FAIL — `ModuleNotFoundError: scripts.energy_stakes_gate` and `TypeError: unexpected keyword argument 'energy_stakes_enabled'`

- [ ] **Step 4: Implement the gate** — `services/orion-hub/scripts/energy_stakes_gate.py`:

```python
"""Energy as a stake for curiosity's scheduled spend (ORION_ENERGY_STAKES_ENABLED, default off).

Reads the latest energy_stakes_snapshot row orion-energy materializes. Holds only
when that row is fresh AND says the house's projected bill is at/over Rocky Mountain
Power's own forecast. Unknown, stale, or unreadable never holds -- a broken meter must
not silence curiosity. Never reads house_share_cost_usd.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from orion.schemas.attention_schema import AttentionSchemaV1

HOLD_REASON = "held_off:energy_stakes"
HOLD_PRESSURES = frozenset({"near_forecast", "over_forecast"})

_SNAPSHOT_SQL = """
SELECT as_of, pressure, pressure_reason, projected_to_forecast_ratio,
       orion_projected_total_usd, forecast_total_usd, marginal_usd_per_kwh, importer_state
FROM energy_stakes_snapshot
ORDER BY as_of DESC
LIMIT 1
"""


@dataclass(frozen=True)
class EnergyHold:
    as_of: datetime
    pressure: str
    pressure_reason: str
    ratio: Optional[float]
    projected_total_usd: Optional[float]
    forecast_total_usd: Optional[float]
    marginal_usd_per_kwh: Optional[float]


def _aware(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def energy_stakes_hold(
    snapshot: Optional[Mapping[str, Any]], *, now: datetime, max_age_sec: float
) -> Optional[EnergyHold]:
    if not snapshot:
        return None
    as_of = _aware(snapshot.get("as_of"))
    if as_of is None or (now - as_of).total_seconds() > max_age_sec:
        return None
    pressure = snapshot.get("pressure")
    if pressure not in HOLD_PRESSURES:
        return None
    return EnergyHold(
        as_of=as_of, pressure=pressure, pressure_reason=str(snapshot.get("pressure_reason") or pressure),
        ratio=snapshot.get("projected_to_forecast_ratio"),
        projected_total_usd=snapshot.get("orion_projected_total_usd"),
        forecast_total_usd=snapshot.get("forecast_total_usd"),
        marginal_usd_per_kwh=snapshot.get("marginal_usd_per_kwh"),
    )


async def read_latest_energy_stakes(database_url: str) -> Optional[dict[str, Any]]:
    if not database_url:
        return None
    import asyncpg

    conn = await asyncpg.connect(dsn=database_url)
    try:
        row = await conn.fetchrow(_SNAPSHOT_SQL)
    finally:
        await conn.close()
    return dict(row) if row else None


def _usd(value: Optional[float]) -> str:
    return "unknown" if value is None else f"${value:.2f}"


def hold_attention_row(hold: EnergyHold, *, now: datetime) -> AttentionSchemaV1:
    rate = "unknown" if hold.marginal_usd_per_kwh is None else f"${hold.marginal_usd_per_kwh:.4f}/kWh"
    return AttentionSchemaV1(
        entry_id=f"curiosity:{HOLD_REASON}:{hold.as_of.isoformat()}",
        generated_at=now,
        process="curiosity",
        attended_id=None,
        attention_reason=HOLD_REASON,
        reason_narrative=(
            f"Held a scheduled investigation: projected bill {_usd(hold.projected_total_usd)} vs "
            f"Rocky Mountain Power forecast {_usd(hold.forecast_total_usd)} "
            f"({hold.pressure}, {hold.pressure_reason}); the next kWh costs {rate}."
        ),
        narrative_kind="computed",
    )
```

- [ ] **Step 5: Wire into the loop** — in `services/orion-hub/scripts/curiosity_investigation.py`:
  - Import near L207: `from .energy_stakes_gate import HOLD_REASON as ENERGY_HOLD_REASON, EnergyHold, energy_stakes_hold, hold_attention_row` — match how this module imports its sibling modules (check the existing sibling-import style, e.g. `curiosity_offer_decisions`; if they use `from scripts.x import`, do the same). Ensure `Awaitable`, `Callable`, `Mapping`, `Any` are imported from `typing`.
  - `__init__` keyword params (append at the end of the keyword-only list) and body:

```python
        energy_stakes_enabled: bool = False,
        energy_stakes_reader: Optional[Callable[[], Awaitable[Optional[Mapping[str, Any]]]]] = None,
        energy_stakes_max_age_sec: float = 1800.0,
```

```python
        self.energy_stakes_enabled = bool(energy_stakes_enabled)
        self.energy_stakes_reader = energy_stakes_reader
        self.energy_stakes_max_age_sec = float(energy_stakes_max_age_sec)
```

  - In `tick()`, directly after the scheduling block's `return reason` (the `if reason is not None:` block that logs `curiosity_investigation_blocked reason=%s`) and before the `pg_readonly_role` check:

```python
        # Discretionary spend only: a forced run is an operator's call, not curiosity's.
        if self.energy_stakes_enabled and not force:
            if await self._energy_stakes_hold(now) is not None:
                return ENERGY_HOLD_REASON
```

  - Add the method next to `_publish_attention_schema`:

```python
    async def _energy_stakes_hold(self, now: datetime) -> Optional[EnergyHold]:
        """A fresh snapshot at/over RMP's forecast holds; anything else never does."""
        reader = self.energy_stakes_reader
        if reader is None:
            return None
        try:
            snapshot = await asyncio.wait_for(reader(), timeout=5.0)
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_energy_stakes_unreadable -- not holding", exc_info=True)
            return None
        hold = energy_stakes_hold(snapshot, now=now, max_age_sec=self.energy_stakes_max_age_sec)
        if hold is None:
            return None
        logger.info(
            "curiosity_investigation_blocked reason=%s pressure=%s ratio=%s snapshot_as_of=%s",
            ENERGY_HOLD_REASON, hold.pressure, hold.ratio, hold.as_of.isoformat(),
        )
        if self._bus is not None:
            try:
                row, corr = bind_correlation(hold_attention_row(hold, now=now))
                await self._bus.publish(
                    ATTENTION_SCHEMA_CHANNEL,
                    BaseEnvelope(
                        kind=ATTENTION_SCHEMA_KIND, source=self._source_ref,
                        correlation_id=corr, payload=row.model_dump(mode="json"),
                    ),
                )
            except Exception:  # noqa: BLE001
                logger.warning("curiosity_energy_hold_publish_failed", exc_info=True)
        return hold
```

- [ ] **Step 6: Settings + wiring** — `services/orion-hub/app/settings.py` next to `HUB_CURIOSITY_SPEND_LOG_ENABLED`:

```python
    # Let curiosity hold a scheduled investigation when orion-energy's stakes snapshot
    # says the house bill is at/over RMP's forecast. Off = curiosity unchanged.
    ORION_ENERGY_STAKES_ENABLED: bool = Field(default=False, alias="ORION_ENERGY_STAKES_ENABLED")
    ORION_ENERGY_STAKES_MAX_AGE_SEC: float = Field(default=1800.0, alias="ORION_ENERGY_STAKES_MAX_AGE_SEC")
```

`services/orion-hub/.env_example` next to `HUB_CURIOSITY_SPEND_LOG_ENABLED=true`:

```text
# Curiosity may hold a scheduled run when the house bill trends at/over RMP's forecast.
ORION_ENERGY_STAKES_ENABLED=false
# A stakes snapshot older than this never holds anything.
ORION_ENERGY_STAKES_MAX_AGE_SEC=1800
```

In `services/orion-hub/scripts/main.py`, import `read_latest_energy_stakes` (same sibling-import style as the other `scripts` imports in that file) and add to the `CuriosityInvestigation(...)` call (~L591):

```python
                energy_stakes_enabled=settings.ORION_ENERGY_STAKES_ENABLED,
                energy_stakes_reader=lambda: read_latest_energy_stakes(os.getenv("DATABASE_URL", "").strip()),
                energy_stakes_max_age_sec=settings.ORION_ENERGY_STAKES_MAX_AGE_SEC,
```

Check `services/orion-hub/docker-compose.yml`: if its `environment:` block enumerates curiosity keys explicitly (search `HUB_CURIOSITY_SPEND_LOG_ENABLED`), add the two new keys the same way; if they arrive via `env_file` only, leave compose alone. Run `python scripts/sync_local_env_from_example.py` and `python scripts/check_env_template_parity.py`.

- [ ] **Step 7: Run tests**

Run: `cd services/orion-hub && PYTHONPATH=../..:. $PY -m pytest tests/test_energy_stakes_gate.py tests/test_curiosity_investigation.py tests/test_curiosity_spend_log.py -q`
Expected: all pass (existing curiosity tests unchanged → flag-off behavior unchanged).

- [ ] **Step 8: CI** — add `services/orion-hub/tests/test_energy_stakes_gate.py` and `services/orion-hub/scripts/energy_stakes_gate.py` to `.github/workflows/orion-reading-tests.yml` (path filters and the pytest command that already runs `test_curiosity_investigation.py`).

- [ ] **Step 9: Commit**

```bash
git add services/orion-hub/scripts/energy_stakes_gate.py services/orion-hub/scripts/curiosity_investigation.py \
  services/orion-hub/scripts/main.py services/orion-hub/app/settings.py services/orion-hub/.env_example \
  services/orion-hub/tests/test_energy_stakes_gate.py services/orion-hub/tests/test_curiosity_investigation.py \
  .github/workflows/orion-reading-tests.yml
git diff --cached --check
git commit -m "feat(hub): curiosity holds on energy stakes behind ORION_ENERGY_STAKES_ENABLED"
```

---

### Task 8: Hub Energy strip

**Files:**
- Create: `services/orion-hub/scripts/energy_routes.py`, `services/orion-hub/static/js/energy-strip.js`, `services/orion-hub/static/js/energy-strip.test.js`
- Modify: `services/orion-hub/scripts/api_routes.py` (~L196, ~L223), `services/orion-hub/templates/index.html` (after the cooling strip ~L3185; script tags ~L4228), `services/orion-hub/static/js/biometrics-view.js` (~L645, ~L661, ~L688), `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`
- Test: `services/orion-hub/tests/test_energy_routes.py`, `services/orion-hub/tests/test_energy_strip_panel.py`, `static/js/energy-strip.test.js`

**Interfaces:**
- Consumes: SQL tables `energy_stakes_snapshot`, `energy_importer_status`, `energy_reconcile`, `energy_usage_interval` (Task 2 / Plan 1 column names).
- Produces: `GET /api/energy/latest` → `{"ok": bool, "stakes": dict|None, "importer": dict|None, "reconcile": {"actual"?: dict, "forecast"?: dict}}`; `GET /api/energy/usage/daily?days=14` → `{"ok": bool, "days": int, "points": [{"day": "YYYY-MM-DD", "kwh": float, "hours": int}]}`; `window.OrionEnergyStrip = {activate, deactivate, refresh}`; Hub setting `HUB_ENERGY_TIMEZONE` (default `America/Denver`).

- [ ] **Step 1: Write failing route tests** — `services/orion-hub/tests/test_energy_routes.py` (copy the module preamble — env defaults loop + `_ensure_hub_scripts_import_path()` — verbatim from `test_cabinet_cooling_routes.py` lines 1–42, then):

```python
from scripts import energy_routes  # noqa: E402

AS_OF = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(energy_routes.router)
    return TestClient(app)


def test_latest_shapes_and_keeps_unknown_null(client, monkeypatch) -> None:
    stakes = {"as_of": AS_OF, "pressure": "unknown", "pressure_reason": "no_forecast_total",
              "cycle_to_date_total_usd": 17.2, "forecast_total_usd": None, "cycle_start": date(2026, 9, 1)}
    importer = {"as_of": AS_OF, "state": "healthy", "reason": "usage_fresh", "usage_lag_hours": 26.0}
    reconcile = [{"reconcile_kind": "actual", "billing_period_start": date(2026, 8, 12),
                  "orion_total_usd": 99.0, "utility_total_usd": 97.13, "delta_usd": 1.87,
                  "reconcile_gap": None, "bucket_deltas": '{"customer_charge": 0.16}'}]

    async def fake():
        return stakes, importer, reconcile

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    body = client.get("/api/energy/latest").json()
    assert body["ok"] is True
    assert body["stakes"]["forecast_total_usd"] is None
    assert body["stakes"]["as_of"] == "2026-09-27T18:00:00Z"
    assert body["stakes"]["cycle_start"] == "2026-09-01"
    assert body["importer"]["state"] == "healthy"
    assert body["reconcile"]["actual"]["bucket_deltas"] == {"customer_charge": 0.16}


def test_latest_without_rows_is_not_ok(client, monkeypatch) -> None:
    async def fake():
        return None, None, []

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    assert client.get("/api/energy/latest").json() == {"ok": False, "stakes": None, "importer": None, "reconcile": {}}


def test_latest_db_failure_is_reported(client, monkeypatch) -> None:
    async def boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(energy_routes, "_latest_query", boom)
    body = client.get("/api/energy/latest").json()
    assert body["ok"] is False and body["error"] == "energy_unavailable"


def test_daily_usage(client, monkeypatch) -> None:
    async def fake(days, tz):
        assert (days, tz) == (7, energy_routes.settings.HUB_ENERGY_TIMEZONE)
        return [{"day": date(2026, 9, 26), "kwh": 24.5, "n": 24}]

    monkeypatch.setattr(energy_routes, "_daily_query", fake)
    body = client.get("/api/energy/usage/daily?days=7").json()
    assert body == {"ok": True, "days": 7, "points": [{"day": "2026-09-26", "kwh": 24.5, "hours": 24}]}
```

(Add `from datetime import date` to the preamble imports.)

- [ ] **Step 2: Write failing panel + JS tests**

`services/orion-hub/tests/test_energy_strip_panel.py`:

```python
from __future__ import annotations

from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
INDEX = (HUB / "templates/index.html").read_text()
JS = (HUB / "static/js/energy-strip.js").read_text()
BIOMETRICS_VIEW_JS = (HUB / "static/js/biometrics-view.js").read_text()


def test_strip_markup_present() -> None:
    for element_id in (
        "energyStrip", "energyImporterState", "energyCycleToDate", "energyProjected",
        "energyForecast", "energyMarginal", "energyPressure", "energyReconcileActual",
        "energyReconcileForecast", "energyDailyBars",
    ):
        assert f'id="{element_id}"' in INDEX


def test_script_loaded_with_cache_bust() -> None:
    assert '<script src="/static/js/energy-strip.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in INDEX


def test_js_polls_both_endpoints() -> None:
    assert "/api/energy/latest" in JS and "/api/energy/usage/daily" in JS
    assert "window.OrionEnergyStrip" in JS


def test_cabinet_subview_activates_and_deactivates_the_strip() -> None:
    assert "window.OrionEnergyStrip.activate()" in BIOMETRICS_VIEW_JS
    assert BIOMETRICS_VIEW_JS.count("window.OrionEnergyStrip.deactivate()") == 2
```

`services/orion-hub/static/js/energy-strip.test.js`:

```javascript
const test = require("node:test");
const assert = require("node:assert/strict");
const strip = require("./energy-strip.js");

const { fmtUsd, fmtRate, pressureLabel, importerLabel, reconcileLine, barHeights } = strip;

test("unknown money renders as unknown, never $0.00", () => {
  assert.equal(fmtUsd(null), "unknown");
  assert.equal(fmtUsd(undefined), "unknown");
  assert.equal(fmtUsd(0), "$0.00");
  assert.equal(fmtUsd(-2), "-$2.00");
  assert.equal(fmtUsd(88.4), "$88.40");
});

test("rate and labels", () => {
  assert.equal(fmtRate(0.12), "$0.1200/kWh");
  assert.equal(fmtRate(null), "unknown");
  assert.equal(pressureLabel("over_forecast"), "over RMP forecast");
  assert.equal(pressureLabel("bogus"), "unknown");
  assert.equal(importerLabel(null), "importer: no status yet");
  assert.equal(importerLabel({ state: "reauth_required", reason: "session_expired" }), "importer: reauth_required (session_expired)");
});

test("reconcile lines say gaps plainly", () => {
  assert.equal(reconcileLine("actual", null), "No closed bill reconciled yet.");
  assert.equal(reconcileLine("forecast", null), "No RMP forecast yet.");
  assert.equal(
    reconcileLine("actual", { billing_period_start: "2026-08-12", reconcile_gap: "usage_incomplete" }),
    "Last bill 2026-08-12: Orion can't price this period yet (usage_incomplete)",
  );
  assert.equal(
    reconcileLine("forecast", { billing_period_start: "2026-09-11", orion_total_usd: 88.4, utility_total_usd: 80, delta_usd: 8.4 }),
    "RMP forecast 2026-09-11: Orion $88.40 vs RMP $80.00 (diff $8.40)",
  );
});

test("bar heights scale to the max day", () => {
  assert.deepEqual(barHeights([{ kwh: 10 }, { kwh: 20 }, { kwh: 0 }], 60), [30, 60, 0]);
  assert.deepEqual(barHeights([], 60), []);
  assert.deepEqual(barHeights([{ kwh: 0 }], 60), [0]);
});
```

- [ ] **Step 3: Run to verify failure**

Run:
```bash
cd services/orion-hub && PYTHONPATH=../..:. $PY -m pytest tests/test_energy_routes.py tests/test_energy_strip_panel.py -q
node --test static/js/energy-strip.test.js
```
Expected: FAIL — missing module / file.

- [ ] **Step 4: Routes** — `services/orion-hub/scripts/energy_routes.py` (match `cabinet_cooling_routes.py`'s `settings` import and `logger` naming):

```python
"""Hub read APIs for house electricity (orion-energy tables written by sql-writer).

NULL stays null in JSON: an unknown dollar amount is never rendered as $0.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Query

from .settings import settings

logger = logging.getLogger("orion-hub.energy")
router = APIRouter(prefix="/api/energy", tags=["energy"])

_STAKES_SQL = """
SELECT as_of, cycle_start, cycle_end, covered_through, cycle_accumulated_kwh, cycle_to_date_total_usd,
       marginal_usd_per_kwh, orion_projected_total_usd, forecast_total_usd, forecast_as_of,
       projected_to_forecast_ratio, importer_state, pressure, pressure_reason, tariff_version
FROM energy_stakes_snapshot ORDER BY as_of DESC LIMIT 1
"""
_IMPORTER_SQL = """
SELECT as_of, state, reason, source, last_success_at, last_attempt_at, latest_interval_end, usage_lag_hours
FROM energy_importer_status ORDER BY as_of DESC LIMIT 1
"""
_RECONCILE_SQL = """
SELECT DISTINCT ON (reconcile_kind)
       reconcile_kind, billing_period_start, billing_period_end, utility_as_of, utility_kwh,
       utility_total_usd, utility_basis, orion_kwh, orion_total_usd, orion_method, reconcile_gap,
       delta_kwh, delta_usd, delta_pct, bucket_deltas, tariff_version, computed_at
FROM energy_reconcile
ORDER BY reconcile_kind, billing_period_start DESC, computed_at DESC
"""
_DAILY_SQL = """
SELECT (interval_start AT TIME ZONE $2)::date AS day, SUM(energy_kwh) AS kwh, COUNT(*) AS n
FROM energy_usage_interval
WHERE interval_start >= $1
GROUP BY 1 ORDER BY 1
"""


async def _connect():
    import asyncpg

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return await asyncpg.connect(dsn=database_url)


async def _latest_query() -> tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], list[Mapping[str, Any]]]:
    conn = await _connect()
    try:
        return (
            await conn.fetchrow(_STAKES_SQL),
            await conn.fetchrow(_IMPORTER_SQL),
            list(await conn.fetch(_RECONCILE_SQL)),
        )
    finally:
        await conn.close()


async def _daily_query(days: int, tz: str) -> list[Mapping[str, Any]]:
    conn = await _connect()
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return list(await conn.fetch(_DAILY_SQL, since, tz))
    finally:
        await conn.close()


def _jsonable(row: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        elif isinstance(value, date):
            value = value.isoformat()
        elif key == "bucket_deltas" and isinstance(value, str):
            value = json.loads(value)
        out[key] = value
    return out


@router.get("/latest")
async def api_energy_latest() -> dict[str, Any]:
    try:
        stakes, importer, reconcile = await _latest_query()
    except Exception:  # noqa: BLE001
        logger.warning("energy_latest_unavailable", exc_info=True)
        return {"ok": False, "error": "energy_unavailable", "stakes": None, "importer": None, "reconcile": {}}
    return {
        "ok": stakes is not None,
        "stakes": _jsonable(stakes),
        "importer": _jsonable(importer),
        "reconcile": {row["reconcile_kind"]: _jsonable(row) for row in reconcile},
    }


@router.get("/usage/daily")
async def api_energy_usage_daily(days: int = Query(14, ge=1, le=90)) -> dict[str, Any]:
    try:
        rows = await _daily_query(days, settings.HUB_ENERGY_TIMEZONE)
    except Exception:  # noqa: BLE001
        logger.warning("energy_daily_unavailable", exc_info=True)
        return {"ok": False, "error": "energy_unavailable", "days": days, "points": []}
    points = [
        {"day": r["day"].isoformat() if isinstance(r["day"], date) else str(r["day"]), "kwh": float(r["kwh"]), "hours": int(r["n"])}
        for r in rows
    ]
    return {"ok": True, "days": days, "points": points}
```

Add to `services/orion-hub/app/settings.py` (near the `CABINET_*` settings):

```python
    HUB_ENERGY_TIMEZONE: str = Field(default="America/Denver", alias="HUB_ENERGY_TIMEZONE")
```

and to `services/orion-hub/.env_example` near the cabinet keys:

```text
# Local day boundary for the Hub Energy strip's daily kWh bars.
HUB_ENERGY_TIMEZONE=America/Denver
```

Mount in `services/orion-hub/scripts/api_routes.py` next to the cabinet cooling lines:

```python
from .energy_routes import router as energy_router
```

```python
router.include_router(energy_router)
```

- [ ] **Step 5: Template** — in `services/orion-hub/templates/index.html`, directly after the closing tag of the cooling strip block (the one containing `id="cabinetCoolingWattsChart"`), add a section using the same wrapper/tile Tailwind classes the cooling strip uses (copy its outer `<section>`/`<div>` classes and tile classes exactly), with these ids and default text:

```html
<section id="energyStrip" aria-label="House electricity">
  <div>
    <h3>House electricity — Rocky Mountain Power</h3>
    <span id="energyImporterState">importer: no status yet</span>
  </div>
  <div>
    <div><div>Cycle to date</div><div id="energyCycleToDate">unknown</div></div>
    <div><div>Orion projects</div><div id="energyProjected">unknown</div></div>
    <div><div>RMP forecast</div><div id="energyForecast">unknown</div></div>
    <div><div>Next kWh</div><div id="energyMarginal">unknown</div></div>
    <div><div>Pressure</div><div id="energyPressure">unknown</div></div>
  </div>
  <p id="energyReconcileActual">No closed bill reconciled yet.</p>
  <p id="energyReconcileForecast">No RMP forecast yet.</p>
  <div id="energyDailyBars" aria-label="Daily house kWh, last 14 days"></div>
</section>
```

Next to `<script src="/static/js/cabinet-sensors.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>` add:

```html
<script src="/static/js/energy-strip.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>
```

- [ ] **Step 6: JS** — `services/orion-hub/static/js/energy-strip.js`:

```javascript
(function () {
  "use strict";

  const LATEST_URL = "/api/energy/latest";
  const DAILY_URL = "/api/energy/usage/daily?days=14";
  const POLL_MS = 60000;
  const BAR_MAX_PX = 60;
  let timer = null;

  function isNum(v) {
    return v !== null && v !== undefined && Number.isFinite(Number(v));
  }

  // Unknown money is "unknown" -- rendering it as $0.00 would say the house was free.
  function fmtUsd(v) {
    if (!isNum(v)) return "unknown";
    const n = Number(v);
    return (n < 0 ? "-$" : "$") + Math.abs(n).toFixed(2);
  }

  function fmtRate(v) {
    return isNum(v) ? "$" + Number(v).toFixed(4) + "/kWh" : "unknown";
  }

  function pressureLabel(p) {
    return (
      { over_forecast: "over RMP forecast", near_forecast: "near RMP forecast", normal: "under RMP forecast" }[p] ||
      "unknown"
    );
  }

  function importerLabel(imp) {
    if (!imp) return "importer: no status yet";
    return "importer: " + imp.state + " (" + imp.reason + ")";
  }

  function reconcileLine(kind, r) {
    if (!r) return kind === "actual" ? "No closed bill reconciled yet." : "No RMP forecast yet.";
    const label = (kind === "actual" ? "Last bill " : "RMP forecast ") + r.billing_period_start;
    if (r.reconcile_gap) return label + ": Orion can't price this period yet (" + r.reconcile_gap + ")";
    return label + ": Orion " + fmtUsd(r.orion_total_usd) + " vs RMP " + fmtUsd(r.utility_total_usd) + " (diff " + fmtUsd(r.delta_usd) + ")";
  }

  function barHeights(points, maxPx) {
    const values = points.map(function (p) { return isNum(p.kwh) ? Number(p.kwh) : 0; });
    const max = values.reduce(function (a, b) { return Math.max(a, b); }, 0);
    return values.map(function (v) { return max > 0 ? Math.round((v / max) * maxPx) : 0; });
  }

  function setText(id, text) {
    const el = typeof document !== "undefined" ? document.getElementById(id) : null;
    if (el) el.textContent = text;
  }

  function renderLatest(body) {
    const s = (body && body.stakes) || {};
    setText("energyImporterState", importerLabel(body && body.importer));
    setText("energyCycleToDate", fmtUsd(s.cycle_to_date_total_usd));
    setText("energyProjected", fmtUsd(s.orion_projected_total_usd));
    setText("energyForecast", fmtUsd(s.forecast_total_usd));
    setText("energyMarginal", fmtRate(s.marginal_usd_per_kwh));
    setText("energyPressure", pressureLabel(s.pressure));
    const rec = (body && body.reconcile) || {};
    setText("energyReconcileActual", reconcileLine("actual", rec.actual));
    setText("energyReconcileForecast", reconcileLine("forecast", rec.forecast));
  }

  function renderDaily(body) {
    const el = typeof document !== "undefined" ? document.getElementById("energyDailyBars") : null;
    if (!el) return;
    const points = (body && body.points) || [];
    const heights = barHeights(points, BAR_MAX_PX);
    el.replaceChildren();
    points.forEach(function (p, i) {
      const bar = document.createElement("div");
      bar.style.height = heights[i] + "px";
      bar.style.width = "8px";
      bar.style.background = "currentColor";
      bar.title = p.day + ": " + Number(p.kwh).toFixed(1) + " kWh (" + p.hours + " h)";
      el.appendChild(bar);
    });
  }

  async function poll() {
    try {
      const [latest, daily] = await Promise.all([
        fetch(LATEST_URL).then(function (r) { return r.json(); }),
        fetch(DAILY_URL).then(function (r) { return r.json(); }),
      ]);
      renderLatest(latest);
      renderDaily(daily);
    } catch (err) {
      setText("energyImporterState", "energy API unavailable");
    }
  }

  function activate() {
    if (timer) return;
    poll();
    timer = setInterval(poll, POLL_MS);
  }

  function deactivate() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  const api = {
    activate: activate,
    deactivate: deactivate,
    refresh: poll,
    fmtUsd: fmtUsd,
    fmtRate: fmtRate,
    pressureLabel: pressureLabel,
    importerLabel: importerLabel,
    reconcileLine: reconcileLine,
    barHeights: barHeights,
  };

  if (typeof window !== "undefined") {
    window.OrionEnergyStrip = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
```

In `services/orion-hub/static/js/biometrics-view.js`, next to each `window.OrionCabinetSensors.deactivate();` call (two places: subview switch ~L645 and modal close ~L688) add:

```javascript
    if (window.OrionEnergyStrip && typeof window.OrionEnergyStrip.deactivate === "function") {
      window.OrionEnergyStrip.deactivate();
    }
```

and inside the `name === "cabinet"` branch after `window.OrionCabinetSensors.activate();`'s `if` block:

```javascript
      if (window.OrionEnergyStrip && typeof window.OrionEnergyStrip.activate === "function") {
        window.OrionEnergyStrip.activate();
      }
```

Run `python scripts/sync_local_env_from_example.py`.

- [ ] **Step 7: Run tests**

```bash
cd services/orion-hub && PYTHONPATH=../..:. $PY -m pytest tests/test_energy_routes.py tests/test_energy_strip_panel.py tests/test_cabinet_sensors_panel.py -q
node --test static/js/energy-strip.test.js static/js/cabinet-sensors.test.js
```

Expected: all pass. (`orion-static-gates.yml` already runs every `static/js/*.test.js`.)

- [ ] **Step 8: Commit**

```bash
git add services/orion-hub/scripts/energy_routes.py services/orion-hub/scripts/api_routes.py \
  services/orion-hub/templates/index.html services/orion-hub/static/js/energy-strip.js \
  services/orion-hub/static/js/energy-strip.test.js services/orion-hub/static/js/biometrics-view.js \
  services/orion-hub/app/settings.py services/orion-hub/.env_example \
  services/orion-hub/tests/test_energy_routes.py services/orion-hub/tests/test_energy_strip_panel.py
git diff --cached --check
git commit -m "feat(hub): Energy strip with stakes, importer state, reconcile, daily kWh"
```

---

### Task 9: Reconcile eval, docs, live smoke, PR

**Files:**
- Create: `services/orion-energy/evals/test_energy_reconcile_replay_eval.py`, `docs/superpowers/pr-reports/2026-09-27-orion-energy-watcher-plan-2-pr.md`
- Modify: `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md` (Status line), `.github/workflows/orion-energy-tests.yml` (only if new paths need filters)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Eval** — `services/orion-energy/evals/test_energy_reconcile_replay_eval.py`:

```python
"""Reconcile against the REAL tariff with Plan 1's hand oracle, plus the file-drop path end to end.

The oracle is written from published Schedule 1 numbers (same as the bill-replay eval),
so reconcile cannot pass by agreeing with the tariff code.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.bills import scan_bills
from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import (
    ENERGY_RECONCILE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyUsageIntervalV1,
)

REPO = Path(__file__).resolve().parents[3]
DENVER = ZoneInfo("America/Denver")
START = datetime(2026, 9, 1, tzinfo=DENVER).astimezone(timezone.utc)
RETRIEVED = datetime(2026, 10, 2, tzinfo=timezone.utc)

MULT = (1 + (7.63 - 0.53) / 100) * (1 + (1.17 + 3.84 + 0.17) / 100)
ORACLE_ENERGY = (400 * 0.098332 + 320 * 0.125263) * MULT
ORACLE_TOTAL = ORACLE_ENERGY + 12.00 + 0.16


def _pipeline() -> EnergyPipeline:
    led = UsageLedger(
        load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"), tz=DENVER, cycle_start_day=1,
    )
    return EnergyPipeline(ledger=led, channels=EnergyChannels("u", "a", "r"), usage_point_id="UP123")


def _hours(n: int) -> list[EnergyUsageIntervalV1]:
    return [
        EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="UP123", interval_start=START + timedelta(hours=h),
            interval_end=START + timedelta(hours=h + 1), energy_kwh=1.0, retrieved_at=RETRIEVED,
        )
        for h in range(n)
    ]


def test_matching_bill_reconciles_to_zero() -> None:
    p = _pipeline()
    p.ingest_intervals(_hours(720), now=RETRIEVED)
    bill = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        kwh_billed=720.0, energy_charge=ORACLE_ENERGY, taxes=5.0, current_charges=ORACLE_TOTAL + 5.0,
        retrieved_at=RETRIEVED,
    )
    rec = [o.payload for o in p.ingest_bills([bill], now=RETRIEVED) if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.utility_basis == "pre_tax"
    assert rec.orion_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert rec.delta_usd == pytest.approx(0.0, abs=1e-6)
    assert rec.delta_kwh == pytest.approx(0.0)
    assert rec.bucket_deltas["energy_charge"] == pytest.approx(0.0, abs=1e-6)


def test_ten_flat_days_project_to_the_full_month_oracle() -> None:
    p = _pipeline()
    p.ingest_intervals(_hours(240), now=RETRIEVED)
    fc = EnergyBillForecastV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        as_of=START + timedelta(days=10), projected_total_usd=ORACLE_TOTAL, retrieved_at=START + timedelta(days=10),
    )
    rec = [o.payload for o in p.ingest_bills([fc], now=RETRIEVED) if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.orion_kwh == pytest.approx(720.0)
    assert rec.orion_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert rec.delta_usd == pytest.approx(0.0, abs=1e-6)


def test_hand_entered_bill_file_reaches_reconcile(tmp_path) -> None:
    inbox, processed = tmp_path / "bills/inbox", tmp_path / "bills/processed"
    inbox.mkdir(parents=True)
    (inbox / "sept.json").write_text(json.dumps({
        "kind": "energy.bill.actual.v1", "billing_period_start": "2026-09-01",
        "billing_period_end": "2026-10-01", "kwh_billed": 720, "current_charges": round(ORACLE_TOTAL, 2),
    }))
    p = _pipeline()
    p.ingest_intervals(_hours(720), now=RETRIEVED)
    out = p.ingest_bills(scan_bills(inbox, processed, now=RETRIEVED), now=RETRIEVED)
    rec = [o.payload for o in out if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.utility_basis == "tax_unknown"
    assert abs(rec.delta_usd) < 0.01
```

Run: `PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/evals -q`
Expected: all pass (Plan 1 eval + 3 new).

- [ ] **Step 2: Full focused gate run**

```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
python scripts/check_compose_no_relative_mounts.py
PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py orion/energy/tests -q
PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/tests services/orion-energy/evals -q
```

plus the sql-writer and Hub commands from Tasks 2, 7, 8 and `node --test services/orion-hub/static/js/energy-strip.test.js`. Record every command and result for the PR report. If a `check_*.py` script does not exist, say so in the report — do not invent it.

- [ ] **Step 3: Docker build + live smoke (orion-energy core)**

```bash
scripts/safe_docker_build.sh orion-energy config
scripts/safe_docker_build.sh orion-energy build orion-energy
scripts/safe_docker_build.sh orion-energy --profile portal build orion-energy-portal
```

Live path (requires sql-writer on this branch for the new tables — build/restart it via `scripts/safe_docker_build.sh orion-sql-writer up -d --build` only if Juniper approves touching the shared deployment; otherwise mark UNVERIFIED and print the restart commands):

```bash
scripts/safe_docker_build.sh orion-energy up -d --build orion-energy
docker logs --tail=50 ${PROJECT}-orion-energy 2>&1 | grep -E "energy_status|energy_ledger_replayed"
```

Expected evidence: an `energy_status state=... pressure=...` log line within `ENERGY_STATUS_INTERVAL_SEC`. With a real bill JSON dropped into `/mnt/storage-warm/orion-energy/bills/inbox/`: an `energy_ingested ... bills=1` line, and after sql-writer restart:

```sql
SELECT reconcile_kind, billing_period_start, orion_total_usd, utility_total_usd, delta_usd, reconcile_gap
FROM energy_reconcile ORDER BY computed_at DESC LIMIT 3;
SELECT as_of, state, reason FROM energy_importer_status ORDER BY as_of DESC LIMIT 1;
```

Portal live path is **UNVERIFIED** until Juniper runs the headed reauth (README "Portal"). Hub strip live path is UNVERIFIED until Hub is redeployed with this branch; do not restart Hub or sql-writer on the shared deployment without approval — print the commands.

- [ ] **Step 4: Spec status + PR report** — change the spec's Status line to:

```text
- **Status:** Plan 1 (cost primitive) merged (PR #2373). Plan 2 (portal, bill reconcile, stakes, Hub) implemented on `feat/orion-energy-watcher-plan-2`; portal selectors UNVERIFIED until first live reauth.
```

Write `docs/superpowers/pr-reports/2026-09-27-orion-energy-watcher-plan-2-pr.md` in the AGENTS.md §18 template shape, including: acceptance checks 1–8 from the spec with the test/eval that proves each; the §0A metric-gate note (the stakes snapshot is a projection of already-gated inputs — metered usage, versioned tariff, RMP's own forecast, importer freshness — not a new continuous signal; `pressure` thresholds are operator knobs with a pre-tax-vs-maybe-taxed bias that leans toward fewer holds); env keys added in orion-energy / sql-writer / hub; restart commands for orion-energy, orion-sql-writer, orion-hub; UNVERIFIED items (portal selectors, live Hub strip, live curiosity hold).

- [ ] **Step 5: Commit, graph refresh, push**

```bash
git add services/orion-energy/evals/test_energy_reconcile_replay_eval.py \
  docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md \
  docs/superpowers/pr-reports/2026-09-27-orion-energy-watcher-plan-2-pr.md
git diff --cached --check
git commit -m "test(energy): reconcile replay eval; docs: Plan 2 status and PR report"
scripts/safe_graphify_update.sh
git push -u origin feat/orion-energy-watcher-plan-2
```

(The controller runs the final whole-branch review, opens the PR with the report body, and watches CI.)
