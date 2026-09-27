# Orion energy watcher — house electricity as a stake

- **Date:** 2026-09-26
- **Status:** Plan 1 (cost primitive) merged (PR #2373). Plan 2 (portal, bill reconcile, stakes, Hub) implemented on `feat/orion-energy-watcher-plan-2`; portal selectors UNVERIFIED until first live reauth.
- **Branch:** Plan 1 `feat/orion-energy-watcher` (merged, PR #2373); Plan 2 `feat/orion-energy-watcher-plan-2`
- **Worktree:** Plan 1 `/mnt/scripts/Orion-Sapienform-orion-energy-watcher`; Plan 2 `/mnt/scripts/Orion-Sapienform-orion-energy-watcher-plan-2`

## Arsonist summary

Orion already meters **machine** watts (`PowerIntent` → biometrics settle → joules) and is growing **cabinet cooling** watts. What it does not have is **house-level electrical truth**: what Rocky Mountain Power billed, what the meter actually used hour by hour, and what an extra GPU hour costs **in dollars on this bill cycle**. Without that, “opportunity cost” and “attention with stakes” stay abstract — Orion can see temperature and utilization, but not the bill Juniper pays.

This design adds a thin `orion-energy` service that adapts PacifiCorp’s portal Green Button path (same stack as Pacific Power / RMP), prices intervals with a **versioned tariff**, joins settled power intents into dual labeled costs, reconciles against RMP bill/forecast, and optionally lets **one** spend gate (curiosity) feel energy as a stake — default off.

## Decisions locked

| Topic | Choice |
|---|---|
| Success criteria | **B + C**: price a Circe/GPU run in $ on this cycle; reconcile Orion estimate vs RMP bill/forecast |
| Attribution | **C**: publish both `estimated_run_cost_usd` (Orion meters × marginal rate) and `house_share_cost_usd`; autonomy may use **only** the first |
| Auth | Persistent Playwright profile + `reauth_required`; MFA stays on; **file-drop bootstrap** so pricing never depends on scraper uptime |
| Attention | Thin curiosity consumer behind `ORION_ENERGY_STAKES_ENABLED=false`; no new attention winner |
| Architecture | New `services/orion-energy/` — not folded into biometrics |
| Utility source | Adapt `nburns/pacificpower-import` Playwright + ESPI parser for `rockymountainpower.net`; drop Home Assistant half |
| Cost model | Accrue/marginal from Green Button × versioned tariff; scrape bill actual + forecast as reconciliation ground truth — do not reduce everything to flat ¢/kWh |
| Cycle usage contiguity | Cycle position is **known only for a contiguous prefix from billing-cycle start** (local midnight on `ENERGY_BILLING_CYCLE_START_DAY`). Mid-cycle holes → unknown (not zero). Accrual prices the prefix only; run-cost gap `no_cycle_usage` means missing or incomplete cycle data. Re-price pending settlements when coverage fills; expire pending after `ENERGY_RUN_COST_PENDING_HOURS`. |

## Current architecture (grounded)

### What already exists

- **`PowerIntentV1` / `PowerIntentSettledV1`** (`orion/schemas/power.py`): workloads declare expected draw; biometrics settles with joules / peak / mean watts. `None` means unknown, never coerce to zero.
- **Cabinet cooling watts** (`orion-zwave` → `home.cooling.sample.v1`): portable AC load, Hub Cabinet strip — pattern for thin home telemetry → bus → sql-writer → Hub.
- **Attention with stakes** (`docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md`): noticing is free; spending is mostly blind. P6 says don’t invent a power budget until `power_draw_log` lands — this design adds a **dollar** stake from the utility, not a fake watt budget.
- **Consequential action / power budget** (`docs/superpowers/specs/2026-08-28-consequential-action-space-and-power-budget-design.md`): first budget with a physical wall is UPS/circuit contention — complementary, not replaced.
- **Internal economy** (`docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md`): metabolic allocation still proposal-gated; **out of scope** for v1.
- **Dev `CostEstimate`**: session pricing only — not household electricity.

### What does not exist

- No Rocky Mountain Power / Green Button / ESPI ingest
- No tariff engine, bill actual/forecast events, or reconcile artifact
- No `orion-energy` service
- No consumer that prices autonomy in household dollars

### External seam (researched, not yet live-verified in this repo)

- RMP portal appears to share PacifiCorp Azure B2C stack with Pacific Power (`csapps.rockymountainpower.net`, `B2C_1A_PAC_SIGNIN`).
- `nburns/pacificpower-import`: Playwright login, Green Button ESPI XML, hourly `IntervalReading`s, 2-year backfill, daily incremental, ~24h utility lag, 3-day rolling re-fetch; portal request is client-side encrypted (UI drive intentional). MFA-disable in that project is **rejected** for Orion.
- No public RMP Green Button Connect (OAuth) CMD endpoint found for third-party registration.

Live confirmation of RMP Green Button download selectors and billing DOM remains **UNVERIFIED** until the first scraper spike.

## Missing questions (parked, not blocking the spec)

1. Exact Utah Schedule 1 block boundaries and adjustment line-items for the first `tariff_version` — operator fills from current RMP price summary when implementing the tariff YAML.
2. Which host owns the Playwright profile (Athena vs dedicated) — ops choice at deploy.
3. Whether curiosity is still the right first consumer after P1/P2 of attention-with-stakes lands — if curiosity trigger is superseded, retarget the same snapshot to whichever gate actually spends the expensive turn. Flag and snapshot stay.

## Proposed schema / API / bus

### Events (payload kinds; all carry provenance)

| Kind | Role |
|---|---|
| `energy.usage.observed.v1` | Hourly (or interval) house kWh from ESPI / file drop |
| `energy.bill.actual.v1` | Closed billing period truth |
| `energy.bill.forecast.v1` | RMP in-cycle estimate (null if portal lacks it) |
| `energy.cost.accrued.v1` | Tariff-priced interval + cycle position |
| `energy.run_cost.estimated.v1` | Dual labeled costs joined to a settled power intent |
| `energy.reconcile.v1` | Orion accrued vs forecast/actual deltas |
| `energy.stakes.snapshot.v1` | Materialized view for spend gates + Hub |
| `energy.importer.status.v1` | `healthy` / `stale` / `reauth_required` / `degraded` |

### Core fields (illustrative contracts)

**Usage**

```text
source: rockymountain_power | file_drop
usage_point_id
interval_start, interval_end, interval_seconds
energy_kwh
quality
retrieved_at
source_period
```

**Bill actual**

```text
billing_period_start, billing_period_end
kwh_billed
energy_charge, customer_charge, adjustments, fees, taxes, credits
current_charges, amount_due, due_date
statement_artifact_id?   # optional PDF later
retrieved_at
```

**Bill forecast**

```text
as_of
days_into_cycle
projected_kwh
projected_total_usd
retrieved_at
```

**Accrued**

```text
interval_start, interval_end
energy_kwh
marginal_usd_per_kwh
interval_cost_usd
cycle_accumulated_kwh
cycle_estimated_total_usd
tariff_version
```

**Run cost (never collapse the two USD fields)**

```text
intent_id, workload_kind, node, gpu_index?
window_start, window_end
energy_joules?            # null if settlement blind
estimated_run_cost_usd?   # Orion meters × marginal — autonomy may use
house_share_cost_usd?     # context only; null if no overlapping house interval
tariff_version
correlation_id?
```

**Hard rule:** unknown / missing / `no_samples` → leave USD **null**. Never coerce to `0.0`.

### Channels

Register under `orion/bus/channels.yaml` with matching `schema_id`s in `orion/schemas/registry.py`:

```text
orion:energy:usage:observed
orion:energy:bill:actual
orion:energy:bill:forecast
orion:energy:cost:accrued
orion:energy:run_cost:estimated
orion:energy:reconcile
orion:energy:stakes:snapshot
orion:energy:importer:status
```

### Persistence

`orion-sql-writer` tables mirroring the event shapes (usage intervals unique on usage_point + interval_start; late re-fetch upserts by newer `retrieved_at`). Hub read APIs for latest snapshot, usage history, reconcile history.

### HTTP / ops

- Health: importer status + last successful fetch age
- Optional: Hub upload endpoint or watched drop directory for ESPI XML / bill artifacts
- No credentials on the bus; browser profile path + secrets via env only

## Service shape

```text
Rocky Mountain Power portal          File drop (XML / bill)
        │                                    │
        │ Playwright (persistent profile)    │
        ▼                                    ▼
              services/orion-energy/
              ├── espi parser
              ├── billing scrape (history + forecast if present)
              ├── tariff engine (versioned YAML/config)
              ├── run-cost join (consumes PowerIntentSettled)
              ├── reconcile
              └── publish bus + status
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  sql-writer      stakes snapshot   curiosity gate
  + Hub Energy         │            (flag default off)
                       └── held_off:energy_stakes rows
```

### Ingest behavior

- Daily incremental + **3-day rolling re-fetch** (late AMI intervals).
- ~24h utility lag treated as normal; past that → `stale`.
- Session death → `reauth_required`; operator re-auths in headed browser once; no MFA-disable design; no credential replay storm.
- File-drop uses the **same** parsers and event kinds (`source=file_drop`).

### Tariff + reconciliation

- Deterministic Schedule 1–style seasonal blocks + named adjustments in versioned config.
- Each interval priced at **marginal** rate given cycle kWh position.
- Reconcile Orion cycle estimate vs `bill.forecast` (in-cycle) and `bill.actual` (after close); emit deltas by fee bucket where possible.
- Fix systematic miss via **tariff/config patches**, not a second latent model.

### Attention consumer (v1)

- Materialize `energy.stakes.snapshot.v1`.
- Curiosity investigation path only, behind `ORION_ENERGY_STAKES_ENABLED` (default `false`).
- When on: may hold discretionary runs when cycle estimate is near/above forecast, or expensive seasonal block + weak expected value; use `estimated_run_cost_usd` only.
- Every hold leaves `held_off:energy_stakes` (inspectable). Flag off → curiosity behavior unchanged.

## Files likely to touch (implementation)

```text
docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md   # this file
services/orion-energy/                                             # new service
orion/schemas/energy*.py                                           # contracts
orion/bus/channels.yaml
orion/schemas/registry.py
services/orion-sql-writer/app/models/ + route map
services/orion-hub/                                                # Energy strip + optional upload
services/orion-hub/scripts/curiosity_investigation.py              # flag-gated stakes read
config/energy/tariff.*.yaml                                        # versioned rates
tests/ + services/orion-energy/tests/ + evals/
vendor or docs note: attribution to nburns/pacificpower-import
```

## Non-goals

- Public Green Button Connect My Data OAuth registration (no usable public CMD endpoint found)
- Disabling MFA / putting session cookies on Postgres or the bus
- Porting the Home Assistant half of `pacificpower-import`
- Fusing run cost and house-share into one number
- Rewriting the motor allocator or internal-economy metabolic budget in this patch
- PDF OCR as the primary bill path (history scrape first; PDF optional later)
- Claiming Orion “feels” cost — only that a spend gate can see a real dollar stake
- Keyword / phrase triggers on user chat about bills or money

## Acceptance checks

1. Green Button XML (portal or file drop) → hourly rows in Postgres + `energy.usage.observed` on the bus.
2. New intervals update cycle accrued $; `tariff_version` recorded on accrued and run-cost events.
3. After `PowerIntentSettled` with joules: `estimated_run_cost_usd` non-null; after `no_samples` / null joules: USD fields null (not zero).
4. `house_share_cost_usd` only when overlapping house interval exists; never substituted for (3) in autonomy.
5. Bill actual and/or forecast ingested; `energy.reconcile` shows Orion vs RMP delta.
6. Importer can enter `reauth_required` / `stale` / `degraded` visibly; stale path does not emit fabricated $0 usage.
7. Flag off: curiosity unchanged in tests. Flag on: fixture produces at least one `held_off:energy_stakes` decision row.
8. Metric/signal gate: any new continuous signal wired into attention/cognition must pass CLAUDE.md §0A metric quality gate before further consumers build on it (stakes snapshot is a projection of already-gated inputs, not a new biometric).

## Risks / concerns

| Severity | Concern | Mitigation |
|---|---|---|
| High | Portal UI / encryption changes break Playwright | File-drop path; status `reauth_required`/`degraded`; no silent success on empty |
| Med | Whole-house meter ≠ Circe | Dual labeled costs; autonomy uses meter-side only |
| Med | Tariff incompleteness vs real bill | Reconcile artifact; versioned config patches |
| Low | Curiosity ceases to be the expensive turn | Snapshot stays; retarget consumer without redesigning ingest |

## Recommended next patch

Implementation is **two plans** (this spec is one contract; do not one-shot all eight steps):

**Plan 1 — cost primitive (unblocks B without portal):**
1. Schemas + channels + sql-writer tables + fixtures.
2. File-drop ESPI path.
3. Tariff engine + `energy.cost.accrued` + run-cost join on `PowerIntentSettled`.
4. Debug surface: Postgres tables + README queries (Hub strip is Plan 2).

**Plan 2 — portal + reconcile + stakes (C + attention):**
5. Playwright RMP adapter + importer status / reauth.
6. Bill/forecast scrape + reconcile.
7. Stakes snapshot + curiosity flag (default off).
8. Hub Energy strip.

## How to disable / roll back

- Stop `orion-energy` compose service; unsubscribe sql-writer routes if needed.
- `ORION_ENERGY_STAKES_ENABLED=false` restores curiosity immediately.
- Tariff/config and schemas are additive; dropping the service leaves historical SQL rows inert.

## Trace that proves it worked

- Bus: usage / accrued / run_cost / reconcile / importer.status envelopes with correlation ids.
- Postgres: interval upsert after 3-day re-fetch; run_cost row linked to `intent_id`.
- Hub: Energy strip shows accrued vs forecast and importer state.
- Curiosity: with flag on, a hold row `held_off:energy_stakes` in the existing decision/log surface.

## Related specs

- `docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md`
- `docs/superpowers/specs/2026-08-28-consequential-action-space-and-power-budget-design.md`
- `docs/superpowers/specs/2026-08-30-power-intent-prior-design.md`
- `docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md`
- `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md` (thin home telemetry pattern)

## External reference

- [nburns/pacificpower-import](https://github.com/nburns/pacificpower-import) — PacifiCorp Green Button Playwright + ESPI parser (adapt; do not vendor HA output).
