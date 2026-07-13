# Autonomy Origination Measurement Gate

`measure_autonomy_gate.py` is a **Step-0, read-only** measurement instrument. It
answers two empirical questions from durable history and prints two deterministic
GO / NO-GO verdicts *before* any downstream cognition code is written.

It **writes nothing to the substrate, emits no events, and flips no flags.** It
is a measurement, not a cognition change.

## Why this gate exists

Two downstream specs are cathedrals if the questions below fail, so we measure
first:

- **Verdict (a) — endogenous drift.** Does `SelfStateV1` actually drift during
  exogenous silence? Gates the **endogenous-drive-origination spec (Step 1)**
  (`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`).
  If self-state is flat when no receipts/turns arrive, spontaneous origination
  has no signal to fire on.
- **Verdict (b) — internal economy.** How often do ≥2 drives co-activate, and
  does `resource_pressure` actually rise? Gates the
  **internal-economy-scarcity-allocation spec (Step 4)**
  (`docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md`).
  If drives rarely co-activate or resource pressure sits near zero, the scarcity
  allocator never binds and is ornamental.

## How to run

All commands below use `scripts/analysis/...` paths **relative to the repo
root**, so run them from the repo/worktree root.

On this host, use the repo venv (system `python` lacks `psycopg2`) with the host
env overrides:

```bash
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
AUTONOMY_GRAPH_QUERY_URL=http://localhost:3030/orion/query \
/mnt/scripts/Orion-Sapienform/venv/bin/python scripts/analysis/measure_autonomy_gate.py --window-days 7
```

Generic / in-container form (defaults resolve on the docker `app-net` network):

```bash
python scripts/analysis/measure_autonomy_gate.py --window-days 7
# or a sub-day window:
python scripts/analysis/measure_autonomy_gate.py --window-hours 1
```

- `--window-days` (int) and `--window-hours` (float) are mutually exclusive.
  Neither given → defaults to `7` days (`DEFAULT_WINDOW_DAYS`). The window is
  `now - window` through `now`.
- Within the window, activity is bucketed into fixed `300`-second buckets
  (`WINDOW_SEC`) to classify each interval as `silent` or `busy`.
- `psycopg2` is required for live DB metrics. If it is unavailable or the DB is
  unreachable, `open_readonly_connection` degrades gracefully (returns `None`,
  adds a caveat, and self-state/receipt metrics are empty) — it does not crash.
- **Exit code**: `0` for a completed measurement (either verdict came back `GO`
  or `NO-GO`); `2` if either verdict is `UNMEASURABLE` (see below) — a caller
  (cron, CI, a human) can branch on this without parsing the report text.

### Environment variables

`run()` reads two env vars (both have in-container defaults):

| Var | Default (docker `app-net`) | Purpose |
| --- | --- | --- |
| `POSTGRES_URI` | `postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney` | read-only Postgres session |
| `AUTONOMY_GRAPH_QUERY_URL` | `http://orion-athena-fuseki:3030/orion/query` | Fuseki SPARQL query endpoint |

When running **from the host** (outside the docker `app-net` network), override
both to the published ports (`localhost:55432` for Postgres, `localhost:3030`
for Fuseki), as shown in the host command above.

## Data sources

- **`substrate_self_state`** — `SelfStateV1` rows (`self_state_json`,
  `generated_at`). Supplies `dimension_trajectory`, per-dimension scores,
  `trajectory_condition`, `overall_surprise`, and `resource_pressure`.
- **`substrate_reduction_receipts`** — receipt `created_at` timestamps. This is
  the "exogenous input happened" signal used to classify each bucket as
  `silent` (no receipts and no turns) or `busy`.
- **Fuseki drives graph** `http://conjourney.net/graph/autonomy/drives` — read
  via a read-only SPARQL `SELECT`. `orion-rdf-writer` persists every
  `DriveAudit` with the active signal at the assessment level:
  `DriveAudit --orion:hasDriveAssessment--> DriveAssessment --orion:driveActive-> true`,
  so an audit's active-drive count == number of its assessments with
  `driveActive = true`. (An older `orion:highlightsActiveDrive` projection is
  equivalent — one triple per active drive — but the assessment-level boolean is
  the primary schema.) The query aggregates this **server-side** into a
  co-activation histogram via a nested `GROUP BY` (inner: per-audit `SUM`; outer:
  bin by active-count), so Fuseki returns only ~7 rows regardless of window size
  rather than transferring hundreds of thousands of per-audit rows (which timed
  out on large windows). The bus audit channel is redis pub/sub with no
  replayable history, which is why the durable Fuseki time-series is queried
  instead.

### Caveats

- **Turn timestamps are currently unavailable** (`turn_count=0`): no turn store
  is wired into this measurement, so bucket classification uses receipts only.
  This is reported as a coverage caveat in the output.
- Every adapter degrades gracefully: missing DB, unreachable Fuseki, or absent
  columns produce empty metrics plus a caveat rather than a crash.
- **Retention-bound caveat**: if the requested window reaches further back than
  a source's actual retention (e.g. `substrate_self_state` currently retains
  only ~3 days live while `substrate_reduction_receipts` retains ~10), the
  report names which source is the binding constraint and by how many hours —
  older buckets in an over-long window reflect *zero rows*, not measured
  absence, and this caveat is how the report says so explicitly instead of
  silently under-covering.

## Verdict rules

Thresholds are the **seed decision boundary**. The report always prints the raw
numbers so a human can override the mechanical verdict. Both verdicts can
return a third value, **`UNMEASURABLE`**, distinct from `GO`/`NO-GO` — see
below.

### (a) Endogenous drift — `verdict_drift`

**UNMEASURABLE** iff both silent and busy self-state row counts are `0` (dead
or unreachable Postgres source — nothing was measured, not measured-flat).

Otherwise, **GO** iff, in **silent** buckets:

- `median_abs_trajectory >= DRIFT_MIN_MEDIAN_ABS_TRAJECTORY` (**0.03**), **AND**
- silent `dim_score_variance >= DRIFT_VARIANCE_RATIO` (**0.25**) × busy
  `dim_score_variance`. When busy variance is `0`, the ratio test passes iff
  silent variance `> 0`.

**NO-GO** means self-state is input-following / flat; the endogenous spec must
change its dynamics source before it is built.

### (b) Internal economy — `verdict_economy`

**UNMEASURABLE** iff **either** input is empty: zero `DriveAudit` records in
the requested window (dead/unreachable Fuseki, or a source that stopped being
written to — see the 2026-07-13 re-run in
`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`
for a real example: RDF materialization for `DriveAudit` was disabled
2026-06-19, and the previous instrument silently reported this as a
behavioral `NO-GO` for three weeks before anyone noticed), **or** zero
self-state rows (dead/unreachable Postgres — the source `pressure` is built
from). Both inputs are checked independently; guarding only one would leave
the other free to silently degrade to `0.0` and produce a real-looking
`NO-GO` string.

Otherwise, **GO** iff:

- `coactivation_frac >= COACTIVATION_MIN_FRAC` (**0.10**), **AND**
- fraction of self-state rows with `resource_pressure >= RESOURCE_PRESSURE_LEVEL`
  (**0.3**) is `>= RESOURCE_PRESSURE_MIN_FRAC` (**0.05**).

**NO-GO** means the economy allocator would be a cathedral at current
activation / pressure levels — do not build.

**`UNMEASURABLE` must never be treated as `NO-GO`** by anything reading this
gate's output — it means the instrument didn't get real data, not that the
answer is no.

## Outputs & monitoring

No data snapshot is required (read-only). All artifacts land under
`/tmp/autonomy-gate/`:

- `report.md` — the full report (also printed to stdout), with both verdicts and
  the raw numbers behind each.
- `before_after.csv` — silent-vs-busy self-state metrics.
- `progress.log` — streaming progress; each line carries event title, percent
  done, rows processed / total, rate, and anomaly count.

Monitor a run with:

```bash
tail -f /tmp/autonomy-gate/progress.log
```

## Read-only guarantee

`open_readonly_connection` issues `SET default_transaction_read_only = on`, then
verifies the session is read-only via `SHOW default_transaction_read_only` and
**refuses to run if it is not `on`**. Drive-audit reads use a SPARQL `SELECT`
only (no `UPDATE`). No runtime service, schema, channel, env, or substrate row is
modified.
