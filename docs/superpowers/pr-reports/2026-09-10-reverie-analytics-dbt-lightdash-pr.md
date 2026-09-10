# Reverie analytics with dbt Core and Lightdash OSS

## Status

`DONE_WITH_CONCERNS`: the implementation, isolated database rail, and local
Lightdash runtime are verified. Production role/model installation and the
human-owned Lightdash first-admin/project/content upload are intentionally not
performed.

## Summary

- Added a self-hosted analytics service using dbt-postgres 1.9.0 and Lightdash
  OSS 2.184.6.
- Modeled one narrow subject, persisted reverie chains, as a staging view, fact
  view, terminal-reason dimension, and gap-free UTC date dimension.
- Added one governed distinct-chain metric, explicit join cardinality, a
  starter dashboard, deterministic tests, and a read-only reconciliation eval.
- Added dedicated transformer and reader roles without widening Orion's
  existing `orion_readonly` role.
- Excluded theme keys, narrative JSON, proposal IDs, and salience from the mart.

## Outcome moved

Orion now has a small, inspectable analytical seam from a real operational
event table to documented dbt views and version-controlled Lightdash content.
It does not infer success, failure, service uptime, or cognitive quality from
terminal reasons or missing rows.

## Current architecture before

- PostgreSQL service: `services/orion-sql-db`, live database `conjourney`.
- Source relation: `public.substrate_reverie_chain`.
- Producer: `services/orion-thought/app/chain.py::run_reverie_chain` constructs
  the record after its loop; `app/store.py::persist_reverie_chain` inserts it.
- No dbt project, analytical schema, Lightdash deployment, or governed BI
  metric existed.
- `mind_runs` was rejected as the first subject because the inspected live rows
  were degenerate across trigger, status, error, and router profile.

## Architecture touched

```text
public.substrate_reverie_chain
  -> analytics.stg_reverie_chains
  -> analytics.fct_reverie_chains
     + analytics.dim_reverie_outcomes
     + analytics.dim_reverie_dates
  -> Lightdash semantic Explore
  -> version-controlled charts/dashboard
```

Lightdash metadata uses an internal-only PostgreSQL container. Required
S3-compatible object storage uses pinned MinIO containers. Lightdash and MinIO
bind to localhost by default. The warehouse reader cannot read the operational
source.

## Files changed

- `services/orion-analytics/`: service config, dbt project, models, tests, eval,
  role bootstrap, Lightdash content, and operator documentation.
- `docs/superpowers/plans/2026-09-10-reverie-analytics-dbt-lightdash.md`:
  architecture and metric-quality gate.
- `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.json`, and
  `graphify-out/manifest.json`: required full worktree graph rebuild.

## Schema, bus, and API

- No bus channel, shared event schema, application API, or operational table
  changed.
- New PostgreSQL views live only in the fixed `analytics` schema.
- Fact grain: one row per persisted `chain_id`.
- Metric: distinct count of `reverie_chain_id`.
- Joins: fact-to-outcome and fact-to-date are explicit many-to-one joins;
  date-to-fact is one-to-many for the zero-day chart.

## Env and config

- Added `services/orion-analytics/.env_example`; local ignored `.env` was synced.
- Passwords and connection location are environment-driven.
- `analytics`, `orion_analytics_transformer`, and `orion_analytics_reader` are
  fixed security-policy identifiers, preventing bootstrap/profile drift.
- Anonymous dbt and Lightdash analytics are disabled.
- `ORION_BUS_URL` remains the required root value; this analytics service does
  not consume the bus.

## Tests

- Contract tests: 6 passed.
- dbt compile: 4 models compiled.
- dbt run: 4 views created in the disposable PostgreSQL fixture.
- dbt test: 37 passed, including keys, accepted values, relationships, source
  reconciliation, date-spine continuity, and join fanout.
- dbt docs generate: passed.
- Lightdash 2.184.6 JSON schemas: 3 dbt YAML files, 3 chart files, and 1
  dashboard file passed.
- `git diff --check`: passed.

## Evals

The isolated read-only reconciliation returned:

```text
source_rows=4
fact_rows=4
distinct_fact_ids=4
joined_rows=4
source_fact_delta=0
duplicate_fact_delta=0
join_fanout_delta=0
zero_activity_days=7
```

The reader authenticated with `default_transaction_read_only=on`, could read
all 4 fact rows, could not read the source, and a create-table probe was
rejected.

## Docker, build, and smoke

- dbt image: `ghcr.io/dbt-labs/dbt-postgres:1.9.0`.
- Lightdash image: `lightdash/lightdash:2.184.6`.
- Metadata PostgreSQL: `postgres:15.14-alpine`, internal network only.
- MinIO and bucket-init images are pinned to dated releases.
- Compose validation passed through `scripts/safe_docker_build.sh`.
- Lightdash health reported version 2.184.6, no pending migration, and disabled
  RudderStack configuration.
- Published Lightdash port verified as `127.0.0.1:8265->8080`.

## Review findings fixed

- Finding: `created_at` was described as chain-start time.
  - Fix: documented the actual post-loop, pre-persistence record time.
  - Evidence: producer inspection and updated source/model docs.
- Finding: Lightdash initially bound its first-admin surface to all interfaces.
  - Fix: localhost default with an explicit, documented deployment override.
  - Evidence: Docker published-port smoke.
- Finding: a fact-time dashboard filter removed null date-spine rows.
  - Fix: mapped the missing-days tile to the date dimension field.
  - Evidence: exact-version dashboard schema validation and regression test.
- Finding: reused analytics roles could retain elevated attributes, memberships,
  grants, or ownership.
  - Fix: demotion, ACL/membership cleanup, and atomic fail-closed audits.
  - Evidence: adversarial disposable-DB probes rejected an undeclared
    transformer-owned table and a reader-owned analytics table.
- Finding: schema and principal env overrides could drift from bootstrap policy.
  - Fix: made all three identifiers fixed and added a contract test.
  - Evidence: env parity and independent review PASS.

## Restart required

No existing Orion service restart is required. After operator bootstrap and
password configuration:

```bash
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt run
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt test
scripts/safe_docker_build.sh orion-analytics --profile analytics up -d
```

## Risks and concerns

- Production roles and views are `UNVERIFIED`; bootstrap was exercised only in
  an isolated PostgreSQL container.
- Lightdash first-admin registration, project creation, semantic upload,
  rendered dashboard behavior, and generated-SQL inspection are `UNVERIFIED`
  because they require a human account/project/token.
- Only persisted chains are visible. Refractory suppression and pre-insert
  failures leave no row, and a missing day does not prove downtime.
- `refractory` and `low_salience` are code-declared dimension members but were
  absent from the inspected live source rows.
- The CLI requires Node.js 24 or newer.

## PR link

None. Per request, this worktree is not committed or pushed and no PR was
created.
